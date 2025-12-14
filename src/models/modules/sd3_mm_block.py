from typing import Optional, Dict, Any

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormZero, RMSNorm, FP32LayerNorm, LpNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

import torch.nn.functional as F
import torch

@maybe_allow_in_graph
class MMDiTAttn(nn.Module):
    def __init__(
        self,
        query_dim: int,
        sd3_pretrained_attn: Attention = None,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        if cross_attention_norm is None:
            self.norm_cross = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None
        
        self.sd3_attn = sd3_pretrained_attn

    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        condition_dict = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        attn1 = self.sd3_attn
        attn2 = self
        batch_size, seq_len, _ = hidden_states.shape

        query = attn1.to_q(hidden_states)
        key = attn1.to_k(hidden_states)
        value = attn1.to_v(hidden_states)

        inner_dim = key.shape[-1]
        n_heads = attn1.heads
        head_dim = inner_dim // n_heads
        
        query = query.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        if attn1.norm_q is not None:
            query = attn1.norm_q(query)
        if attn1.norm_k is not None:
            key = attn1.norm_k(key)

        # txt_len = 0
        if encoder_hidden_states is not None:
            # _, txt_len, _ = encoder_hidden_states.shape
            encoder_hidden_states_query_proj = attn1.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn1.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn1.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, n_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, n_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, n_heads, head_dim
            ).transpose(1, 2)

            if attn1.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn1.norm_added_q(encoder_hidden_states_query_proj)
            if attn1.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn1.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        
        if condition_dict is not None:
            for i, cond_type in enumerate(condition_dict['cond_types']):
                cond_latents = condition_dict['cond_latents'][i]
                
                cond_query = attn2.to_q(cond_latents)
                cond_key = attn2.to_k(cond_latents)
                cond_value = attn2.to_v(cond_latents)

                cond_query = cond_query.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
                cond_key = cond_key.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
                cond_value = cond_value.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

                if attn2.norm_q is not None:
                    cond_query = attn2.norm_q(cond_query)
                if attn2.norm_k is not None:
                    cond_key = attn2.norm_k(cond_key)
                
                query = torch.cat([query, cond_query], dim=2)
                key = torch.cat([key, cond_key], dim=2)
                value = torch.cat([value, cond_value], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, n_heads * head_dim)

        if condition_dict is not None:
            for i in range(len(condition_dict['cond_types']) - 1, -1, -1):
                cond_len = condition_dict['cond_latents'][i].shape[1]
                hidden_states, cond_latents = (
                    hidden_states[:, :-cond_len],
                    hidden_states[:, -cond_len:],
                )
                # linear proj
                cond_latents = attn2.to_out[0](cond_latents)
                # dropout
                cond_latents = attn2.to_out[1](cond_latents)
                condition_dict['cond_latents'][i] = cond_latents
        
        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : seq_len],
                hidden_states[:, seq_len :],
            )
            if not attn1.context_pre_only:
                encoder_hidden_states = attn1.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn1.to_out[0](hidden_states)
        # dropout
        hidden_states = attn1.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states, condition_dict
        else:
            return hidden_states, condition_dict

@maybe_allow_in_graph
class MMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        sd3_pretrained_block = None,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.attn = MMDiTAttn(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=None,
            qk_norm=qk_norm,
            eps=1e-6,
            sd3_pretrained_attn=sd3_pretrained_block.attn
        )
        sd3_pretrained_block.attn = None

        if sd3_pretrained_block.attn2 is not None:
            self.attn2 = sd3_pretrained_block.attn2
            sd3_pretrained_block.attn2 = None
        else:
            self.attn2 = None

        self.sd3_block = sd3_pretrained_block

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb: torch.FloatTensor,
        condition_dict = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        block1 = self.sd3_block
        block2 = self
        joint_attention_kwargs = joint_attention_kwargs or {}

        if block1.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = block1.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block1.norm1(hidden_states, emb=temb)
        if block1.context_pre_only:
            norm_encoder_hidden_states = block1.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block1.norm1_context(
                encoder_hidden_states, emb=temb
            )

        condition_tmp_list = []
        if condition_dict is not None:
            for i, cond_type in enumerate(condition_dict['cond_types']):
                cond_latents = condition_dict['cond_latents'][i]
                norm_cond_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = block2.norm1(cond_latents, emb=temb)
                condition_tmp_list.append(
                    [cond_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp]
                )
                condition_dict['cond_latents'][i] = norm_cond_latents

        # MM-Attn
        attn_output, context_attn_output, condition_dict = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            condition_dict=condition_dict,
            attention_mask=attention_mask,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if block1.use_dual_attention:
            attn_output2 = self.attn2(
                hidden_states=norm_hidden_states2,
                **joint_attention_kwargs
            )
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2
        
        norm_hidden_states = block1.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = block1.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if block1.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output
            
            norm_encoder_hidden_states = block1.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = block1.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        # Process attention outputs for the `condition control`.
        if condition_dict is not None:
            for i, cond_type in enumerate(condition_dict['cond_types']):
                residual_cond_latents = condition_dict['cond_latents'][i]
                cond_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = condition_tmp_list[i]
                residual_cond_latents = cond_gate_msa.unsqueeze(1) * residual_cond_latents
                cond_latents = cond_latents + residual_cond_latents
                
                norm_cond_latents = block2.norm2(cond_latents)
                norm_cond_latents = norm_cond_latents * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                cond_ff_output = block2.ff(norm_cond_latents)
                cond_ff_output = gate_mlp.unsqueeze(1) * cond_ff_output
                
                condition_dict['cond_latents'][i] = cond_latents + cond_ff_output

        return hidden_states, encoder_hidden_states, condition_dict