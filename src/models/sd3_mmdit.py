from typing import Union, Optional, List, Any, Dict

import copy

from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.loaders import FromOriginalModelMixin, SD3Transformer2DLoadersMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import PatchEmbed
from torch import nn, Tensor, LongTensor

import torch

from .modules.sd3_mm_block import MMDiTBlock

logger = logging.get_logger(__name__)

class MaskDit_sd3_5(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    SD3Transformer2DLoadersMixin
):
    def __init__(
        self,
        sd3_transformer: SD3Transformer2DModel,
        **args,
    ):
        super().__init__()
        num_layers = len(sd3_transformer.transformer_blocks)
        sd3_config = sd3_transformer.config
        attention_head_dim = sd3_config.attention_head_dim
        num_attention_heads = sd3_config.num_attention_heads
        pos_embed_max_size = sd3_config.pos_embed_max_size
        qk_norm = sd3_config.qk_norm
        sample_size = sd3_config.sample_size
        patch_size = sd3_config.patch_size
        in_channels = sd3_config.in_channels

        self.inner_dim = num_attention_heads * attention_head_dim
        self.n_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self._config = sd3_config

        self.mmdit_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=i == num_layers - 1,
                        qk_norm=qk_norm,
                        use_dual_attention=False,
                        sd3_pretrained_block=sd3_transformer.transformer_blocks[i],
                ) for i in range(num_layers)
            ])
        sd3_transformer.transformer_blocks = None
        self.msk_pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.sd3_transformer = sd3_transformer

    @property
    def config(self):
        return self._config

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        pooled_projections: Tensor = None,
        timestep: LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        condition_dict: Optional[Dict[str, List[Tensor]]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        transformer = self.sd3_transformer
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(transformer, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        height, width = hidden_states.shape[-2:]
        hidden_states = transformer.pos_embed(hidden_states)
        temb = transformer.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

        if condition_dict is not None:
            condition_dict = copy.deepcopy(condition_dict)
            for i, cond_type in enumerate(condition_dict['cond_types']):
                if cond_type == "mask": 
                    cond_latents = condition_dict['cond_latents'][i]
                    condition_dict['cond_latents'][i] = self.msk_pos_embed(cond_latents)

        # MMDiT Block Forward!
        for mmdit_block in self.mmdit_blocks:
            hidden_states, encoder_hidden_states, condition_dict = mmdit_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                condition_dict=condition_dict,
            )

        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)

        patch_size = transformer.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], transformer.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(transformer, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def switch_mask_branch(self, grad_stat: bool = False):
        total_frozen_params = 0

        frozen_modules = ['msk_pos_embed']
        for module_name in frozen_modules:
            module = getattr(self, module_name)
            for param in module.parameters():
                param.requires_grad_(grad_stat)
                total_frozen_params += param.numel()
        
        # Freeze mask branch module in transformer blocks
        frozen_modules_1 = ["norm1", "norm2", "ff"]
        frozen_modules_2 = ["to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k"]
        for mmdit_block in self.mmdit_blocks:
            for module_name in frozen_modules_1:
                module = getattr(mmdit_block, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()
            
            # Freeze attn qkv module in mask branch
            attn = mmdit_block.attn
            for module_name in frozen_modules_2:
                module = getattr(attn, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()
        
        print(f"A total of {total_frozen_params / (1024 * 1024):.2f} M params' grad has been switched as {grad_stat}!")

    def switch_text_branch(self, grad_stat: bool = False):
        total_frozen_params = 0
        # freeze text branch module in SD3
        freeze_modules = ['time_text_embed', 'context_embedder']
        for module_name in freeze_modules:
            module = getattr(self.sd3_transformer, module_name)
            for param in module.parameters():
                param.requires_grad_(grad_stat)
                total_frozen_params += param.numel()

        # freeze text branch module in transformer blocks
        freeze_modules_1 = ["norm1_context", "norm2_context", "ff_context"]
        freeze_modules_2 = ["add_k_proj", "add_v_proj", "add_q_proj", "to_add_out", "norm_added_q", "norm_added_k"]
        for mmdit_block in self.mmdit_blocks:
            transformer_block = mmdit_block.sd3_block
            assert transformer_block is not None
            # freeze norm layer
            for module_name in freeze_modules_1:
                module = getattr(transformer_block, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()
            
            # freeze attn qkv module in text branch
            attn = mmdit_block.attn.sd3_attn
            for module_name in freeze_modules_2:
                module = getattr(attn, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()

        print(f"A total of {total_frozen_params / (1024 * 1024):.2f} M params' grad has been switched as {grad_stat}!")

    def switch_denoising_branch(self, grad_stat: bool = False):
        total_frozen_params = 0
        # freeze denoising branch module in SD3
        freeze_modules = ['pos_embed']
        for module_name in freeze_modules:
            module = getattr(self.sd3_transformer, module_name)
            for param in module.parameters():
                param.requires_grad_(grad_stat)
                total_frozen_params += param.numel()
        
        # Freeze denoising branch module in transformer blocks
        freeze_modules_1 = ["norm1", "norm2", "ff"]
        freeze_modules_2 = ["to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k"]
        
        for mmdit_block in self.mmdit_blocks:
            transformer_block = mmdit_block.sd3_block
            assert transformer_block is not None
            # freeze norm layer
            for module_name in freeze_modules_1:
                module = getattr(transformer_block, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()

            # freeze attn qkv module in denoising branch
            attn = mmdit_block.attn.sd3_attn
            for module_name in freeze_modules_2:
                module = getattr(attn, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(grad_stat)
                        total_frozen_params += param.numel()
            
            if mmdit_block.attn2 is not None:
                attn2 = mmdit_block.attn2
                for module_name in freeze_modules_2:
                    module = getattr(attn2, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad_(grad_stat)
                            total_frozen_params += param.numel()

        print(f"A total of {total_frozen_params / (1024 * 1024):.2f} M params' grad has been switched as {grad_stat}!")