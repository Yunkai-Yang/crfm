from typing import Union, List, Optional

from diffusers.utils import logging

import torch

logger = logging.get_logger(__name__)

def _get_t5_prompt_embeds(
        text_encoder,
        tokenizer,
        tokenizer_max_length,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ):
        device = text_encoder.device
        dtype = text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds #[B,256,4096]

def _get_clip_prompt_embeds(
        text_encoder,
        tokenizer,
        tokenizer_max_length,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
    ):
        device = text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2] 
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

@torch.no_grad
def encode_prompt(
    prompt,
    clip_tokenizer_list,
    clip_text_encoder_list,
    t5_tokenizer,
    t5_text_encoder,
    device = None,
    max_sequence_length: int = 256,
    num_images_per_prompt = 1
):
    device = device or t5_text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = prompt
    prompt_3 = prompt

    clip_text_encoder_1, clip_text_encoder_2 = clip_text_encoder_list
    clip_tokenizer_1, clip_tokenizer_2 = clip_tokenizer_list
    tokenizer_max_length = clip_tokenizer_1.model_max_length

    prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
        prompt=prompt,
        text_encoder=clip_text_encoder_1,
        tokenizer=clip_tokenizer_1,
        tokenizer_max_length=tokenizer_max_length,
    )
    prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
        prompt=prompt_2,
        text_encoder=clip_text_encoder_2,
        tokenizer=clip_tokenizer_2,
        tokenizer_max_length=tokenizer_max_length,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
    t5_prompt_embed = _get_t5_prompt_embeds(
        prompt=prompt_3,
        max_sequence_length=max_sequence_length,
        text_encoder=t5_text_encoder,
        tokenizer=t5_tokenizer,
        tokenizer_max_length=tokenizer_max_length
    )  
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2) 
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
    
    return prompt_embeds, pooled_prompt_embeds


def load_sd3_text_processer(ckpt_path, device, weight_dtype):
    from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast

    clip_tokenizer_list = [
        CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer"),
        CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer_2"),
    ]
    clip_text_encoder_list = [
        CLIPTextModelWithProjection.from_pretrained(ckpt_path, subfolder="text_encoder").to(device, dtype=weight_dtype),
        CLIPTextModelWithProjection.from_pretrained(ckpt_path, subfolder="text_encoder_2").to(device, dtype=weight_dtype),
    ]
    t5_tokenizer = T5TokenizerFast.from_pretrained(ckpt_path, subfolder="tokenizer_3")
    t5_text_encoder = T5EncoderModel.from_pretrained(ckpt_path, subfolder="text_encoder_3").to(device, dtype=weight_dtype)
    return clip_tokenizer_list, t5_tokenizer, clip_text_encoder_list, t5_text_encoder
