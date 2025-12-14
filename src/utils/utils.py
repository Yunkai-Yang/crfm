from typing import List, Optional, Union, Dict, Any
from contextlib import contextmanager

import inspect
import json

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

import torch

def encode_images(vae: torch.nn.Module, pixels: torch.Tensor, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

def _prepare_latents(
    batch_size, num_channels_latents, height, width, dtype, device, generator,
):
    shape = (
        batch_size,
        num_channels_latents,
        int(height),
        int(width),
    )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_sigmas(scheduler, timesteps, device, n_dim=4, weight_dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=device, dtype=weight_dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def load_transformer_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if "_class_name" in config:
        del config['_class_name']
    if "_diffusers_version" in config:
        del config["_diffusers_version"]

    return config

def get_msk_bran_lora_modules(sd3_dit):
    msk_bran_lora_modules = []

    for i in range(len(sd3_dit.mmdit_blocks)):
        msk_bran_lora_modules.extend([
            f'mmdit_blocks.{i}.norm1.linear',
            f'mmdit_blocks.{i}.ff.net.2',
            f'mmdit_blocks.{i}.attn.to_q',
            f'mmdit_blocks.{i}.attn.to_k',
            f'mmdit_blocks.{i}.attn.to_v',
            f'mmdit_blocks.{i}.attn.to_out.0',
        ])
    return msk_bran_lora_modules

def get_denoising_bran_lora_modules(sd3_dit):
    denoising_bran_lora_modules = ["sd3_transformer.proj_out"]

    for i, block in enumerate(sd3_dit.mmdit_blocks):
        denoising_bran_lora_modules.extend([
            f'mmdit_blocks.{i}.norm1.linear',
            f'mmdit_blocks.{i}.ff.net.2',
            f'mmdit_blocks.{i}.attn.to_q',
            f'mmdit_blocks.{i}.attn.to_k',
            f'mmdit_blocks.{i}.attn.to_v',
            f'mmdit_blocks.{i}.attn.to_out.0',
        ])
        if block.attn2 is not None:
            denoising_bran_lora_modules.extend([
                f'mmdit_blocks.{i}.attn2.to_q',
                f'mmdit_blocks.{i}.attn2.to_k',
                f'mmdit_blocks.{i}.attn2.to_v',
                f'mmdit_blocks.{i}.attn2.to_out.0',
            ])
    return denoising_bran_lora_modules

@contextmanager
def preserve_requires_grad(model):
    requires_grad_backup = {name: param.requires_grad for name, param in model.named_parameters()}
    yield

    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_backup[name]

def get_sd3_lora_modules(sd3_dit):
    denoising_bran_lora_modules = ["proj_out"]

    for i, block in enumerate(sd3_dit.transformer_blocks):
        denoising_bran_lora_modules.extend([
            f'transformer_blocks.{i}.sd3_block.norm1.linear',
            f'transformer_blocks.{i}.sd3_block.ff.net.2',
            f'transformer_blocks.{i}.attn.sd3_attn.to_q',
            f'transformer_blocks.{i}.attn.sd3_attn.to_k',
            f'transformer_blocks.{i}.attn.sd3_attn.to_v',
            f'transformer_blocks.{i}.attn.sd3_attn.to_out.0',
        ])
        if block.attn2 is not None:
            denoising_bran_lora_modules.extend([
                f'transformer_blocks.{i}.attn2.to_q',
                f'transformer_blocks.{i}.attn2.to_k',
                f'transformer_blocks.{i}.attn2.to_v',
                f'transformer_blocks.{i}.attn2.to_out.0',
            ])
    return denoising_bran_lora_modules