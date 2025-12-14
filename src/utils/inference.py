from typing import List, Optional, Union, Dict, Any
from tqdm import tqdm

import torch

from .utils import _prepare_latents, retrieve_timesteps, calculate_shift

@torch.no_grad
def batch_imgage_generation(
    transformer,
    vae,
    scheduler,
    image_processor,
    prompt_embeds,
    pooled_prompt_embeds,
    condition_dict = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    height: int = 512,
    width: int = 512,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    sigmas: Optional[List[float]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    mu: Optional[float] = None,
    guidance_scale: float = 4.5,
    initial_latents: Optional[torch.Tensor] = None
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    device = prompt_embeds.device
    lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    )

    # Prepare latent variables
    batch_size = prompt_embeds.shape[0]
    latents_width = int(width / vae_scale_factor)
    latents_height = int(height / vae_scale_factor)
    num_channels_latents = transformer._config.in_channels
    
    if initial_latents is None:
        latents = _prepare_latents(
            batch_size,
            num_channels_latents,
            latents_height,
            latents_width,
            prompt_embeds.dtype,
            device,
            generator,
        )
    else:
        latents = initial_latents
    latent_dtype = latents.dtype

    # Prepare timesteps
    scheduler_kwargs = {}
    if scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // transformer.config.patch_size) * (
            width // transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    use_classifier_free_guidance = False
    if negative_prompt_embeds is not None:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        use_classifier_free_guidance = True

    with tqdm(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if use_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])
            
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
                condition_dict=condition_dict,
            )[0]

            if use_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents != latent_dtype:
                latents = latents.to(dtype=latent_dtype)
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type=output_type)

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)