from typing import List, Optional, Union, Dict, Any

from tqdm import tqdm
from torch import nn
from torchvision.transforms.functional import normalize
from diffusers.image_processor import VaeImageProcessor

from .utils import (
    _prepare_latents,
    retrieve_timesteps,
    calculate_shift,
    get_sigmas
)

import torch

def control_rf_matching(
    vector_field_preds: torch.Tensor,
    timestep_latents,
    conditional_model,
    image_processor: VaeImageProcessor,
    conditions,
    vae,
    sigmas,
    weight_dtype,
    step_size: float = 0.1,
    ignore_index: int = 255,
    l2_norm: bool = True,
    is_synth: bool = True,
    return_pre_synth: bool = False
):
    pre_synth = None
    conditional_model.eval()
    vector_field_preds = vector_field_preds.detach().requires_grad_(True)
    latent_preds = timestep_latents - sigmas * vector_field_preds
    latent_preds = (latent_preds / vae.config.scaling_factor) + vae.config.shift_factor
    latent_preds = latent_preds.to(dtype=weight_dtype)
    image_preds = vae.decode(latent_preds, return_dict=False)[0]
    
    if return_pre_synth:
        pre_synth = image_processor.postprocess(
            image_preds.detach().to(dtype=torch.float32).cpu(),
            output_type='pil',
        )

    image_preds = image_processor.postprocess(image_preds, output_type='pt').clamp(0, 1)
    image_preds = normalize(image_preds, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(dtype=weight_dtype)
    label_preds = conditional_model.predict(image_preds)
    label_preds = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in label_preds])
    conditional_loss = nn.functional.cross_entropy(label_preds, conditions, ignore_index=ignore_index, reduction='none')
    valid_mask = ((conditions != 255) * (label_preds.argmax(1) != conditions)).float()
    conditional_loss = (valid_mask * conditional_loss).sum(dim=(-1,-2)) / (valid_mask.sum(dim=(-1,-2)) + 1e-10)
    conditional_loss.mean().backward()
    
    with torch.no_grad():
        cond_grad = vector_field_preds.grad 
        
        if l2_norm:
            grad_norm = torch.norm(cond_grad.view(cond_grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            cond_grad = cond_grad / (grad_norm + 1e-10) 
        if is_synth:
            vector_field_preds *= (1 - step_size)

        vector_field_preds -= (step_size * cond_grad)
        vector_field_preds.clamp_(-1, 1)
    
    vector_field_preds.grad.zero_()

    return vector_field_preds.detach().requires_grad_(False), pre_synth

def inference_with_crfm(
    transformer,
    conditional_model,
    vae,
    scheduler,
    image_processor: VaeImageProcessor,
    prompt_embeds,
    pooled_prompt_embeds,
    condition_dict = None,
    condition_targets: torch.Tensor = None,
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
    ignore_index: int = 255,
    max_step_size: float = 0.2,
    l2_norm: bool = False,
    step_scheme: str = 'LD', # LD: Linear Decrease; LI: Linear Increase; C: Constant
    num_rectified: int = 1,
    rectified_step: int = 7,
    is_synth: bool = False,
    initial_latents: Optional[torch.Tensor] = None,
    return_pre_synth: bool = False,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    device = prompt_embeds.device
    lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    )

    pre_synths = []
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
            
            with torch.no_grad():
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
            
            if i < rectified_step:
                sigmas = get_sigmas(scheduler=scheduler, timesteps=timestep, device=device, weight_dtype=latent_dtype, n_dim=latents.ndim)
                
                if step_scheme == 'C':
                    # 1. Set step size as a constant
                    step_size = max_step_size
                elif step_scheme == 'LI':
                    # 2. Set step size to increase linearly
                    step_size = max_step_size * ((i + 1) / num_inference_steps)
                elif step_scheme == 'LD':
                    # 3. Set step size to decrease linearly
                    step_size = max_step_size * (1 - ((i + 1) / num_inference_steps))
                
                for _ in range(num_rectified):
                    noise_pred, pre_synth = control_rf_matching(
                        vector_field_preds=noise_pred,
                        timestep_latents=latents,
                        conditional_model=conditional_model,
                        conditions=condition_targets,
                        image_processor=image_processor,
                        vae=vae,
                        sigmas=sigmas,
                        weight_dtype=latent_dtype,
                        ignore_index=ignore_index,
                        step_size=step_size,
                        l2_norm=l2_norm,
                        is_synth=is_synth,
                        return_pre_synth=return_pre_synth,
                    )
                    if pre_synth is not None:
                        pre_synths.append(pre_synth)
            
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents != latent_dtype:
                latents = latents.to(dtype=latent_dtype)
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                progress_bar.update()
    with torch.no_grad():
        if output_type == "latent":
            image = latents

        else:
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)
    if return_pre_synth:
        return StableDiffusion3PipelineOutput(images=image), pre_synths
    else:
        return StableDiffusion3PipelineOutput(images=image)