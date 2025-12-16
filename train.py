from datetime import datetime
from packaging import version

import os
import json, math
import argparse
import copy
import logging
import functools

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.import_utils import is_xformers_available
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from peft import LoraConfig
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel
)

import torch
import diffusers
import torch.utils.checkpoint
import transformers

from src.utils.utils import (
    encode_images,
    load_transformer_config
)
from src.utils.hook import save_lora_adapter_hook
from src.models.sd3_mmdit import MaskDit_sd3_5
from src.models.modules.sd3_mm_block import MMDiTBlock
from src.datasets.pretrain_dataset import SegmentationDataset, collate_fn

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
logger = get_logger(__name__, log_level="INFO")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument( "--pretrained_model_name_or_path", type=str, default="", required=True)
    parser.add_argument("--work_dir", type=str, default="work_dir",)
    parser.add_argument("--pretrained_mmdit", type=str, default=None,)
    parser.add_argument("--sub_work_dir", type=str, default="")
    parser.add_argument("--data_root",type=str,default="")
    parser.add_argument("--train_file",type=str,default="")
    parser.add_argument("--vectors_path",type=str,default="")
    parser.add_argument("--num_cls", type=int, default=100,)
    parser.add_argument("--datameta", type=str, default=None,)
    parser.add_argument("--checkpointing_steps",type=int,default=4000,)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--mixed_precision",type=str,default="bf16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--cache_dir",type=str,default="cache",)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=20000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--scale_lr",action="store_true",default=False,)
    parser.add_argument("--lr_scheduler",type=str,default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500,)
    parser.add_argument("--weighting_scheme",type=str,default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument("--dataloader_num_workers",type=int,default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True)

    args = parser.parse_args()
    args.revision = None
    args.variant = None
    if len(args.sub_work_dir) > 0:
        args.work_dir = os.path.join(args.work_dir, args.sub_work_dir)
    args.work_dir = os.path.join(args.work_dir, datetime.now().strftime('%y_%m_%d-%H:%M'))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.datameta is not None:
        with open(os.path.join(args.data_root, args.datameta), 'r', encoding='utf8') as f:
            args.num_cls = json.load(f)['num_cls']
    pretrained_ckpt = args.pretrained_model_name_or_path
    args.transfomer_config = os.path.join(pretrained_ckpt, "transformer", "config.json")
    return args

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    if args.pretrained_mmdit is None:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = MaskDit_sd3_5(sd3_transformer=transformer).to(accelerator.device, dtype=weight_dtype)
    else:
        sd3_config = load_transformer_config(args.transfomer_config)
        transformer = SD3Transformer2DModel(**sd3_config)
        transformer = MaskDit_sd3_5(sd3_transformer=transformer).to(accelerator.device, dtype=weight_dtype)
        model_state_dict = load_file(args.pretrained_mmdit)
        transformer.load_state_dict(model_state_dict, strict=True)
    
    # transformer.requires_grad_(False)
    vae.requires_grad_(False)
    logger.info("All models keeps requires_grad = False")

    logger.info(f"Trainable lora is set successfully")

    # Enable mask, denoising branch gradient to update.
    # Freeze text branch.
    transformer.switch_mask_branch(True)
    transformer.switch_denoising_branch(True)
    transformer.switch_text_branch(False)

    # hook, not having resume from ckpt temporarily!
    accelerator.register_save_state_pre_hook(
        functools.partial(
            save_lora_adapter_hook,
            accelerator=accelerator,
        )
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    optimizer_cls = torch.optim.AdamW

    full_ft_params = []
    for k, p in transformer.named_parameters():
        if p.requires_grad:
                full_ft_params.append(p)

    optimizer = optimizer_cls(
        [
            {"params": full_ft_params, 'lr': args.learning_rate},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Optimizer initialized successfully.")

    # Preprocessing the datasets.
    train_dataset = SegmentationDataset(
        data_root=args.data_root,
        txt_file=args.train_file,
        vectors_path=args.vectors_path,
        size=args.resolution,
        num_cls=args.num_cls,
        vae_scale_factor=vae_scale_factor,
    )
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("Training dataset and Dataloader initialized successfully.")

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    logger.info(f"lr_scheduler:{args.lr_scheduler} initialized successfully.")

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("MaskDiT", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.work_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                pooled_prompt_embeds = batch['pooled_prompt_embeds'].to(accelerator.device, dtype=weight_dtype)
                prompt_embeds = batch['prompt_embeds'].to(accelerator.device, dtype=weight_dtype)
                latent_image = encode_images(
                    pixels=batch["pixel_values"].to(accelerator.device),
                    vae=vae,
                    weight_dtype=weight_dtype
                )

                # Condition preprocess
                condition_dict = batch['condition_dict']
                for i, cond_type in enumerate(condition_dict['cond_types']):
                    cond_latents = condition_dict['cond_latents'][i]
                    cond_latents = encode_images(
                        pixels=cond_latents.to(accelerator.device),
                        vae=vae,
                        weight_dtype=weight_dtype
                    )
                    condition_dict['cond_latents'][i] = cond_latents

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latent_image)
                bsz = latent_image.shape[0]

                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latent_image.ndim, dtype=latent_image.dtype)
                noisy_model_input = (1.0 - sigmas) * latent_image + sigmas * noise

            with accelerator.accumulate(transformer):
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs={},
                    condition_dict=condition_dict,
                    return_dict=False,
                )[0]
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                # flow matching loss
                target = noise - latent_image

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.work_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)