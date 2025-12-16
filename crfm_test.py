import os
import logging
import argparse, json

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers import SD3Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from safetensors.torch import load_file
from mmseg.apis import init_model

from src.models.sd3_mmdit import MaskDit_sd3_5
from src.utils.utils import encode_images
from src.utils.crfm import inference_with_crfm
from src.datasets.infer_dataset import SegmentationDataset, collate_fn

import torch
import transformers
import diffusers

# If you are using RTX 40x series GPUs, you need to add the following lines.
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'

logger = get_logger(__name__, log_level="INFO")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="testing script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="",)
    parser.add_argument("--pretrained_mmdit",type=str,default=None)
    parser.add_argument("--lora_ckpt",type=str,default="")
    parser.add_argument("--data_root",type=str,default="")
    parser.add_argument("--json_file",type=str,default="")
    parser.add_argument("--vectors_path",type=str,default="")
    parser.add_argument("--work_dir",type=str,default="output/test_result")
    parser.add_argument("--cache_dir",type=str,default="cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers",type=int,default=0,)
    parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed running: local_rank")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--num_cls", type=int, default=100)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--datameta", type=str, default=None,)
    parser.add_argument("--mmseg_config",type=str,default='')
    parser.add_argument("--mmseg_ckpt",type=str,default='')
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--rectified_step", type=int, default=4)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.revision = None
    args.variant = None
    pretrained_ckpt = args.pretrained_model_name_or_path
    args.transfomer_config = os.path.join(pretrained_ckpt, "transformer", "config.json")
    args.updated_mmdit = os.path.join(args.lora_ckpt, "model.safetensors")

    if args.datameta is not None:
        with open(os.path.join(args.data_root, args.datameta), 'r', encoding='utf8') as f:
            args.num_cls = json.load(f)['num_cls']
    return args

def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    transformer = MaskDit_sd3_5(sd3_transformer=transformer).to(accelerator.device, dtype=weight_dtype)

    # Load MMDiT checkpoint
    model_state_dict = load_file(args.updated_mmdit)
    transformer.load_state_dict(
        model_state_dict, strict=False
    )
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    ).to(accelerator.device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    control_model = init_model(args.mmseg_config, checkpoint=args.mmseg_ckpt).to(accelerator.device, dtype=weight_dtype)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    control_model.requires_grad_(False)
    control_model.eval()

    val_dataset = SegmentationDataset(
        data_root=args.data_root,
        txt_file=args.json_file,
        vae_scale_factor=vae_scale_factor,
        vectors_path=args.vectors_path,
        size=args.resolution,
        num_cls=args.num_cls,
        debug=args.debug,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    transformer, val_dataloader = accelerator.prepare(
        transformer, val_dataloader
    )

    num_inference_steps = args.num_inference_steps
    rectified_step = args.rectified_step

    for idx, batch in enumerate(val_dataloader):
        if idx < args.skip:
            continue

        pooled_prompt_embeds = batch['pooled_prompt_embeds'].to(accelerator.device)
        prompt_embeds = batch['prompt_embeds'].to(accelerator.device)
            
        cond_types = batch['cond_types']
        cond_latents = batch['cond_latents']
        condition_dict = {
            "cond_types": [],
            "cond_latents": [],
        }
        with torch.no_grad():
            for i, (cdtn_type, img_per_cdtn) in enumerate(zip(cond_types, cond_latents)):
                img_per_cdtn = img_per_cdtn.to(accelerator.device, dtype=weight_dtype)
                img_per_cdtn = encode_images(vae, img_per_cdtn, weight_dtype)

                condition_dict["cond_types"].append(cdtn_type)
                condition_dict["cond_latents"].append(img_per_cdtn)
            
            result_img_list = inference_with_crfm(
                transformer=transformer,
                vae=vae,
                scheduler=noise_scheduler,
                image_processor=image_processor,
                conditional_model=control_model,
                condition_targets=batch['control_condtions'].to(accelerator.device),
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                condition_dict=condition_dict,
                width=args.resolution,
                height=args.resolution,
                max_step_size=0.1,
                rectified_step=rectified_step,
                ignore_index=255,
            ).images

            if args.debug:
                from PIL import Image

                generated_result = result_img_list[0]
                new = Image.new('RGB', (generated_result.width * 3, generated_result.height))
                new.paste(generated_result, (0, 0))
                new.paste(batch['label_PIL'][0], (generated_result.width, 0))
                new.paste(batch['target_PIL'][0], (generated_result.width * 2, 0))
                new.save('test.jpg')

            break

if __name__ == "__main__":
    args = parse_args()
    main(args)
