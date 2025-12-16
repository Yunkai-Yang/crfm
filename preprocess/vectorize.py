import argparse
import json
import os, sys
current_dir = os.path.dirname(__file__)
parrent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.abspath(parrent_dir))
os.chdir(parrent_dir)

from tqdm import tqdm
from safetensors.torch import save_file
from src.utils.vectorize import load_sd3_text_processer, encode_prompt


if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument( "--pretrained_model_name_or_path",type=str,default="pretrained_models/sd3-5_medium")
    parser.add_argument("--mixed_precision",type=str,default="bf16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--max_sequence_length",type=int,default=256, help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--work_dir",type=str,default="vectors")
    parser.add_argument("--dataset",type=str,default="demo")
    parser.add_argument("--subset",type=str,default="")
    parser.add_argument("--data_root",type=str,default="")
    parser.add_argument("--src_json_file",type=str,default="index.jsonl")
    parser.add_argument("--out_json_file",type=str,default="index_.jsonl")
    parser.add_argument("--device",type=int,default=0)
    args = parser.parse_args()
    args.revision = None
    args.variant = None

    if args.subset == "":
        out_dir = os.path.join(args.work_dir, args.dataset)
    else:
        out_dir = os.path.join(args.work_dir, args.dataset, args.subset)
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    clip_tokenizer_list, t5_tokenizer, clip_text_encoder_list, t5_text_encoder = load_sd3_text_processer(
        args.pretrained_model_name_or_path, device=device, weight_dtype=weight_dtype
    )

    with open(os.path.join(args.data_root, args.out_json_file), 'w', encoding='utf-8') as writer:
        with open(os.path.join(args.data_root, args.src_json_file), "rt") as reader:
            for idx, line in tqdm(enumerate(reader)):
                item = json.loads(line)
                item['index'] = idx

                try:
                    prompt = item['prompts']
                except:
                    prompt = item['prompt']
                    del item['prompt']
                    item['prompts'] = prompt

                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    prompt=prompt,
                    clip_text_encoder_list=clip_text_encoder_list,
                    clip_tokenizer_list=clip_tokenizer_list,
                    t5_tokenizer=t5_tokenizer,
                    t5_text_encoder=t5_text_encoder,
                    max_sequence_length=args.max_sequence_length,
                )
                vectors_dict = {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    }
                save_file(vectors_dict, os.path.join(out_dir, f'{idx}.safetensors'))
                json_string = json.dumps(
                    item,
                    ensure_ascii=False
                )
                writer.write(json_string + "\n")