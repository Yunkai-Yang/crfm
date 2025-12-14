import numpy as np
import json
from torch.utils.data import Dataset
from safetensors.torch import load_file
from PIL import Image
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from diffusers.image_processor import VaeImageProcessor
import torch

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root,
        txt_file,
        vae_scale_factor=8,
        size=None,
        vectors_path=None,
        debug=False,
        num_cls=None
    ):
        assert vectors_path is not None, 'you must vectorize your prompts, before you train mmdit'
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2 ,do_resize=True, do_convert_rgb=True)
        self.data_root = data_root
        self.data_paths = txt_file
        self.data = []
        with open(self.data_paths, "r") as f:
            for index, line in enumerate(f):
                item = json.loads(line)
                item['index'] = index
                self.data.append(item)
        self._length = len(self.data)
        self.size = size
        self.debug = debug
        
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2 ,do_resize=True, do_convert_rgb=True)
        self.msk_factor = 255.0 / float(127 if num_cls is None else num_cls)

        self.vectors_path = vectors_path

    def __len__(self):
        return self._length
    
    def load_image(self, pth):
        return Image.open(pth).convert('RGB')
    
    def __getitem__(self, i):
        item = self.data[i]
        index = item['index']
        
        txt_vectors = load_file(os.path.join(self.vectors_path, f'{index}.safetensors'))

        lbl_path = os.path.join(self.data_root, item['source'])
        image_name = os.path.basename(item['target'])

        # process label
        label_ = Image.open(lbl_path)
        label = np.array(label_).astype(np.float32) * self.msk_factor
        label = Image.fromarray(label.astype(np.uint8)).convert('RGB')

        label_condition = self.image_processor.preprocess(label, width=self.size, height=self.size).squeeze(0)

        if self.debug:
            target = self.load_image(os.path.join(self.data_root, item['target']))
            return {
                **txt_vectors,
                "condition_latents": [label_condition],
                "condition_types":['mask'],
                "label_PIL": label,
                "target_PIL": target,
                "control_condtion": torch.from_numpy(np.array(label_)).long(),
                }
        else:
            return {
                **txt_vectors,
                "condition_latents": [label_condition],
                "condition_types":['mask'],
                "img_name": image_name,
                "control_condtion": torch.from_numpy(np.array(label_)).long(),
                }

def collate_fn(examples):
    pooled_prompt_embeds = torch.stack([example['pooled_prompt_embeds'].squeeze(0) for example in examples])
    pooled_prompt_embeds = pooled_prompt_embeds.to(memory_format=torch.contiguous_format)
    prompt_embeds = torch.stack([example['prompt_embeds'].squeeze(0) for example in examples])
    prompt_embeds = prompt_embeds.to(memory_format=torch.contiguous_format)

    condition_types= examples[0]["condition_types"]
    condition_latents = [[] for _ in range(len(condition_types))]
    for example in examples:
        conditions = example['condition_latents']
        for i, cdtn in enumerate(conditions):
            condition_latents[i].append(cdtn)
    
    for i in range(len(condition_latents)):
        cdtn_i = torch.stack(condition_latents[i], dim=0)
        condition_latents[i] = cdtn_i.to(memory_format=torch.contiguous_format).float()

    if 'label_PIL' in examples[0]:
        return {
            "cond_latents": condition_latents,
            "cond_types": condition_types,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "label_PIL": [elem['label_PIL'] for elem in examples],
            "target_PIL": [elem['target_PIL'] for elem in examples],
        }
    else:
        return {
            "cond_latents": condition_latents,
            "cond_types": condition_types,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "img_names": [example['img_name'] for example in examples]
        }