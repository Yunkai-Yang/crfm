import numpy as np
import random
import json
from torch.utils.data import Dataset
from safetensors.torch import load_file
from PIL import Image
import os
from diffusers.image_processor import VaeImageProcessor
import torch

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root,
        txt_file,
        vae_scale_factor=8,
        size=None,
        flip_p=0.5,
        vectors_path=None,
        num_cls=None
    ):
        assert vectors_path is not None, 'you must vectorize your prompts, before you train mmdit'
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2 ,do_resize=True, do_convert_rgb=True)
        self.data_root = data_root
        self.data_paths = txt_file
        self.data = []
        with open(self.data_paths, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self._length = len(self.data)
        self.size = size
        self.flip_p = flip_p
        self.augment_p = 0.5
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

        path = os.path.join(self.data_root, item['target'])
        lbl_path = os.path.join(self.data_root, item['source'])

        target = self.load_image(path)
      
        # process label
        label = self.load_image(lbl_path)
        label = np.array(label).astype(np.float32) * self.msk_factor
        label = Image.fromarray(label.astype(np.uint8)).convert('RGB')

        flip = random.random() < self.flip_p
        if flip:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        pixel_values = self.image_processor.preprocess(target, width=self.size, height=self.size).squeeze(0)
        label_condition = self.image_processor.preprocess(label, width=self.size, height=self.size).squeeze(0)

        return {
            **txt_vectors,
            "pixel_values": pixel_values,
            "cond_latents": [label_condition],
            "cond_types":['mask'],
        }

def collate_fn(examples):
    pooled_prompt_embeds = torch.stack([example['pooled_prompt_embeds'].squeeze(0) for example in examples])
    pooled_prompt_embeds = pooled_prompt_embeds.to(memory_format=torch.contiguous_format)
    prompt_embeds = torch.stack([example['prompt_embeds'].squeeze(0) for example in examples])
    prompt_embeds = prompt_embeds.to(memory_format=torch.contiguous_format)

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    condition_types= examples[0]["cond_types"]
    condition_latents = [[] for _ in range(len(condition_types))]
    for example in examples:
        conditions = example['cond_latents']
        for i, cdtn in enumerate(conditions):
            condition_latents[i].append(cdtn)
    
    for i in range(len(condition_latents)):
        cdtn_i = torch.stack(condition_latents[i], dim=0)
        condition_latents[i] = cdtn_i.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_embeds": prompt_embeds,
        "condition_dict": {
            "cond_latents": condition_latents,
            "cond_types": condition_types,
        },
    }