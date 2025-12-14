# Control Rectified Flow Matching(CRFM)

<br>

> **Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation**
> `<br>`
> Yunkai Yang, [Yudong Zhang](https://yudongzhang.com/), Kunquan Zhang, Jinxiao Zhang, Xinying Chen, Haohuan Fu, [Runmin Dong](https://dongrunmin.github.io)
> `<br>`
> Sun Yat-sen University & Tsinghua University

<br>

<p align="center">
  <img src="assets/visulization.jpg" width="90%" height="90%">
</p>

## Overview

<p align="center">
  <img src="assets/overview.png" width="90%" height="90%">
</p>

(a) The overall workflow of the task-oriented data synthesis framework (TODSynth) consists of three stages: (a) Training stage using an MM-DiT generative model conditioned on text and mask. (b) Sampling stage with the proposed control-rectify flow matching (CRFM). (c) Downstream tasks trained on a combination of real and synthetic data.

## Environment setup

```bash
conda create -n crfm python=3.11
conda activate crfm
pip install -r requirements.txt
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" 
pip3 install "mmsegmentation>=1.0.0"
```

### Optional
Then modify the loading logic of the `load_from_local` function located in `mmengine/runner/checkpoint.py` as:
```python
def load_from_local(filename, map_location):
    """Load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint
```

## Download Models

1. **Stable Diffusion 3.5**

```bash
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./sd3.5_medium
```

## Dataset preparation

## Model inference

```bash
```

## Model training

```bash
```