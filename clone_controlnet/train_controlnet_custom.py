import argparse
import contextlib
import gc
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import(
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler
)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

'''
argument는 그대로 사용
train 로직에 주석을 한글로 표시할 것


'''

def parse_args():
    parser = argparse.ArgumentParser(description="Customize train controlnet script")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="...",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="...",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="...",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="...",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="...",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help='...'
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="..."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="..."
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=int,
        default=None,
        help="...",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="...",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_ture",
        help="...",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="...",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="...",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="..."
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="..."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="when cosine_with_restarts schduler"
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default= 1.0,
        help="power factor of polynomial scheduler"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="the number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="for adam optimizer"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="for adam optimizer"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="for adam optimizer"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="for adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="max gradient norm"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="push hub?"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="the name of repo to keep sync local `output_dir`"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="for tensorboard"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="on Ampere GPUs, use tf32, for speed up training?"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="`tensorboard`, `wandb`, `comet_ml` or `all`"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="bf16 needs Ampere GPU"
    )
    

    
    return args

def main(args):
    ...

if __name__ == "__main__":
    args = parse_args()
    main(args)