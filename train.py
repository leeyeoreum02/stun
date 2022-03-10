import os
import random
import argparse
from typing import Callable, Sequence

import numpy as np

import torch
from torch import nn
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from stun.model.base_model import BaseModel
from stun.model.configuration import StunX4TConfig
from stun.model.stun import Stun
from stun.data_module import PBVS2022DataModule
import stun.augmentations as A
from stun.augmentations.pytorch import ToTensor
from stun.utils import split_train_valid_id


def seed_everything(seed: int = 36) -> None:
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--use_sch', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    args = parser.parse_args()
    return args


def get_train_transforms(height: int, width: int) -> Sequence[Callable]:
    return A.Compose([
        A.RandomCrop(height, width, scale=4, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Normalize(),
        ToTensor(),
    ])


def get_valid_transforms(height: int, width: int) -> Sequence[Callable]:
    return A.Compose([
        A.RandomCrop(height, width, scale=4, p=1.0),
        A.Normalize(),
        ToTensor(),
    ])


def train_base(
    model_name: str,
    args: argparse.Namespace,
    seed: int = 36,
) -> None:
    train_ids, val_ids = split_train_valid_id(
        'data/new_train/640_flir_hr_bicubicnoise',
        seed=seed
    )

    train_transforms = get_train_transforms(height=96, width=96)
    val_transforms = get_valid_transforms(height=96, width=96)

    data_module = PBVS2022DataModule(
        train_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        valid_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        predict_input_dir='data/test/evaluation1/hr_x4',
        train_label_dir='data/new_train/640_flir_hr',
        valid_label_dir='data/new_train/640_flir_hr',
        train_image_ids=train_ids,
        valid_image_ids=val_ids,
        train_transforms=train_transforms,
        valid_transforms=val_transforms,
    )

    if args.use_sch:
        model_name += '-sch'

    cfg = StunX4TConfig()
    model = Stun(cfg)
    criterion = nn.L1Loss()
    model = BaseModel(
        model,
        criterion,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        use_sch=args.use_sch,
    )

    progress_bar = RichProgressBar()
    callbacks = [progress_bar]

    if args.use_swa:
        weight_averaging = StochasticWeightAveraging()
        callbacks.append(weight_averaging)
        model_name += '-swa'

    ckpt_path = os.path.join('weights', model_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    checkpoint = ModelCheckpoint(
        monitor='final_val_psnr',
        dirpath=ckpt_path,
        filename='{epoch}-{final_val_psnr:.4f}',
        save_top_k=5,
        mode='max',
        save_weights_only=True,
    )
    callbacks.append(checkpoint)

    gpus = list(map(int, args.gpus.split(',')))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=gpus,
        # strategy=DDPPlugin(find_unused_parameters=False),
        strategy=DDPPlugin(find_unused_parameters=True),
        # precision=16,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)


def main() -> None:
    seed = 36
    seed_everything(seed)

    args = get_args()

    train_base('stunx4t', args, seed=seed)


if __name__ == '__main__':
    main()
