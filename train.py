import os
import argparse
from typing import Callable, Sequence

from torch import nn
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from stun.model.base_model import BaseModel
from stun.model import configuration
from stun.model.stun import Stun
from stun.data_module import PBVS2022DataModule
import stun.augmentations as A
from stun.augmentations.pytorch import ToTensor
from stun.utils import split_train_valid_id, seed_everything


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--use_sch', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    args = parser.parse_args()
    return args


def get_train_transforms(height: int, width: int, scale: int) -> Sequence[Callable]:
    return A.Compose([
        # A.RandomCrop(height, width, scale=scale, p=1.0),
        A.Pad(scale=scale, min_height=height, min_width=width),
        A.HorizontalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Normalize(),
        ToTensor(),
    ])


def get_valid_transforms(height: int, width: int, scale: int) -> Sequence[Callable]:
    return A.Compose([
        # A.RandomCrop(height, width, scale=scale, p=1.0),
        A.Pad(scale=scale, min_height=height, min_width=width),
        A.Normalize(),
        ToTensor(),
    ])


def train_base(
    args: argparse.Namespace,
    seed: int = 36,
) -> None:
    model_name = args.model_name

    train_ids, val_ids = split_train_valid_id(
        'data/new_train/640_flir_hr_bicubicnoise',
        seed=seed
    )

    train_transforms = get_train_transforms(height=160, width=160, scale=4)
    val_transforms = get_valid_transforms(height=160, width=160, scale=4)

    data_module = PBVS2022DataModule(
        train_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        # train_input_dir='data/new_train/320_axis_mr',
        valid_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        # valid_input_dir='data/new_train/320_axis_mr',
        predict_input_dir='data/test/evaluation1/hr_x4',
        # predict_input_dir='data/test/evaluation2/mr_real',
        train_label_dir='data/new_train/640_flir_hr',
        valid_label_dir='data/new_train/640_flir_hr',
        train_image_ids=train_ids,
        valid_image_ids=val_ids,
        train_transforms=train_transforms,
        valid_transforms=val_transforms,
    )

    cfg = configuration.__dict__[f'{model_name}_config']()
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

    if args.use_sch:
        model_name += '-sch'

    if args.use_swa:
        weight_averaging = StochasticWeightAveraging()
        callbacks.append(weight_averaging)
        model_name += '-swa'

    ckpt_path = os.path.join(args.save_dir, 'weights', model_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    checkpoint = ModelCheckpoint(
        monitor='val_psnr',
        dirpath=ckpt_path,
        filename='{epoch}-{val_psnr:.4f}-{val_ssim:.4f}',
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

    train_base(args, seed=seed)


if __name__ == '__main__':
    main()
