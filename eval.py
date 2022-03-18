import os
import argparse
from typing import Callable, List, Sequence

import cv2
import numpy as np

import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from stun.model.base_model import BaseModel
from stun.model import configuration
from stun.model.stun import Stun
from stun.data_module import PBVS2022DataModule
import stun.augmentations as A
from stun.augmentations.pytorch import ToTensor
from stun.utils import seed_everything


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluating Model')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    return args


def get_predict_transforms() -> Sequence[Callable]:
    return A.Compose([
        A.Normalize(),
        ToTensor(),
    ])


def get_submission(
    outputs: List[Tensor],
    task: str,
    save_dir: os.PathLike,
) -> None:
    if task == 'x2':
        subdir = 'evaluation2'
        file_head = 'ev2'
    elif task == 'x4':
        subdir = 'evaluation1'
        file_head = 'ev1'

    save_path = os.path.join(save_dir, subdir, task)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    outputs = torch.concat(outputs).cpu().detach().numpy()

    for filename, output in enumerate(outputs, start=1):
        output = output.transpose(1, 2, 0)
        output = np.concatenate([output * 255.0] * 3, axis=-1).astype(np.uint8)
        print(output)
        cv2.imwrite(os.path.join(save_path, f'{file_head}_{str(filename).zfill(3)}.jpg'), output)


def eval(
    args: argparse.Namespace,
    task: str,
    save_dir: os.PathLike = 'submissions',
) -> None:
    predict_transforms = get_predict_transforms()

    data_module = PBVS2022DataModule(
        train_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        # train_input_dir='data/new_train/320_axis_mr',
        valid_input_dir='data/new_train/640_flir_hr_bicubicnoise',
        # valid_input_dir='data/new_train/320_axis_mr',
        predict_input_dir='data/test/evaluation1/hr_x4',
        # predict_input_dir='data/test/evaluation2/mr_real',
        train_label_dir='data/new_train/640_flir_hr',
        valid_label_dir='data/new_train/640_flir_hr',
        predict_transforms=predict_transforms
    )

    cfg = configuration.__dict__[f'{args.model_name}_config']()
    model = Stun(cfg)
    model = BaseModel(model, criterion=None, task=task)

    progress_bar = RichProgressBar()

    gpus = list(map(int, args.gpus.split(',')))

    trainer = pl.Trainer(
        gpus=gpus,
        # strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
        callbacks=[progress_bar],
    )

    ckpt = torch.load(args.weight_path)
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)

    get_submission(outputs, task=task, save_dir=save_dir)


def main() -> None:
    seed = 36
    seed_everything(seed)

    args = get_args()

    eval(args, task='x4')


if __name__ == '__main__':
    main()
