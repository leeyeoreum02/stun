import os
import argparse
from typing import Callable, List, Sequence, Dict

import numpy as np
import pandas as pd

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


def get_predict_transforms(height: int, width: int) -> Sequence[Callable]:
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    
def get_submission(
    outputs: List[Tensor], 
    save_dir: os.PathLike, 
    save_filename: str,
    label_decoder: Dict[int, str]
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    outputs = [o.detach().cpu().numpy() for batch in outputs
                                        for o in batch]
    preds = np.array([label_decoder[int(val)] for val in outputs])
    
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    
    save_file_path = os.path.join(save_dir, save_filename)
    
    submission.to_csv(save_file_path, index=False)
     
    
def eval(
    args: argparse.ArgumentParser, 
    csv_feature_dict: Dict[str, List[float]], 
    label_encoder: Dict[str, int], 
    label_decoder: Dict[int, str],
    submit_save_dir: os.PathLike = 'submissions',
    submit_save_name: str = 'baseline_submission.csv',
) -> None:
    test_data = split_data(mode='test')
    
    predict_transforms = get_predict_transforms(args.height, args.width)
    
    data_module = CustomDataModule(
        test=test_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        predict_transforms=predict_transforms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    if 'tta' in args.model_name:
        model = models.__dict__[args.model_name](
            max_len=24*6, 
            embedding_dim=512, 
            num_features=len(csv_feature_dict), 
            class_n=len(label_encoder),
            tta_transforms=get_tta_transforms(),
            is_onehot=True,
        )
    else:
        model = models.__dict__[args.model_name](
            max_len=24*6, 
            embedding_dim=512, 
            num_features=len(csv_feature_dict), 
            class_n=len(label_encoder),
            is_onehot=True,
        )
    
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
    
    save_file_path = get_onehot_submission(
        outputs, submit_save_dir, submit_save_name, n_class=len(label_encoder))
    
    get_submission_from_onehot(
        save_file_path, submit_save_dir, 
        f"{'-'.join(submit_save_name.split('-')[:-1])}.csv", label_decoder
    )


def main() -> None:
    seed = 42
    seed_everything(seed)
    
    args = get_args()
    
    if args.encoding == 25:
        csv_feature_dict, label_encoder, label_decoder = initialize_n25()
    elif args.encoding == 111:
        csv_feature_dict, label_encoder, label_decoder = initialize()
    else:
        raise Exception("encoding parameter must be '25' or '111'.")
    
    weight_name = args.weight_path.split(os.sep)[1]
    epoch = args.weight_path.split(os.sep)[-1].split('-')[0].split('=')[-1]
    
    submit_save_name = f'{weight_name}-e{epoch}-tw{args.width}-th{args.height}-onehot.csv'
    
    eval(
        args, csv_feature_dict, label_encoder, label_decoder,
        submit_save_name=submit_save_name
    )


if __name__ == '__main__':
    main()