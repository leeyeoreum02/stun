import os
from glob import glob
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


__all__ = ['PBVS2022Dataset', 'PBVS2022DataModule']


class PBVS2022Dataset(Dataset):
    def __init__(
        self,
        mode: str,
        input_dir: os.PathLike,
        image_ids: Optional[List[str]] = None,
        label_dir: Optional[os.PathLike] = None,
        transforms: Optional[Sequence[Callable]] = None,
    ) -> None:
        super(PBVS2022Dataset, self).__init__()

        if mode not in ['train', 'test']:
            raise Exception('parameter `mode` must be `train` or `test`')

        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = [
                os.path.splitext(image_path)[0].split(os.sep)[-1]
                for image_path in sorted(glob(os.path.join(input_dir, '*.jpg')))
            ]

        self.mode = mode
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        image_id = self.image_ids[index]

        input_image = self.get_image(image_id, image_type='input')

        if self.mode == 'test':
            if self.transforms is not None:
                input_image = self.transforms(input=input_image)['input']

            return input_image

        label_image = self.get_image(image_id, image_type='label')

        if self.transforms is not None:
            transformed = self.transforms(input=input_image, label=label_image)

            input_image = transformed['input']
            label_image = transformed['label']

        return input_image, label_image

    def get_image(self, image_id: int, image_type: str) -> np.ndarray:
        if image_type not in ['input', 'label']:
            raise Exception('parameter `image_type` must be `input` or `label`')

        if image_type == 'input':
            image_path = os.path.join(self.input_dir, f'{image_id}.jpg')
        else:
            image_path = os.path.join(self.label_dir, f'{image_id}.jpg')

        image = cv2.imread(image_path)

        if image_type == 'input':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        return image

    def get_submir(self, image_type: str) -> os.PathLike:
        if image_type not in ['input', 'label']:
            raise Exception('parameter `image_type` must be `input` or `label`')

        if image_type == 'input':
            sub_dir = self.input_dir.split(os.sep)[-1]
        else:
            sub_dir = self.label_dir.split(os.sep)[-1]

        return sub_dir


class PBVS2022DataModule(LightningDataModule):
    def __init__(
        self,
        train_input_dir: os.PathLike,
        valid_input_dir: os.PathLike,
        predict_input_dir: os.PathLike,
        train_label_dir: os.PathLike,
        valid_label_dir: os.PathLike,
        predict_label_dir: Optional[os.PathLike] = None,
        train_image_ids: Optional[List[str]] = None,
        valid_image_ids: Optional[List[str]] = None,
        predict_image_ids: Optional[List[str]] = None,
        train_transforms: Optional[os.PathLike] = None,
        valid_transforms: Optional[os.PathLike] = None,
        predict_transforms: Optional[os.PathLike] = None,
        num_workers: int = 8,
        batch_size: int = 16,
    ) -> None:
        super(PBVS2022DataModule, self).__init__()

        self.train_input_dir = train_input_dir
        self.valid_input_dir = valid_input_dir
        self.predict_input_dir = predict_input_dir

        self.train_label_dir = train_label_dir
        self.valid_label_dir = valid_label_dir
        self.predict_label_dir = predict_label_dir

        self.train_image_ids = train_image_ids
        self.valid_image_ids = valid_image_ids
        self.predict_image_ids = predict_image_ids

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.predict_transforms = predict_transforms

        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None) -> None:
        self.train_dataset = PBVS2022Dataset(
            mode='train',
            input_dir=self.train_input_dir,
            image_ids=self.train_image_ids,
            label_dir=self.train_label_dir,
            transforms=self.train_transforms,
        )
        self.valid_dataset = PBVS2022Dataset(
            mode='train',
            input_dir=self.valid_input_dir,
            image_ids=self.valid_image_ids,
            label_dir=self.valid_label_dir,
            transforms=self.valid_transforms,
        )
        self.predict_dataset = PBVS2022Dataset(
            mode='test',
            input_dir=self.predict_input_dir,
            image_ids=self.predict_image_ids,
            label_dir=self.predict_label_dir,
            transforms=self.predict_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
