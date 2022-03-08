from typing import Any, Dict

import numpy as np

import torch
from torch import Tensor

from ..core.base_transforms import DualTransform


__all__ = ['ToTensor']


class ToTensor(DualTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0) -> None:
        super(ToTensor, self).__init__(always_apply=always_apply, p=p)

    def _apply(self, image: np.ndarray, **params) -> Tensor:
        if len(image.shape) != 3:
            raise ValueError('Summerzoo augmentation only supports images in HWC format')

        return torch.from_numpy(image.transpose(2, 0, 1))

    def apply(self, image: np.ndarray, **params) -> Tensor:
        return self._apply(image)

    def apply_to_label(self, image: np.ndarray, **params) -> Tensor:
        return self._apply(image)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict:
        return {}
