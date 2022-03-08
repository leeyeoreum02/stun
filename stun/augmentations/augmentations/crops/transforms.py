import random
from typing import Dict

import numpy as np

from . import functional as F
from ...core.base_transforms import DualTransform, LabelOnlyTransform


__all__ = ['ModCrop', 'RandomCrop']


class ModCrop(LabelOnlyTransform):
    def __init__(self, scale: int = 4, always_apply: bool = False, p: float = 1.0) -> None:
        super(ModCrop, self).__init__(always_apply, p)

        self.scale = scale

    def apply_to_label(self, image: np.ndarray, **params) -> np.ndarray:
        return F.modcrop(image, self.scale)


class RandomCrop(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        scale: int = 4,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(RandomCrop, self).__init__(always_apply, p)

        self.height = height
        self.width = width
        self.scale = scale

    def apply(self, image: np.ndarray, h_start: int = 0, w_start: int = 0, **params) -> np.ndarray:
        return F.random_crop(image, self.height, self.width, h_start, w_start)

    def get_params(self) -> Dict[str, float]:
        return {'h_start': random.random(), 'w_start': random.random()}

    def apply_to_label(self, image: np.ndarray, h_start: int = 0, w_start: int = 0, **params) -> np.ndarray:
        label_height = self.height * self.scale
        label_width = self.width * self.scale

        return F.random_crop(image, label_height, label_width, h_start, w_start)
