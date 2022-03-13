from typing import Optional, Tuple
import cv2

import numpy as np

from . import functional as F
from ..core.base_transforms import DualTransform


__all__ = ['Normalize', 'HorizontalFlip', 'Pad']


class Normalize(DualTransform):
    def __init__(
        self,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        standardize: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(Normalize, self).__init__(always_apply, p)

        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.standardize = standardize

    def _apply(self, image: np.ndarray, **params) -> np.ndarray:
        return F.normalize(image, self.mean, self.std, self.max_pixel_value, self.standardize)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return self._apply(image, **params)

    def apply_to_label(self, image: np.ndarray, **params) -> np.ndarray:
        return self._apply(image, **params)


class HorizontalFlip(DualTransform):
    def _apply(self, image: np.ndarray, **params) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] > 1 and image.dtype == np.uint8:
            return F.hflip_cv2(image)

        return F.hflip(image)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return self._apply(image)

    def apply_to_label(self, image: np.ndarray, **params) -> np.ndarray:
        return self._apply(image)


class Pad(DualTransform):
    def __init__(
        self,
        scale: int,
        min_height: int,
        min_width: int,
        border_mode: cv2.BORDER_CONSTANT = cv2.BORDER_REFLECT_101,
        value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(Pad, self).__init__(always_apply, p)

        self.scale = scale
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return F.pad(image, self.min_height, self.min_width, self.border_mode, self.value)

    def apply_to_label(self, image: np.ndarray, **params) -> np.ndarray:
        return F.pad(image, self.min_height * self.scale, self.min_width * self.scale, self.border_mode, self.value)
