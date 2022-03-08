import random
from typing import Dict

import numpy as np

from ...core.base_transforms import DualTransform


__all__ = ['RandomRotate90']


class RandomRotate90(DualTransform):
    def _apply(self, image: np.ndarray, factor: int = 0, **params) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(image, factor))

    def get_params(self) -> Dict[str, int]:
        return {'factor': random.randint(0, 3)}

    def apply(self, image: np.ndarray, factor: int = 0, **params) -> np.ndarray:
        return self._apply(image, factor, **params)

    def apply_to_label(self, image: np.ndarray, factor: int = 0, **params) -> np.ndarray:
        return self._apply(image, factor, **params)
