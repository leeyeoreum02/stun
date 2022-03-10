from __future__ import division

import cv2
import numpy as np


__all__ = ['normalize', 'hflip', 'hflip_cv2']


def normalize(
    image: np.ndarray, mean: np.ndarray, std: np.ndarray, max_pixel_value: int = 255.0, standardize=False
) -> np.ndarray:
    _, _, c = image.shape

    if not standardize:
        temp = np.array([max_pixel_value] * c, dtype=np.float32)

        denominator = np.reciprocal(temp, dtype=np.float32)

        image = image.astype(np.float32)
        image *= denominator

        return image

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    return image


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv2(img):
    return cv2.flip(img, 1)
