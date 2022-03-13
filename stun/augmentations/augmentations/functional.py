from __future__ import division

from typing import Optional

import cv2
import numpy as np


__all__ = ['normalize', 'hflip', 'hflip_cv2', 'pad']


def normalize(
    image: np.ndarray, mean: np.ndarray, std: np.ndarray, max_pixel_value: int = 255.0, standardize=False
) -> np.ndarray:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

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


def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    border_mode: cv2.BORDER_CONSTANT = cv2.BORDER_REFLECT_101,
    value: Optional[int] = None,
) -> np.ndarray:
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            "Invalid result shape. Got: {}. Expected: {}".format(
                img.shape[:2], (max(min_height, height), max(min_width, width))
            )
        )

    return img
