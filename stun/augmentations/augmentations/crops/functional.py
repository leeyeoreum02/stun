from typing import Tuple

import numpy as np


__all__ = [
    'modcrop',
    'get_random_crop_coords',
    'random_crop',
]


def modcrop(image: np.ndarray, scale: int = 4) -> np.ndarray:
    image_copy = np.copy(image)

    h, w, c = image_copy.shape
    h_r, w_r = h % scale, w % scale
    image_copy = image_copy[:h - h_r, :w - w_r, :]

    return image_copy


def get_random_crop_coords(
    height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float
) -> Tuple[int]:
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width

    return x1, y1, x2, y2


def random_crop(
    img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float
) -> np.ndarray:
    height, width = img.shape[:2]

    if height < crop_height or width < crop_width:
        raise ValueError(
            f'Requested crop size ({crop_height}, {crop_width}) is '
            f'larger than the image size ({height}, {width})'
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]

    return img
