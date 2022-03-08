import os
import random
from glob import glob
import collections.abc
from itertools import repeat
from typing import List, Tuple


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def split_train_valid_id(
    image_dir: os.PathLike,
    split_rate: float = 0.1,
    seed: int = 36
) -> Tuple[List[str]]:
    image_ids = [
        os.path.splitext(image_path)[0].split(os.sep)[-1]
        for image_path in sorted(glob(os.path.join(image_dir, '*.jpg')))
    ]
    random.Random(seed).shuffle(image_ids)

    split_point = round(len(image_ids) * split_rate)
    valid_image_ids = image_ids[:split_point]
    train_image_ids = image_ids[split_point:]

    return train_image_ids, valid_image_ids
