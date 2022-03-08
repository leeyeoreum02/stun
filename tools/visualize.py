import os
import shutil
import random
from functools import partial
from typing import List

import cv2
from tqdm import tqdm

from stun.data_module import PBVS2022Dataset


def _draw_image(
    image_id: int,
    dataset: PBVS2022Dataset,
    save_path: os.PathLike,
    image_type: str,
) -> None:
    image = dataset.get_image(image_id, image_type=image_type)

    image_name = f'{image_id}.jpg'
    cv2.imwrite(os.path.join(save_path, image_name), image)


def draw_dataset(
    dataset: PBVS2022Dataset,
    save_dir: os.PathLike,
    n_images: int = 100,
    seed: int = 36,
) -> None:
    image_ids = dataset.image_ids
    random.Random(seed).shuffle(image_ids)

    for image_type in ['input', 'label']:
        sub_dir = dataset.get_submir(image_type=image_type)
        save_path = os.path.join(save_dir, sub_dir)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        draw_image = partial(
            _draw_image, dataset=dataset, save_path=save_path, image_type=image_type
        )
        tuple(map(draw_image, tqdm(image_ids[:n_images])))


def _draw_transformed_image(
    i: int,
    image_ids: List[str],
    dataset: PBVS2022Dataset,
    input_save_dir: os.PathLike,
    label_save_dir: os.PathLike,
) -> None:
    image_id = image_ids[i]
    input_image, label_image = dataset[i]

    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)

    image_name = f'{image_id}.jpg'
    cv2.imwrite(os.path.join(input_save_dir, image_name), input_image)
    cv2.imwrite(os.path.join(label_save_dir, image_name), label_image)


def draw_transformed_image(
    dataset: PBVS2022Dataset,
    save_dir: os.PathLike,
    n_images: int = 100,
) -> None:
    image_ids = dataset.image_ids

    input_sub_dir = dataset.get_submir(image_type='input')
    input_save_path = os.path.join(save_dir, input_sub_dir)

    if not os.path.exists(input_save_path):
        os.makedirs(input_save_path)
    else:
        shutil.rmtree(input_save_path)
        os.makedirs(input_save_path)

    label_sub_dir = dataset.get_submir(image_type='label')
    label_save_path = os.path.join(save_dir, label_sub_dir)

    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    else:
        shutil.rmtree(label_save_path)
        os.makedirs(label_save_path)

    draw_image = partial(
        _draw_transformed_image,
        image_ids=image_ids,
        dataset=dataset,
        input_save_dir=input_save_path,
        label_save_dir=label_save_path,
    )
    tuple(map(draw_image, tqdm(tuple(range(n_images)))))
