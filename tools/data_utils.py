import os
import shutil
from glob import glob
from functools import partial

from tqdm import tqdm


def _copy_image(
    image_path: os.PathLike,
    mode: str,
    dst_dir: os.PathLike,
    n_train_image: int = 951,
) -> None:
    if mode not in ['train', 'valid']:
        raise Exception('parameter `mode` must be `train` or `test`')

    image_filename = image_path.split(os.sep)[-1]
    if mode == 'train':
        shutil.copyfile(
            image_path,
            os.path.join(dst_dir, image_filename)
        )
    else:
        valid_image_id = int(os.path.splitext(image_filename)[0]) + n_train_image
        valid_filename = str(valid_image_id).zfill(4) + '.jpg'
        shutil.copyfile(
            image_path,
            os.path.join(dst_dir, valid_filename)
        )


def combine_train_valid(
    old_train_dir: os.PathLike,
    old_valid_dir: os.PathLike,
    new_train_dir: os.PathLike,
) -> None:
    for sub_dir in ['320_axis_mr', '640_flir_hr', '640_flir_hr_bicubicnoise']:
        dst_dir = os.path.join(new_train_dir, sub_dir)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        else:
            shutil.rmtree(dst_dir)
            os.makedirs(dst_dir)

        train_paths = glob(os.path.join(old_train_dir, sub_dir, '*.jpg'))

        copy_train = partial(_copy_image, mode='train', dst_dir=dst_dir)
        tuple(map(copy_train, tqdm(train_paths)))

        valid_paths = glob(os.path.join(old_valid_dir, sub_dir, '*.jpg'))

        copy_valid = partial(_copy_image, mode='valid', dst_dir=dst_dir)
        tuple(map(copy_valid, tqdm(valid_paths)))
