import os

from tools.visualize import draw_dataset
from stun.data_module import PBVS2022DataModule
from stun.utils import split_train_valid_id


def main() -> None:
    ROOT_DIR = os.path.join('..')
    TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'new_train')
    TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test')
    X2_TRAIN_DIR = os.path.join(TRAIN_DIR, '320_axis_mr')
    X4_TRAIN_DIR = os.path.join(TRAIN_DIR, '640_flir_hr_bicubicnoise')
    LABEL_DIR = os.path.join(TRAIN_DIR, '640_flir_hr')
    X2_TEST_DIR = os.path.join(TEST_DIR, 'evaluation2', 'mr_real')
    X4_TEST_DIR = os.path.join(TEST_DIR, 'evaluation1', 'hr_x4')

    x2_train_image_ids, x2_valid_image_ids = split_train_valid_id(X2_TRAIN_DIR)
    x4_train_image_ids, x4_valid_image_ids = split_train_valid_id(X4_TRAIN_DIR)

    x2_data_module = PBVS2022DataModule(
        train_input_dir=X2_TRAIN_DIR,
        valid_input_dir=X2_TRAIN_DIR,
        predict_input_dir=X2_TEST_DIR,
        train_label_dir=LABEL_DIR,
        valid_label_dir=LABEL_DIR,
        train_image_ids=x2_train_image_ids,
        valid_image_ids=x2_valid_image_ids,
    )
    x2_data_module.setup()
    x4_data_module = PBVS2022DataModule(
        train_input_dir=X4_TRAIN_DIR,
        valid_input_dir=X4_TRAIN_DIR,
        predict_input_dir=X4_TEST_DIR,
        train_label_dir=LABEL_DIR,
        valid_label_dir=LABEL_DIR,
        train_image_ids=x4_train_image_ids,
        valid_image_ids=x4_valid_image_ids,
    )
    x4_data_module.setup()

    x2_train_dataset = x2_data_module.train_dataset
    x2_valid_dataset = x2_data_module.valid_dataset
    x4_train_dataset = x4_data_module.train_dataset
    x4_valid_dataset = x4_data_module.valid_dataset

    draw_dataset(x2_train_dataset, os.path.join(ROOT_DIR, 'examples', 'x2_train'))
    draw_dataset(x2_valid_dataset, os.path.join(ROOT_DIR, 'examples', 'x2_valid'))
    draw_dataset(x4_train_dataset, os.path.join(ROOT_DIR, 'examples', 'x4_train'))
    draw_dataset(x4_valid_dataset, os.path.join(ROOT_DIR, 'examples', 'x4_valid'))


if __name__ == '__main__':
    main()
