import os

from tools.visualize import draw_transformed_image
from stun.data_module import PBVS2022DataModule
from stun.utils import split_train_valid_id
import stun.augmentations as A
from stun.augmentations.pytorch import ToTensor


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

    transforms = A.Compose([
        A.Pad(scale=4, min_height=160, min_width=160),
        # A.RandomCrop(height=96, width=96, scale=2),
        # A.HorizontalFlip(),
        # A.RandomRotate90(),
        # A.ModCrop(),
        A.Normalize(),
        # ToTensor(),
    ])

    # x2_data_module = PBVS2022DataModule(
    #     train_input_dir=X2_TRAIN_DIR,
    #     valid_input_dir=X2_TRAIN_DIR,
    #     predict_input_dir=X2_TEST_DIR,
    #     train_label_dir=LABEL_DIR,
    #     valid_label_dir=LABEL_DIR,
    #     train_image_ids=x2_train_image_ids,
    #     valid_image_ids=x2_valid_image_ids,
    #     train_transforms=transforms,
    #     valid_transforms=transforms,
    # )
    # x2_data_module.setup()
    x4_data_module = PBVS2022DataModule(
        train_input_dir=X4_TRAIN_DIR,
        valid_input_dir=X4_TRAIN_DIR,
        predict_input_dir=X4_TEST_DIR,
        train_label_dir=LABEL_DIR,
        valid_label_dir=LABEL_DIR,
        train_image_ids=x4_train_image_ids,
        valid_image_ids=x4_valid_image_ids,
        train_transforms=transforms,
        valid_transforms=transforms,
    )
    x4_data_module.setup()

    # x2_train_dataset = x2_data_module.train_dataset
    # x2_valid_dataset = x2_data_module.valid_dataset
    x4_train_dataset = x4_data_module.train_dataset
    x4_valid_dataset = x4_data_module.valid_dataset

    # print(x2_train_dataset[0])
    # print(x2_valid_dataset[0])
    print(x4_train_dataset[0])
    print(x4_valid_dataset[0])

    # print(x2_train_dataset[0][0].shape, x2_train_dataset[0][0].type())
    # print(x2_train_dataset[0][1].shape, x2_train_dataset[0][1].type())
    # print(x2_valid_dataset[0][0].shape, x2_valid_dataset[0][0].type())
    # print(x2_valid_dataset[0][1].shape, x2_valid_dataset[0][1].type())
    # print(x4_train_dataset[0][0].shape, x4_train_dataset[0][0].type())
    # print(x4_train_dataset[0][1].shape, x4_train_dataset[0][1].type())
    print(x4_valid_dataset[0][0].shape, x4_valid_dataset[0][0].dtype)
    print(x4_valid_dataset[0][1].shape, x4_valid_dataset[0][1].dtype)

    # draw_transformed_image(x2_train_dataset, os.path.join(ROOT_DIR, 'examples', 'transformed_x2_train'))
    # draw_transformed_image(x2_valid_dataset, os.path.join(ROOT_DIR, 'examples', 'transformed_x2_valid'))
    draw_transformed_image(x4_train_dataset, os.path.join(ROOT_DIR, 'examples', 'transformed_x4_train'))
    draw_transformed_image(x4_valid_dataset, os.path.join(ROOT_DIR, 'examples', 'transformed_x4_valid'))


if __name__ == '__main__':
    main()
