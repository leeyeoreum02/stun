import os

from tools.data_utils import combine_train_valid


def main() -> None:
    ROOT_DIR = os.path.join('..', '..')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    combine_train_valid(
        os.path.join(DATA_DIR, 'train'),
        os.path.join(DATA_DIR, 'validation'),
        os.path.join(DATA_DIR, 'new_train'),
    )


if __name__ == '__main__':
    main()
