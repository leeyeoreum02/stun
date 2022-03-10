import os

import cv2


def _check_unique(image_path: os.PathLike) -> bool:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pass
