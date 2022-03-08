import random
import warnings
from typing import Any, Dict, List, Sequence, Union

import numpy as np

from stun.augmentations import random_utils
from summerzoo.augmentations.base_transforms import BasicTransform


__all__ = ['BaseCompose', 'Compose', 'OneOf']


TransformType = Union[BasicTransform, 'BaseCompose']
TransformsSeqType = Sequence[TransformType]


def get_always_apply(transforms: Union['BaseCompose', TransformsSeqType]) -> TransformsSeqType:
    new_transforms: List[TransformType] = []
    for transform in transforms:  # type: ignore
        if isinstance(transform, BaseCompose):
            new_transforms.extend(get_always_apply(transform))
        elif transform.always_apply:
            new_transforms.append(transform)
    return new_transforms


class BaseCompose:
    def __init__(self, transforms: TransformsSeqType, p: float) -> None:
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                'transforms is single transform, but a sequence is expected! Transform will be wrapped into list.'
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args, **data) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> Union['BaseCompose', BasicTransform]:  # type: ignore
        return self.transforms[item]


class Compose(BaseCompose):
    def __init__(self, transforms: TransformsSeqType, p: float = 1.0) -> None:
        super(Compose, self).__init__(transforms, p)

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

    def __call__(self, *args, force_apply: bool = False, **data) -> Dict[str, Any]:
        if args:
            raise KeyError('You have to pass data to augmentations as named arguments, for example: aug(image=image)')

        if self.is_check_args:
            self._check_args(**data)

        assert isinstance(force_apply, (bool, int)), 'force_apply must have bool or int type'

        need_to_run = force_apply or random.random() < self.p
        transforms = self.transforms if need_to_run else get_always_apply(self.transforms)

        for idx, t in enumerate(transforms):
            data = t(force_apply=force_apply, **data)

        data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous

        return data

    @staticmethod
    def _disable_check_args_for_transforms(transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, Compose):
                transform._disable_check_args()

    def _disable_check_args(self) -> None:
        self.is_check_args = False

    def _check_args(self, **kwargs) -> None:
        checked_single = ['image']

        for data_name, data in kwargs.items():
            internal_data_name = {}
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError('{data_name} must be numpy array type')

    @staticmethod
    def _make_targets_contiguous(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result


class OneOf(BaseCompose):
    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> Dict[str, Any]:
        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx: int = random_utils.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data
