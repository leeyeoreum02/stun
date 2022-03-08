from __future__ import absolute_import

import random
from typing import Any, Callable, Dict, List

import numpy as np


__all__ = ['BasicTransform', 'DualTransform', 'InputOnlyTransform', 'LabelOnlyTransform']


class BasicTransform:
    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        self.p = p
        self.always_apply = always_apply

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError('You have to pass data to augmentations as named arguments, for example: aug(image=image)')

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), '{} requires {}'.format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)

            return self.apply_with_params(params, **kwargs)

        return kwargs

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    def apply_with_params(self, params: Dict[str, Any], force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if params is None:
            return kwargs

        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def get_params(self) -> Dict:
        return {}

    @property
    def targets(self) -> Dict[str, Callable]:
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # if hasattr(self, "interpolation"):
        #     params["interpolation"] = self.interpolation
        # if hasattr(self, "fill_value"):
        #     params["fill_value"] = self.fill_value
        params.update({'cols': kwargs['input'].shape[1], 'rows': kwargs['input'].shape[0]})
        return params

    @property
    def target_dependence(self) -> Dict:
        return {}

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            f'Method get_params_dependent_on_targets is not implemented in class {self.__class__.__name__}'
        )


class DualTransform(BasicTransform):
    @property
    def targets(self) -> Dict[str, Callable]:
        return {'input': self.apply, 'label': self.apply_to_label}

    def apply_to_label(self, image: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError(f'Method apply_to_label is not implemented in class {self.__class__.__name__}')


class InputOnlyTransform(BasicTransform):
    @property
    def targets(self):
        return {'input': self.apply}


class LabelOnlyTransform(BasicTransform):
    @property
    def targets(self):
        return {'label': self.apply_to_label}
