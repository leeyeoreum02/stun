import random as py_random
from typing import Optional, Sequence, Union, Any

import numpy as np


NumType = Union[int, float, np.ndarray]
IntNumType = Union[int, np.ndarray]
Size = Union[int, Sequence[int]]


def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))


def choice(
    a: NumType,
    size: Optional[Size] = None,
    replace: bool = True,
    p: Optional[Union[Sequence[float], np.ndarray]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.choice(a, size, replace, p)  # type: ignore
