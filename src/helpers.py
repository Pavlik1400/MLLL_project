from itertools import product
from typing import Tuple


def birthdays_to_digits(b1: int, b2: int) -> Tuple[int, int]:
    n = b1 + b2
    a = n % 10
    b = n // 10
    return a, b


def grid_search(params):
    return [dict(zip(params, v)) for v in product(*params.values())]
