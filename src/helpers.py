from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def birthdays_to_digits(b1: int, b2: int) -> Tuple[int, int]:
    n = b1 + b2
    a = n % 10
    b = n // 10
    return a, b


def fabric_check_key(key: str, obj_dict: Dict, fabric_type: str):
    """Just a fancy helper to fail if you provide bad key to fabric

    Args:
        key (str): name of thing user tries to fabricate
        obj_dict (Dict): dict of names: obj of fabric
        fabric_type (str): what does this fabaric produce?

    Raises:
        ValueError:
    """
    if key not in obj_dict:
        raise ValueError(
            f"Unknown {fabric_type} type: {key}\n  Supported: {list(obj_dict.keys())}"
        )


def show_mnist_img(img: np.ndarray, is_normalized=True, target=None):
    if is_normalized:
        img = img - np.min(img)
        img = img / np.max(img)
    img = img.reshape((28, 28))
    plt.imshow(1.0 - img, cmap="Greys")
    if target is not None:
        plt.title(target)
    plt.show()
