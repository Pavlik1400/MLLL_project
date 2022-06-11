from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from .data_preparation import load
from .helpers import birthdays_to_digits


test_imgs = None
test_labels = None


def evaluate(model, config: Dict):
    if test_imgs is None:
        data = load(*birthdays_to_digits(config["PB"], config["VB"]))
        test_imgs = data.test_data
        test_labels = data.test_targets
    pass