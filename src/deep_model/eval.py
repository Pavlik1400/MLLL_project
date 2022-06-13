import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from typing import Dict
from torch.nn import Module
from src.data_preparation import load
from src.helpers import birthdays_to_digits
from sklearn.metrics import accuracy_score, f1_score


def eval_model(model: Module, config: Dict) -> Dict:
    dataset = load(*birthdays_to_digits(config["PB"], config["VB"]))
    model.eval()

    inputs = torch.from_numpy(dataset.test_data.astype(np.float32))
    labels = dataset.test_targets

    outputs = model(inputs)
    preds = outputs.max(dim=1)[1].numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Test accuracy: {acc}, f1-score: {f1}")
    return {
        "accuracy": acc,
        "f1-score": f1,
    }
