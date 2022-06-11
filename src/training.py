from typing import Dict

import numpy as np

from src.data_preparation import Dataset
from src.helpers import grid_search


def select_model(dataset: Dataset, model_prototype, parameters: Dict):
    parameters_set = grid_search(parameters)
    scores = []
    models = []

    for p in parameters_set:
        model = model_prototype(**p)
        model.fit(dataset.train_data, dataset.train_targets)

        score = model.score(dataset.cv_data, dataset.cv_targets)
        print(f"Cross-validation accuracy={score:.2f}, parameters={p}")

        models.append(model)
        scores.append(score)

    best = np.argmax(scores)
    best_model = models[best]

    score = best_model.score(dataset.train_data, dataset.train_targets)
    print(f"Test accuracy={score:.2f}, parameters={parameters_set[best]}")

    return best_model
