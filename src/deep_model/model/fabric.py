from typing import Dict
from .linear_nn import LinearMnistNN
from ...helpers import fabric_check_key


__MODELS = {
    "linear": LinearMnistNN
}


def get_model(config: Dict):
    model_name = config["deep_model"]["chosen"]
    fabric_check_key(model_name, __MODELS, "model")
    return __MODELS[model_name](**config["deep_model"][model_name])
