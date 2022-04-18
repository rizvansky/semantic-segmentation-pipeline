import yaml
from pipe.registry import registry
from typing import Dict, Union


class Parser:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def parse_params(
            self, name: str, params: Dict[str, Union[int, str, float, Dict]]
    ) -> Dict[str, Union[int, str, float, Dict]]:
        if hasattr(params, "__iter__") and "params" not in params and "name" not in params:
            for k in params.keys():
                if hasattr(params[k], "__iter__") and "params" in params[k] and "name" in params[k]:
                    params[k] = self.parse_params(params[k]["name"], params[k]["params"])

        if hasattr(params, '__iter__') and "params" in params:
            params = self.parse_params(params["name"], params["params"])
        else:
            if params:
                return registry[name](**params)
            else:
                return registry[name]()

        return params

    def parse(self):
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"{self.config_path} is not found!")

        model_config = config["pipeline"]["model"]
        segmentation_model = self.parse_params(model_config["name"], model_config["params"])

        return segmentation_model
