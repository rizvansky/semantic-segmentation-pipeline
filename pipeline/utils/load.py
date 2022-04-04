from pipeline.registry import registry
from collections.abc import Iterable


def parse_params(params):
    if hasattr(params, '__iter__') and "params" in params:
        params = registry[params["name"]](**params["params"])

    if type(params) == dict:
        for k in params.keys():
            if params[k] is not None:
                params[k] = parse_params(params[k])

    return params


def load_model(pipeline, model_name):
    return registry[pipeline[model_name]["name"]](**parse_params(pipeline[model_name]["params"]))
