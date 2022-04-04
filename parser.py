import torch
import yaml
from pipeline.semantic_segmentation import GenericSegmentationModel
from pipeline.registry import registry


class Parser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def parse_params(self, params):
        if hasattr(params, '__iter__') and "params" in params:
            params = registry[params["name"]](**params["params"])

        if type(params) == dict:
            for k in params.keys():
                if params[k] is not None:
                    params[k] = self.parse_params(params[k])

        return params

    def load_model(self, pipeline, model_name):
        return registry[pipeline[model_name]["name"]](**self.parse_params(pipeline[model_name]["params"]))

    def parse(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"{self.config_path} is not found!")

        pipeline = self.config["pipeline"]
        encoder = self.load_model(pipeline, "encoder")
        pipeline["decoder"]["params"]["encoder_channels"] = encoder.out_channels
        decoder = self.load_model(pipeline, "decoder")
        segmentation_head = self.load_model(pipeline, "segmentation_head")
        segmentation_model = GenericSegmentationModel(encoder, decoder, segmentation_head)
        x = torch.rand([1, 3, 224, 224])
        output = segmentation_model(x)
        return segmentation_model
