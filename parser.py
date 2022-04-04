import torch
import yaml
from pipeline.utils import load_model
from pipeline.semantic_segmentation import GenericSegmentationModel


class Parser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def parse(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"{self.config_path} is not found!")

        pipeline = self.config["pipeline"]
        encoder = load_model(pipeline, "encoder")
        decoder = load_model(pipeline, "decoder")
        segmentation_head = load_model(pipeline, "segmentation_head")

        segmentation_model = GenericSegmentationModel(encoder, decoder, segmentation_head)
        x = torch.rand([1, 3, 224, 224])
        output = segmentation_model(x)
        return segmentation_model
