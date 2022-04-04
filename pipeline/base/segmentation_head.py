import torch.nn as nn
from pipeline.registry import registry


@registry.register
class SegmentationHead(nn.Sequential):
    def __init__(self, conv2d, upsampling, activation=None):
        super().__init__(conv2d, upsampling, activation)
