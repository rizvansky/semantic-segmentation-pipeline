import torch.nn as nn
from pipe.registry import registry


@registry.register
class SegmentationHead(nn.Sequential):
    def __init__(self, conv2d, upsampling, activation):
        super().__init__(conv2d, upsampling, activation)
