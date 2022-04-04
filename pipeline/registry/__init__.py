from .registry import Registry
from torch.nn.modules import linear, conv, pooling, activation, upsampling
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Activation


registry = Registry()
registry.add_modules([linear, conv, pooling, activation, upsampling])
registry.register(get_encoder)
registry.register(Activation)
