from torch.nn.modules import linear, conv, pooling, activation
from .registry import Registry


registry = Registry()
registry.add_modules([linear, conv, pooling, activation])
