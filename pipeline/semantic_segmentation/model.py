from segmentation_models_pytorch.base import SegmentationModel
from pipeline.registry import registry


@registry.register
class GenericSegmentationModel(SegmentationModel):
    def __init__(self, encoder, decoder, segmentation_head, classification_head):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = segmentation_head
        self.classification_head = classification_head

        self.initialize()
