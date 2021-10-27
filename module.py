import pytorch_lightning as pl


class EmptyLayer(pl.LightningModule):
    """
    Use for route module
    """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(pl.LightningModule):
    """
    Use for yolo module
    """

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
