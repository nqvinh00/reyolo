import pytorch_lightning as pl


class EmptyLayer(pl.LightningModule):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(pl.LightningModule):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
