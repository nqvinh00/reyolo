import imgaug.augmenters as iaa
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torchvision.transforms as transforms


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class ImageAugmenter(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        image, boxes = data
        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes], shape=image.shape)
        image, bounding_boxes = self.augmentations(
            image=image, bounding_boxes=boxes)
        bounding_boxes = bounding_boxes.clip_out_of_image()
        boxes = np.zeros((len(bounding_boxes), 5))
        for i, box in enumerate(bounding_boxes):
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # (x, y, w, h)
            boxes[i, 0] = box.label
            boxes[i, 1] = (x1 + x2) / 2
            boxes[i, 2] = (y1 + y2) / 2
            boxes[i, 3] = x2 - x1
            boxes[i, 4] = y2 - y1

        return image, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, boxes = data
        h, w, _ = image.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return image, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, boxes = data
        h, w, _ = image.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return image, boxes


class PadSquare(ImageAugmenter):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        image, boxes = data
        # Extract image as PyTorch tensor
        image = transforms.ToTensor()(image)

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = transforms.ToTensor()(boxes)

        return image, targets


class DefaultAugmenter(ImageAugmenter):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1),
                       scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAugmenter(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
