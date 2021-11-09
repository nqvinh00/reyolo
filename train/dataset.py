from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, images_path, image_size, max_objects=100, multiscale=True, transform=None, quick=False):
        with open(images_path, "r") as file:
            self.image_files = [name.rstrip() for name in file.readlines()]

        self.label_files = [
            path.replace("images", "labels").replace(
                ".png", ".txt").replace(".jpg", ".txt")
            for path in self.image_files
        ]

        if quick:
            self.image_files = self.image_files[:1000]

        self.image_size = image_size
        self.max_objects = max_objects
        self.multiscale = multiscale
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        try:
            image_path = self.image_files[index %
                                          len(self.image_files)].rstrip()
            image = np.array(Image.open(
                image_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Cannot read image '{image_path}'.")

        try:
            label_path = self.label_files[index %
                                          len(self.image_files)].rstrip()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Cannot read label '{label_path}'.")
            return

        if self.transform:
            try:
                image, targets = self.transform((image, boxes))
            except Exception:
                print("Cannot apply transform.")
                return

        return image_path, image, targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.image_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        return paths, imgs, targets

    def __len__(self):
        return len(self.image_files)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image
