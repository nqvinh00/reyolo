from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        # label placeholder
        boxes = np.zeros((1, 5))

        # apply transforms
        if self.transform:
            image, _ = self.transform((image, boxes))

        return image_path, image

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, image_size, max_objects=100, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.label_files = []
        self.image_size = image_size
        self.max_objects = max_objects
        self.multiscale = multiscale
        self.min_size = self.image_size - 96
        self.max_size = self.image_size + 96
        self.batch_count = 0
        self.transform = transform

        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, f"Image dir path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.splitext(label_file)[0] + ".txt"
            self.label_files.append(label_file)

    def __getitem(self, index):
        try:
            image_path = self.img_files[index % len(self.img_files)].rstrip()
            image = np.array(Image.open(
                image_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            print(f"Cannot read image '{image_path}'.")
            return

        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
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
                print("Cannot apply transform")
                return

        return image_path, image, targets
