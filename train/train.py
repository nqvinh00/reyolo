from __future__ import division

import os
import argparse
import tqdm
import numpy as np
import random

import torch
from helpers import load_dataset
from augmenter import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from darknet import Darknet
from .dataset import ListDataset

def parse_arg():
    parser = argparse.ArgumentParser(description="reYOLO Training Module")
    parser.add_argument("--cfg", dest="cfg_file", type=str,
                        default="/content/yolov3.cfg", help="Config file path")
    parser.add_argument("--dataset", type=str,
                        default="/content/coco.names", help="Dataset file path")
    parser.add_argument("--train_path", type=str,
                        default="/content/data/trainvalno5k.txt")
    parser.add_argument("--valid_path", type=str,
                        default="/content/data/5k.txt")
    parser.add_argument(
        "--backup", type=str, default="/content/backup/", help="Backup directory path")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs")
    parser.add_arguemnt("--verbose", action="store_true",
                        help="Verbose training")
    parser.add_argument("--cpus", type=int, default=8,
                        help="Number of cpu threads during batch generation")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Checkpoint file path (.weights or .pth)")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--multiscale_train",
                        action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--confidence", type=float,
                        default=0.1, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--seed", type=int, default=-1)
    args, unknown = parser.parse_known_args()
    return args


def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_output_dir():
    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def load_model(path, device, weights=None):
    model = Darknet(path).to(device)
    model.apply(weights_init_normal)

    if weights:
        if weights.endswith(".pth"):
            model.load_state_dict(torch.load(weights, map_location=device))
        else:
            model.load_weight(weights)

    return model


class TrainingModule(pl.LightningModule):
    def __init__(self):
        args = parse_arg()
        if args.seed != -1:
            init_seed(args.seed)

        self.classes = load_dataset(args.dataset)
        self.train_path = args.train_path
        self.valid_path = args.valid_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(args.cfg_file, args.pretrained_weights)
        self.multiscale_train = args.multiscale_train
        self.cpus = args.cpus
        self.epochs = args.epochs
        if args.verbose:
            summary(self.model, input_size=(
                3, self.model.net["height"], self.model.net["height"]))

        self.mini_batch_size = self.model.net["batch"] // self.model.net["subdivision"]

    def train_dataloader(self):
        def set_worker_seed(id):
            seed = torch.initial_seed()
            ss = np.random.SeedSequence([seed])
            np.random.seed(ss.generate_state(4))
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed)

        dataset = ListDataset(self.train_path, image_size=self.model.net["height"], multiscale=self.multiscale_train, transform=AUGMENTATION_TRANSFORMS)
        return DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True, num_workers=self.cpus, pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=set_worker_seed)

    def valid_dataloader(self):
        dataset = ListDataset(
            self.valid_path, image_size=self.model.net["height"], multiscale=False, transform=DEFAULT_TRANSFORMS)
        return DataLoader(dataset, batch_size=self.mini_batch_size, suffle=False, num_workers=self.cpus, pin_memory=True, collate_fn=dataset.collate_fn)

    def create_optimizer(self):
        params = [p for p in self.model.parameters() if p.require_grad]
        if self.model.net["optimizer"] in [None, "adam"]:
            optimizer = optim.Adam(
                params, lr=self.model.net["learning_rate"], weight_decay=self.model.net["decay"])
        elif self.model.net["optimizer"] == "sgd":
            optimizer = optim.SGD(
                params, lr=self.model.net["learning_rate"], weight_decay=self.model.net["decay"], momentum=self.model.net["momentum"])

    def train(self):
        train_dl = self.train_dataloader
        valid_dl = self.valid_dataloader
        for epoch in range(self.epochs):
            self.model.train()
            for batch_id, (_, imgs, targets) in enumerate(tqdm.tqdm(train_dl, desc=f"Training epoch {epoch}")):
                done = len(train_dl) * epoch + batch_id
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device)
