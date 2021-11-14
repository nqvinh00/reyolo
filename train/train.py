from __future__ import division

import os
import argparse
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from wcmatch.pathlib import Path
import pytorch_lightning as pl
from datetime import datetime
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger

from darknet import Darknet
from helpers import load_dataset, save_code_files
from train.dataset import ImageDataset, Transform


def parse_arg():
    parser = argparse.ArgumentParser(description="reYOLO Training Module")
    parser.add_argument("--cfg", dest="cfg_file", type=str,
                        default="./cfg/yolov3-tiny.cfg", help="Config file path")
    parser.add_argument("--dataset", type=str,
                        default="./data/coco.names", help="Dataset file path")
    parser.add_argument("--train_path", type=str,
                        default="./data/trainvalno5k.txt")
    parser.add_argument("--valid_path", type=str,
                        default="./data/5k.txt")
    parser.add_argument("--nms", default=0.4, help="NMS Threshold")
    parser.add_argument("--iou", default=0.5, help="NMS Threshold")
    parser.add_argument("--confidence", default=0.5,
                        help="Object confidence to filter predictions")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs")
    parser.add_argument("--cpus", type=int, default=0,
                        help="Number of cpu threads during batch generation")
    parser.add_argument("--pretrained_weights", default="./weight/yolov3-tiny.weights",
                        type=str, help="Checkpoint file path (.weights or .pt)")
    parser.add_argument("--multiscale_train",
                        action="store_true", help="Allow multi-scale training")
    parser.add_argument("--seed", type=int, default=-1)
    args, _ = parser.parse_known_args()
    return args


def load_model(path, device, weights=None):
    model = Darknet(path).to(device)

    if weights:
        if weights.endswith(".pth"):
            model.load_state_dict(torch.load(weights, map_location=device))
        else:
            model.load_weight(weights)

    return model


class DataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size, cpus):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.cpus = cpus

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.cpus, pin_memory=True, collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus, pin_memory=True, collate_fn=self.val_ds.collate_fn)


class Net(pl.LightningModule):
    def __init__(self, model, img_size, batch_size, args):
        super().__init__()
        self.model = model
        self.valid_path = args.valid_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.nms = args.nms
        self.conf = args.confidence
        self.iou = args.iou

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        images, targets = batch[1:]
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        loss, outputs = self.model(Variable(images.to(device)), targets)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch[1:]
        loss = self.model(imgs, targets)
        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if "optimizer" not in self.model.net or self.model.net["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(
                self.model.net["learning_rate"]), weight_decay=float(self.model.net["decay"]))
        elif self.net["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.model.net["learning_rate"]), weight_decay=float(
                self.model.net["decay"]), momentum=self.model.net["momentum"])

        return optimizer


if __name__ == "__main__":
    level = "DEBUG"
    experiment_root = 'experiments'
    exp_root_path = Path(experiment_root)
    wandb.login()
    experiment_dir = exp_root_path / "train" / f"exp_{datetime.now()}"
    log_file = experiment_dir / f"log_{datetime.now()}.log"
    logger.opt(record=True).add(
        log_file, format=" {time:YYYY-MMM HH:mm:ss} {name}:{function}:{line} <lvl>{message}</>", level=level, rotation="5 MB")
    experiment_dir.mkdir(exist_ok=True)

    args = parse_arg()
    logger.opt(colors=True).info(args)
    save_code_files(experiment_dir, os.path.abspath(''))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    darknet = Darknet(args.cfg_file)
    img_size = int(darknet.net["height"])
    batch_size = int(darknet.net["batch"]) // int(darknet.net["subdivisions"])
    classes = load_dataset(args.dataset)

    train_ds = ImageDataset(images_path=args.train_path, multiscale=args.multiscale_train,
                            image_size=img_size, transform=Transform.train)
    val_ds = ImageDataset(images_path=args.valid_path, multiscale=args.multiscale_train,
                          image_size=img_size, transform=Transform.val)

    model = Net(darknet, img_size, batch_size, args)
    model = model.to(device)

    data_module = DataModule(train_ds, val_ds, batch_size, args.cpus)

    wandb_logger = WandbLogger(
        project="reYOLO", save_dir=experiment_dir, offline=False, name="test")
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir, mode="min", monitor="val_loss")

    logger.opt(colors=True).info("Start training")
    trainer = pl.Trainer(logger=wandb_logger, auto_scale_batch_size='binsearch',
                         num_sanity_val_steps=0, callbacks=[checkpoint_callback], weights_save_path="weights")
    trainer.fit(model, data_module)
