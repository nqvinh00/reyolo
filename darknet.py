from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from .module import *
from .helpers import *


class Darknet(pl.LightningModule):
    def __init__(self, cfg_file, apply_focal_loss=False):
        super(Darknet, self).__init__()
        self.apply_focal_loss = apply_focal_loss
        self.blocks = parse_cfg(cfg_file)
        self.net, self.module_list = create_modules(
            self.blocks, self.apply_focal_loss)
        self.detection_layers = [
            layer[0] for layer in self.module_list if isinstance(layer[0], DetectionLayer)]
        self.image_size = int(self.net["height"])
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        """
        Calculate the output
        Transform the output detection feature maps in a vay can be processed easier
        """

        image_dim = x.shape[2]
        loss = 0
        layer_outputs, detection_outputs = [], []

        for i, (module_def, module) in enumerate(zip(self.blocks, self.module_list)):
            module_type = module_def["type"]

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                x = torch.cat([layer_outputs[int(layer_i)]
                              for layer_i in module_def["layers"].split(",")], 1)
            elif module_type == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_type == "yolo":
                x, layer_loss = module[0](x, targets, image_dim)
                loss += layer_loss
                detection_outputs.append(x)

            layer_outputs.append(x)
        detection_outputs = to_cpu(torch.cat(detection_outputs, 1))
        return detection_outputs if targets is None else (loss, detection_outputs)

    def load_weight(self, file_path):
        # first 5 items in weight file are header information
        # major ver, minor ver, subversion, images seen by the network
        with open(file_path, "rb") as file:
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = self.header_info[3]
            weights = np.fromfile(file, dtype=np.float32)

        cutoff = None
        if "darknet53.conv.74" in file_path:
            cutoff = 75

        n = 0
        for i, (module_def, module) in enumerate(zip(self.blocks, self.module_list)):
            module_type = module_def["type"]
            if i == cutoff:
                break

            # if not convolutional, ignore
            if module_type == "convolutional":
                convol_layer = module[0]
                try:
                    batch_normalize = int(module_def["batch_normalize"])
                except:
                    batch_normalize = 0
                # batch normalize layer
                if batch_normalize:
                    batch_norm_layer = module[1]
                    num_biases = batch_norm_layer.bias.numel()

                    # load weights
                    bnl_biases = torch.from_numpy(weights[n: n + num_biases])
                    n += num_biases

                    bnl_weights = torch.from_numpy(weights[n: n + num_biases])
                    n += num_biases

                    bnl_running_mean = torch.from_numpy(
                        weights[n: n + num_biases])
                    n += num_biases

                    bnl_running_var = torch.from_numpy(
                        weights[n: n + num_biases])
                    n += num_biases

                    # cast weights into dimensions of model weights
                    bnl_biases = bnl_biases.view_as(batch_norm_layer.bias.data)
                    bnl_weights = bnl_weights.view_as(
                        batch_norm_layer.weight.data)
                    bnl_running_mean = bnl_running_mean.view_as(
                        batch_norm_layer.running_mean)
                    bnl_running_var = bnl_running_var.view_as(
                        batch_norm_layer.running_var)

                    # copy data to model
                    batch_norm_layer.bias.data.copy_(bnl_biases)
                    batch_norm_layer.weight.data.copy_(bnl_weights)
                    batch_norm_layer.running_mean.copy_(bnl_running_mean)
                    batch_norm_layer.running_var.copy_(bnl_running_var)
                else:     # convolutional layer
                    num_biases = convol_layer.bias.numel()

                    # load weights
                    convol_biases = torch.from_numpy(
                        weights[n: n + num_biases])
                    n += num_biases

                    # cast weights into dimensions of model weights
                    convol_biases = convol_biases.view_as(
                        convol_layer.bias.data)

                    # copy data to model
                    convol_layer.bias.data.copy_(convol_biases)

                # weights of convolutional layerss
                num_weights = convol_layer.weight.numel()
                convol_weights = torch.from_numpy(weights[n: n + num_weights])
                n += num_weights
                convol_weights = convol_weights.view_as(
                    convol_layer.weight.data)
                convol_layer.weight.data.copy_(convol_weights)


def parse_cfg(file):
    """
    Parse config from file. Returns a list of blocks.
    Each blocks describes a block in neural network to be built.
    """

    file = open(file, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []

    for line in lines:
        if line.startswith("["):                   # Check for new block
            module_defs.append({})                 # Check if block not empty
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")           # get key-value from line
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def create_modules(module_defs, focal_loss):
    # net info about the input and pre-processing
    hyperparams = module_defs.pop(0)
    momentum = float(hyperparams["momentum"])
    module_list = nn.ModuleList()
    in_channels = 3
    output_filters = [int(hyperparams["channels"])]

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        module_type = module_def["type"]

        # check type of block
        # create new module for block
        # append to module list (modules variable)
        if module_type == "convolutional":
            batch_normalize = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2

            # convolutional layer
            convol_layer = nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size, stride=int(
                module_def["stride"]), padding=pad, bias=not batch_normalize)
            modules.add_module("conv_{}".format(module_i), convol_layer)

            # batch norm layer
            if batch_normalize:
                modules.add_module("batch_norm_{}".format(
                    module_i), nn.BatchNorm2d(filters, momentum=momentum, eps=1e-5))
            # linear or leaky relu for yolo
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_{}".format(
                    module_i), nn.LeakyReLU(0.1))
        # maxpool layers
        elif module_type == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])

            if kernel_size == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))

            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)
        # unsample layers
        elif module_type == "upsample":
            upsample = Upsample(scale_factor=int(
                module_def["stride"]), mode="nearest")
            modules.add_module("upsample_{}".format(module_i), upsample)
        # route layer
        elif module_type == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module("route_{}".format(module_i), nn.Sequential())
        # shortcut
        elif module_type == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module("shortcut_{}".format(module_i), nn.Sequential())
        # yolo: detection layer
        elif module_type == "yolo":
            anchor_indexs = [int(x) for x in module_def["mask"].split(",")]

            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_indexs]
            num_classes = int(module_def["classes"])
            image_size = int(hyperparams["height"])

            detection = DetectionLayer(
                anchors, num_classes, focal_loss, image_size)
            modules.add_module("Detection_{}".format(module_i), detection)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
