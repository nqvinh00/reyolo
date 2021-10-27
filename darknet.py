from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from .module import *
from .helpers import *


class Darknet(pl.LightningModule):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """
        Calculate the output
        Transform the output detection feature maps in a vay can be processed easier
        """
        modules = self.blocks[1:]  # skip first element of blocks, which is net info
        outputs = {}
        check = 0

        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                f = int(module["from"])
                x = outputs[i - 1] + outputs[i + f]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors   # anchors
                input_dim = int(self.net["height"])       # input dimension
                num_classes = int(module["classes"])      # number of classes

                # transform
                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                if not check:
                    detections = x
                    check = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weight(self, file_path):
        file = open(file_path, "rb")

        # first 5 items in weight file are header information
        # major ver, minor ver, subversion, images seen by the network
        header = np.fromfile(file, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.network_seen = self.header[3]
        weights = np.fromfile(file, dtype=np.float32)

        n = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            # if not convolutional, ignore
            if module_type == "convolutional":
                module = self.module_list[i]
                try:
                    batch_normalize = int(
                        self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                convol_layer = module[0]

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
    lines = [l for l in lines if len(l) > 0]
    lines = [l for l in lines if l[0] != '#']
    lines = [l.rstrip().lstrip() for l in lines]

    b = {}
    blocks = []

    for l in lines:
        if l[0] == "[":                   # Check for new block
            if len(b) != 0:               # Check if block not empty
                blocks.append(b)
                b = {}
            b["type"] = l[1: -1].rstrip()
        else:
            key, value = l.split("=")     # get key-value from line
            b[key.rstrip()] = value.lstrip()

    blocks.append(b)
    return blocks


def create_modules(blocks):
    # net info about the input and pre-processing
    net = blocks[0]
    modules = nn.ModuleList()
    in_channels = 3
    output_filters = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        module_type = x["type"]

        # check type of block
        # create new module for block
        # append to module list (modules variable)
        if module_type == "convolutional":
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # convolutional layer
            convol_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias
            )
            module.add_module("conv_{}".format(i), convol_layer)

            # batch norm layer
            if batch_normalize:
                batch_norm_layer = nn.BatchNorm2d(num_features=filters)
                module.add_module("batch_norm_{}".format(i), batch_norm_layer)

            if activation == "leaky":      # linear or leaky relu for yolo
                leaky_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(i), leaky_layer)
        # maxpool layers
        elif module_type == "maxpool":
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2)
            )

            if kernel_size == 2 and stride == 1:
                module.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                module.add_module('MaxPool2d', maxpool)
            else:
                module = maxpool
        # unsample layers
        elif module_type == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(i), upsample)
        # route layer
        elif module_type == "route":
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start -= i
            if end > 0:
                end -= i

            route = EmptyLayer()
            module.add_module("route_{}".format(i), route)
            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i + start]
        # shortcut
        elif module_type == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(i), shortcut)
        # yolo: detection layer
        elif module_type == "yolo":
            mask = x["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[m] for m in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(i), detection)

        modules.append(module)
        in_channels = filters
        output_filters.append(filters)

    return (net, modules)
