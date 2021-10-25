from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pytorch_lightning as pl

from helpers import predict_transform
from .module import EmptyLayer, DetectionLayer


class Darknet(pl.LightningModule):
    def __init__(self, config_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(config_file)
        self.info, self.modules = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        check = 0

        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "unsample":
                x = self.modules[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(l) for l in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                f = int(module["from"])
                x = outputs[i - 1] + outputs[i + f]
            elif module_type == "yolo":
                anchors = self.modules[i][0].anchors
                input_dim = int(self.info["height"])    # input dimensions
                num_classes = int(module["classes"])    # number of classes

                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                if not check:
                    detect = x
                    check = 1
                else:
                    detect = torch.cat((detect, x), 1)

            outputs[i] = x
        return detect


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
        if l[0] == "[":                 # Check for new block
            if len(b) != 0:               # Check if block not empty
                blocks.append(b)
                b = {}
            b["type"] = l[1:-1].rstrip()
        else:
            key, value = l.split("=")  # get key-value from line
            b[key.rstrip()] = value.lstrip()

    blocks.append(b)
    return blocks


def create_modules(blocks):
    info = blocks[0]                  # info about the input and pre-processing
    modules = nn.ModuleList()
    in_channels = 3
    output_filters = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":  # check type of block
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            pad = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if pad:
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
        elif x["type"] == "route":        # if route layer
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
                print(i, start, len(output_filters))
                filters = output_filters[i + start]
        elif x["type"] == "shortcut":     # shortcut
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(i), shortcut)
        elif x["type"] == "yolo":         # yolo
            mask = x["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[m] for m in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(i), detection)

        modules.append(module)
        pre_filters = filters
        output_filters.append(filters)

    return (info, modules)
