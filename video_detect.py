from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random

from .helpers import draw_result, get_result, load_dataset, pre_image
from .darknet import Darknet


def parse_arg():
    """
    Parse arguments to detect module
    """

    parser = argparse.ArgumentParser(description="reYOLO Detection Module")
    parser.add_argument("--video", dest="video_file", default="/content/videoplayback.mp4",
                        type=str, help="Image path or directory containing images to perform detection")
    parser.add_argument("--bs", default=1, help="Batch size")
    parser.add_argument("--confidence",
                        default=0.5, help="Object confidence to filter predictions")
    parser.add_argument("--nms", default=0.4, help="NMS Threshold")
    parser.add_argument("--cfg", dest="cfg_file",
                        default="/content/yolov3.cfg", type=str, help="Config file path")
    parser.add_argument("--weights", dest="weights_file",
                        default="/content/yolov3.weights", type=str, help="Weights file path")
    parser.add_argument("--dataset",
                        default="/content/coco.names", type=str, help="Dataset file path")
    parser.add_argument("--colors", dest="colors_file",
                        default="/content/pallete", type=str, help="Colors file path")
    parser.add_argument("--source",
                        default="file", type=str, help="Video source")

    args, unknown = parser.parse_known_args()
    return args


class VideoDetect():
    def __init__(self):
        args = parse_arg()
        self.video_file = args.video_file
        self.batch_size = args.bs
        self.confidence = args.confidence
        self.nms = args.nms
        self.cfg_file = args.cfg_file
        self.weights_file = args.weights_file
        self.classes = load_dataset(args.dataset)
        self.num_classes = len(self.classes)
        self.colors_file = args.colors_file
        self.CUDA = torch.cuda.is_available()
        self.source = args.source

    def load_network(self):
        """
        Setup neural network
        """
        self.model = Darknet(self.cfg_file)
        self.model.load_weight(self.weights_file)
        self.input_dim = int(self.model.net["height"])
        assert self.input_dim % 32 == 0
        assert self.input_dim > 32

    def get_detections(self):
        self.load_network()
        if self.CUDA:         # if cuda available
            self.model.cuda()

        self.model.eval()     # set model in evaluation mode

        # get video capture from source (file/webcam)
        if self.source == "video":
            cap = cv2.VideoCapture(self.video_file)
        else:
            cap = cv2.VideoCapture(0)   # webcam
        assert cap.isOpened(), 'Cannot captutre video source'

        frames = 0
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                image = pre_image(frame, self.input_dim)
                img_dim = frame.shape[1], frame.shape[0]
                img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

                if self.CUDA:
                    img_dim = img_dim.cuda()
                    image = image.cuda()

                with torch.no_grad():
                    prediction = self.model(
                        Variable(image, volatile=True), self.CUDA)
                prediction = get_result(prediction, self.confidence,
                                        self.num_classes, nms_conf=self.nms)
                if type(prediction) == int:
                    frames += 1
                    print("FPS: {:5.4f}".format(
                        frames / (time.time() - start)))
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):    # exit if press q
                        break
                    continue

                img_dim = img_dim.repeat(prediction.size(0), 1)
                scale_factor = torch.min(
                    self.input_dim / img_dim, 1)[0].view(-1, 1)
                prediction[:, [1, 3]] -= (self.input_dim -
                                          scale_factor * img_dim[:, 0].view(-1, 1)) / 2
                prediction[:, [2, 4]] -= (self.input_dim -
                                          scale_factor * img_dim[:, 1].view(-1, 1)) / 2
                prediction[:, 1: 5] /= scale_factor

                for i in range(prediction.shape[0]):
                    prediction[i, [1, 3]] = torch.clamp(
                        prediction[i, [1, 3]], 0.0, img_dim[i, 0])
                    prediction[i, [2, 4]] = torch.clamp(
                        prediction[i, [2, 4]], 0.0, img_dim[i, 1])

                list(map(lambda x: draw_result(
                    x, frame, self.colors, self.classes), prediction))
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                t = time.time() - start
                print("Predicted in {1:6.3f} seconds".format(t))
                print("FPS: {:5.2f}".format(frames / (time.time() - start)))
            else:
                break
