from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd

from darknet import Darknet
from helpers import draw_result, get_result, load_dataset, pre_image


def parse_arg():
    """
    Parse arguments to detect module
    """

    parser = argparse.ArgumentParser(description="reYOLO Detection Module")
    parser.add_argument("--images", default="test_img/dog-cycle-car.png",
                        type=str, help="Image path or directory containing images to perform detection")
    parser.add_argument("--det", default="det", type=str,
                        help="Imgage path or directory to store detections")
    parser.add_argument("--bs", default=1, help="Batch size")
    parser.add_argument("--confidence",
                        default=0.5, help="Object confidence to filter predictions")
    parser.add_argument("--nms", default=0.4, help="NMS Threshold")
    parser.add_argument("--cfg", dest="cfg_file",
                        default="cfg/yolov3.cfg", type=str, help="Config file path")
    parser.add_argument("--weights", dest="weights_file",
                        default="weight/yolov3.weights", type=str, help="Weights file path")
    parser.add_argument("--dataset",
                        default="data/coco.names", type=str, help="Dataset file path")
    parser.add_argument("--colors", dest="colors_file",
                        default="./pallete", type=str, help="Colors file path")

    args, unknown = parser.parse_known_args()
    return args


class ImageDetect():
    def __init__(self):
        args = parse_arg()
        self.images = args.images
        self.cfg_file = args.cfg_file
        self.weights_file = args.weights_file
        self.det = args.det
        self.batch_size = int(args.bs)
        self.confidence = float(args.confidence)
        self.nms = float(args.nms)
        self.CUDA = torch.cuda.is_available()
        self.classes = load_dataset(args.dataset)
        self.num_classes = len(self.classes)
        self.colors_file = args.colors_file

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
        if self.CUDA:           # if cuda available
            self.model.cuda()

        self.model.eval()       # set model in evaluation mode
        read_time = time.time()

        try:
            image_list = [osp.join(osp.realpath("."), self.images, img)
                          for img in os.listdir(self.images)]
        except NotADirectoryError:
            image_list = []
            image_list.append(osp.join(osp.realpath("."), self.images))
        except FileNotFoundError:
            print("No file or directory with name {}".format(self.images))
            exit()

        if not os.path.exists(self.det):
            os.makedirs(self.det)

        load_batch_time = time.time()
        loaded_img_list = [cv2.imread(x) for x in image_list]
        # pytorch variables for images
        img_batches = list(map(pre_image, loaded_img_list, [
                           self.input_dim for i in range(len(image_list))]))
        # dimensions of original images
        img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_img_list]
        img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)

        # create batches
        left_over = 0
        if len(img_dim_list) % self.batch_size:
            left_over = 1

        if self.batch_size != 1:
            num_batches = len(image_list) // self.batch_size + left_over
            img_batches = [torch.car((img_batches[i * self.batch_size: min(
                (i + 1) * self.batch_size, len(img_batches))])) for i in range(num_batches)]

        check = 0
        if self.CUDA:
            img_dim_list = img_dim_list.cuda()

        start_detect_loop_time = time.time()

        # detection loop
        for i, batch in enumerate(img_batches):
            start = time.time()
            if self.CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = self.model(Variable(batch), self.CUDA)

            prediction = get_result(
                prediction, self.confidence, self.num_classes, nms_conf=self.nms)

            end = time.time()
            if type(prediction) == int:
                for img_num, image in enumerate(image_list[i * self.batch_size: min((i + 1) * self.batch_size, len(image_list))]):
                    img_id = i * self.batch_size + img_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(
                        image.split("/")[-1], (end - start) / self.batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("*********************************************")
                continue

            # transform attr from index in batch to index in image list
            prediction[:, 0] += i * self.batch_size
            if not check:           # initialize output
                output = prediction
                check = 1
            else:
                output = torch.cat((output, prediction))

            for img_num, image in enumerate(image_list[i * self.batch_size: min((i + 1) * self.batch_size, len(image_list))]):
                img_id = i * self.batch_size + img_num
                objects = [self.classes[int(x[-1])]
                           for x in output if int(x[0]) == img_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start) / self.batch_size))
                print("{0:20s} {1:s}".format(
                    "Objects Detected:", " ".join(objects)))
                print("*********************************************")

            if self.CUDA:
                torch.cuda.synchronize()

        # draw bouding boxes on images
        try:
            output
        except NameError:
            print("No detection were made")
            exit()

        img_dim_list = torch.index_select(img_dim_list, 0, output[:, 0].long())
        scale_factor = torch.min(
            self.input_dim / img_dim_list, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (self.input_dim - scale_factor *
                              img_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.input_dim - scale_factor *
                              img_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scale_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(
                output[i, [1, 3]], 0.0, img_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(
                output[i, [2, 4]], 0.0, img_dim_list[i, 1])

        output_recast_time = time.time()
        class_load_time = time.time()
        colors = pkl.load(open(self.colors_file, "rb"))
        draw_time = time.time()

        list(map(lambda x: draw_result(x, loaded_img_list, colors, self.classes), output))
        detect_names = pd.Series(image_list).apply(
            lambda x: "{}/detect_{}".format(self.det, x.split("/")[-1]))
        list(map(cv2.imwrite, detect_names, loaded_img_list))

        end = time.time()
        print("Results")
        print("*********************************************")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print("{:25s}: {:2.3f}".format("Reading", load_batch_time - read_time))
        print("{:25s}: {:2.3f}".format("Loading batch",
              start_detect_loop_time - load_batch_time))
        print("{:25s}: {:2.3f}".format("Detection (" + str(len(image_list)
                                                           ) + " images)", output_recast_time - start_detect_loop_time))
        print("{:25s}: {:2.3f}".format("Output processing",
              class_load_time - output_recast_time))
        print("{:25s}: {:2.3f}".format("Drawing boxes", end - draw_time))
        print("{:25s}: {:2.3f}".format("Average time per img",
              (end - load_batch_time) / len(image_list)))

        torch.cuda.empty_cache()


test = ImageDetect()
test.get_detections()
