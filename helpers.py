import torch
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(predict, input_dim, anchors, num_classes, CUDA=True):
    """
    Transfer input (which is output of forward()) into 2d tensor.
    Each row of the tensor corresponds to attributes of a bounding box.
    """

    batch_size = predict.size(0)
    stride = input_dim // predict.size(2)
    grid_size = input_dim // stride
    bounding_box_attrs = num_classes + 5

    predict = predict.view(
        batch_size, bounding_box_attrs * len(anchors), grid_size ** 2)
    predict = predict.transpose(1, 2).contiguous()
    predict = predict.view(batch_size, grid_size ** 2 *
                           len(anchors), bounding_box_attrs)

    # dimensions of anchors are in accordance to height and width attr of net block
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # sigmoid x, y coordinates and objectness score
    # center_x, center_y, object_confidence
    predict[:, :, 0] = torch.sigmoid(predict[:, :, 0])
    predict[:, :, 1] = torch.sigmoid(predict[:, :, 1])
    predict[:, :, 4] = torch.sigmoid(predict[:, :, 4])

    # add center offsets
    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x).view(-1, 1)
    y_offset = torch.FloatTensor(y).view(-1, 1)
    # apply anchors to dimensions of bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        anchors = anchors.cuda()

    xy_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)
    anchors = anchors.repeat(grid_size ** 2, 1).unsqueeze(0)
    predict[:, :, :2] += xy_offset
    # apply sigmoid to class scores
    predict[:, :, 5: num_classes +
            5] = torch.sigmoid(predict[:, :, 5: num_classes + 5])
    # resize detections map to size of input image
    predict[:, :, :4] *= stride

    return predict


def test_input(file_path, img_size):
    img = cv2.imread(file_path)
    img = cv2.resize(img, img_size)
    img_result = img[:, :, ::-1].transpose((2, 0, 1))     # BGR -> RGB
    img_result = img_result[np.newaxis, :, :, :]/255.0    # Add a channel at 0
    img_result = torch.from_numpy(img_result).float()     # Convert to float
    img_result = Variable(img_result)                     # Convert to Variable
    return img_result


def get_result(prediction, confidence, num_classses, nms_conf=0.4):
    # object confidence thresholding
    # each bounding box having objectness score below a threshold
    # set the value of entrie row representing the bounding box to zero
    conf_mask = (prediction[:, :, :4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    # transform center_x, center_y, height, width of box
    # to top_left_corner_x, top_right_corner_y, right_bottom_corner_x, right_bottom_corner_y
    box = prediction.new(prediction.shape)
    box[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box[:, :, :4]
