import torch
import numpy as np


def predict_transform(predict, input_dim, anchors, num_classes, CUDA=True):
    """
    Transfer input into 2d tensor.
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

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # sigmoid
    predict[:, :, 0] = torch.sigmoid(predict[:, :, 0])
    predict[:, :, 1] = torch.sigmoid(predict[:, :, 1])
    predict[:, :, 4] = torch.sigmoid(predict[:, :, 4])

    # add center offsets
    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x).view(-1, 1)
    y_offset = torch.FloatTensor(y).view(-1, 1)
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        anchors = anchors.cuda()

    xy_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)
    anchors = anchors.repeat(grid_size ** 2, 1).unsqueeze(0)
    predict[:, :, :2] += xy_offset
    predict[:, :, 5: num_classes +
            5] = torch.sigmoid(predict[:, :, 5: num_classes + 5])
    predict[:, :, :4] *= stride

    return predict
