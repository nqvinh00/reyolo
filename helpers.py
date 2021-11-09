import torch
from torch.autograd import Variable
import numpy as np
import cv2
import random
import os
import shutil
import fnmatch


def save_code_files(output_path, root_path):
    def match_patterns(include, exclude):
        def _ignore_patterns(path, names):
            # If current path in exclude list, ignore everything
            if path in set(name for pattern in exclude for name in fnmatch.filter([path], pattern)):
                return names
            # Get initial keep list from include patterns
            keep = set(
                name for pattern in include for name in fnmatch.filter(names, pattern))
            # Add subdirectories to keep list
            keep = set(
                list(keep) + [name for name in names if os.path.isdir(os.path.join(path, name))])
            # Remove exclude patterns from keep list
            keep_ex = set(
                name for pattern in exclude for name in fnmatch.filter(keep, pattern))
            keep = [name for name in keep if name not in keep_ex]
            # Ignore files not in keep list
            return set(name for name in names if name not in keep)

        return _ignore_patterns

    dst_dir = os.path.join(output_path, "code")
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(root_path, dst_dir, ignore=match_patterns(include=['*.py', '*.data', '*.cfg'],
                                                              exclude=['experiment*',
                                                                       '*.idea',
                                                                       '*__pycache__',
                                                                       'weights',
                                                                       'wandb',
                                                                       'asets'
                                                                       ]))


def predict_transform(predict, input_dim, anchors, num_classes, CUDA=False):
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

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    xy_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, len(anchors)).view(-1, 2).unsqueeze(0)
    predict[:, :, :2] += xy_offset

    # apply anchors to dimensions of bounding box
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size ** 2, 1).unsqueeze(0)

    predict[:, :, 2: 4] = torch.exp(predict[:, :, 2: 4]) * anchors
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


def get_result(prediction, confidence, num_classes, nms_conf=0.4):
    # object confidence thresholding
    # each bounding box having objectness score below a threshold
    # set the value of entrie row representing the bounding box to zero
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    # transform center_x, center_y, height, width of box
    # to top_left_corner_x, top_right_corner_y, right_bottom_corner_x, right_bottom_corner_y
    box = prediction.new(prediction.shape)
    box[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box[:, :, :4]

    batch_size = prediction.size(0)
    check = False

    # the number of true detections in every image may be different
    # confidence thresholding and nms has to be done for one image at conce
    # must loop over the 1st dimension of prediction
    for i in range(batch_size):
        image_prediction = prediction[i]      # image tensor

        # each bounding box has 85 attri
        # 80 attri are class scores
        max_confidence, max_confidence_score = torch.max(
            image_prediction[:, 5: num_classes + 5], 1)
        max_confidence = max_confidence.float().unsqueeze(1)
        max_confidence_score = max_confidence_score.float().unsqueeze(1)
        image_prediction = torch.cat(
            (image_prediction[:, :5], max_confidence, max_confidence_score), 1)

        non_zero = torch.nonzero(image_prediction[:, 4])
        try:
            image_prediction_ = image_prediction[non_zero.squeeze(
            ), :].view(-1, 7)
        except:
            continue

        if image_prediction_.shape[0] == 0:
            continue

        # get various classes detected in image
        image_classes = get_unique(image_prediction_[:, -1])

        for c in image_classes:
            # nms
            # get detections with 1 particular class
            class_mask = image_prediction_ * \
                (image_prediction_[:, -1] == c).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()
            image_prediction_class = image_prediction_[
                class_mask_index].view(-1, 7)

            # sort detection
            # confidence at top
            confidence_sorted_index = torch.sort(
                image_prediction_class[:, 4], descending=True)[1]
            image_prediction_class = image_prediction_class[confidence_sorted_index]
            index = image_prediction_class.size(0)

            for idx in range(index):
                # get ious of all boxes
                try:
                    ious = get_bounding_boxes_iou(image_prediction_class[idx].unsqueeze(
                        0), image_prediction_class[idx + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # mark zero all detections iou > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_prediction_class[idx + 1:] *= iou_mask

                # remove non-zero entries
                non_zero_index = torch.nonzero(
                    image_prediction_class[:, 4]).squeeze()
                image_prediction_class = image_prediction_class[non_zero_index].view(
                    -1, 7)

            batch_index = image_prediction_class.new(
                image_prediction_class.size(0), 1).fill_(i)
            s = batch_index, image_prediction_class

            if not check:
                output = torch.cat(s, 1)
                check = True
            else:
                output = torch.cat((output, torch.cat(s, 1)))

    try:
        return output
    except:
        return 0


def get_unique(tensor):
    np_tensor = tensor.cpu().numpy()
    unique = np.unique(np_tensor)
    unique_tensor = torch.from_numpy(unique)
    result = tensor.new(unique_tensor.shape)
    result.copy_(unique_tensor)

    return result


def get_bounding_boxes_iou(b1, b2):
    """
    Returns iou of 2 bouding boxes
    """

    # get coordinates of 2 bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    # get coordinates of overclap rectangle
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    # overclap area
    area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return area / (b1_area + b2_area - area)


def resize_image(img, input_dim):
    """
    resize image with unchanged aspect ratio using padding
    """
    width, height = img.shape[1], img.shape[0]
    w, h = input_dim
    new_width = int(width * min(w / width, h / height))
    new_height = int(height * min(w / width, h / height))
    resized_image = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((input_dim[1], input_dim[0], 3), 128)
    canvas[(h - new_height) // 2: (h - new_height) // 2 + new_height,
           (w - new_width) // 2: (w - new_width) // 2 + new_width, :] = resized_image
    return canvas


def pre_image(img, input_dim):
    """
    Prepare image as input for neural network
    """

    img = resize_image(img, (input_dim, input_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def draw_result(x, results, colors, classes):
    t1 = tuple(x[1: 3].int())
    t2 = tuple(x[3: 5].int())
    img = results[int(x[0])]
    text_font = cv2.FONT_HERSHEY_PLAIN
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{}".format(classes[cls])
    cv2.rectangle(img, t1, t2, color, 1)
    text_size = cv2.getTextSize(label, text_font, 1, 1)[0]
    t2 = t1[0] + text_size[0] + 3, t1[1] + text_size[1] + 4
    cv2.rectangle(img, t1, t2, color, -1)
    text_pos = t1[0], t1[1] + text_size[1] + 4
    cv2.putText(img, label, text_pos, text_font, 1, [255, 255, 255], 1)
    return img


def load_dataset(file_path):
    file = open(file_path, "r")
    names = file.read().split("\n")[:-1]
    return names


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    no_obj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # get anchors with best iou
    ious = torch.stack([bounding_box_wh_iou(anchor, gwh)
                       for anchor in anchors])
    _, best_n = ious.max(0)

    # separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # masks
    obj_mask[b, best_n, gj, gi] = 1
    no_obj_mask[b, best_n, gj, gi] = 0

    # set no obj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        no_obj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # one-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1

    # compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = get_bounding_boxes_iou(
        pred_boxes[b, best_n, gj, gi], target_boxes)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf


def bounding_box_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]

    area_1 = torch.min(w1, w2) * torch.min(h1, h2)
    area_2 = (w1 * h1 + 1e-16) + w2 * h2 - area_1
    return area_1 / area_2


def to_cpu(tensor):
    return tensor.detach().cpu()
