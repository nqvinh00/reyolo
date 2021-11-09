import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from helpers import build_targets, to_cpu


class Upsample(pl.LightningModule):
    """
    nn.Upsample is deprecated
    """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class FocalLoss(pl.LightningModule):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, t):
        loss = self.loss_fcn(pred, t)
        prediction = torch.sigmoid(pred)
        pt = t * prediction * (1 - t) * (1 - prediction)
        alpha_factor = t * self.alpha + (1 - t) * (1 - self.alpha)
        m_factor = (1 - pt) ** self.gamma
        loss *= alpha_factor * m_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.redduction == "sum":
            return loss.sum()
        else:
            return loss


class DetectionLayer(pl.LightningModule):
    """
    Use for yolo module
    """

    def __init__(self, anchors, num_classes, apply_focal_loss, image_dim):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.apply_focal_loss = apply_focal_loss
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.image_dim = image_dim
        self.grid_size = 0
        self.focal_loss = FocalLoss(self.bce_loss, gamma=1.5, alpha=0.25)

    def compute_bce_loss(self, inputs, targets, apply_focal_loss):
        if apply_focal_loss:
            self.no_obj_scale = 1
            return self.focal_loss(inputs, targets)
        else:
            self.bce_loss = nn.BCELoss()
            return self.bce_loss(inputs, targets)

    def compute_grid_offsets(self, grid_size, CUDA=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
        self.stride = self.image_dim / self.grid_size

        self.grid_x = torch.arange(g).repeat(
            g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(
            g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, image_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.image_dim = image_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (x.view(num_samples, self.num_anchors, self.num_classes +
                      5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())

        # center x, y
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # width, height
        w = prediction[..., 2]
        h = prediction[..., 3]

        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, CUDA=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        res = (pred_boxes.view(num_samples, -1, 4) * self.stride, pred_conf.view(
            num_samples, -1, 1), pred_cls.view(num_samples, -1, self.num_classes),)
        output = torch.cat(res, -1)

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # loss
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.compute_bce_loss(
                pred_conf[obj_mask], tconf[obj_mask], self.apply_focal_loss)
            loss_conf_no_obj = self.compute_bce_loss(
                pred_conf[no_obj_mask], tconf[no_obj_mask], self.apply_focal_loss)
            loss_conf = self.obj_scale * loss_conf_obj + \
                self.no_obj_scale * loss_conf_no_obj
            loss_cls = self.compute_bce_loss(
                pred_cls[obj_mask], tcls[obj_mask], self.apply_focal_loss)
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_no_obj = pred_conf[no_obj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            precision = torch.sum(iou50 * detected_mask) / \
                (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / \
                (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / \
                (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_no_obj": to_cpu(conf_no_obj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss
