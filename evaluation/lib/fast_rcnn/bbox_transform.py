# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import math
from utils.cython_bbox import bbox_overlaps

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def bbox_voting(cls_dets_after_nms, cls_dets, threshold):
    """
    A nice trick to improve performance durning TESTING.
    Check 'Object detection via a multi-region & semantic segmentation-aware CNN model' for details.
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(cls_dets_after_nms[:, :4], dtype=np.float),
        np.ascontiguousarray(cls_dets[:, :4], dtype=np.float))
    for i in xrange(cls_dets_after_nms.shape[0]):
        candidate_bbox = cls_dets[overlaps[i, :] >= threshold, :]
        cls_dets_after_nms[i, :4] = np.average(candidate_bbox[:, :4], axis=0, weights=candidate_bbox[:, 4])
    return cls_dets_after_nms

def crop_boxes(boxes, crop_shape):
    """
    Crop boxes according given crop shape
    """

    crop_x1 = crop_shape[0]
    crop_y1 = crop_shape[1]
    crop_x2 = crop_shape[2]
    crop_y2 = crop_shape[3]

    l0 = boxes[:, 0] >= crop_x1
    l1 = boxes[:, 1] >= crop_y1
    l2 = boxes[:, 2] <= crop_x2
    l3 = boxes[:, 3] <= crop_y2

    L = l0 * l1 * l2 * l3
    cropped_boxes = boxes[L, :]

    cropped_boxes[:, 0] = cropped_boxes[:, 0] - crop_x1
    cropped_boxes[:, 1] = cropped_boxes[:, 1] - crop_y1
    cropped_boxes[:, 2] = cropped_boxes[:, 2] - crop_x1
    cropped_boxes[:, 3] = cropped_boxes[:, 3] - crop_y1

    return cropped_boxes

def crop_boxes_inv(cropped_boxes, crop_shape):
    """
    Inverse operation of crop_boxes
    """

    crop_x1 = crop_shape[0]
    crop_y1 = crop_shape[1]

    raw_boxes = np.zeros_like(cropped_boxes)

    raw_boxes[:, 0::4] = cropped_boxes[:, 0::4] + crop_x1
    raw_boxes[:, 1::4] = cropped_boxes[:, 1::4] + crop_y1
    raw_boxes[:, 2::4] = cropped_boxes[:, 2::4] + crop_x1
    raw_boxes[:, 3::4] = cropped_boxes[:, 3::4] + crop_y1

    return raw_boxes

def cal_crop_shape(boxes, height, width, padding=0):
    """
    """
    crop_x1 = math.floor(max(0, np.min(boxes[:, 0]) - padding))
    crop_y1 = math.floor(max(0, np.min(boxes[:, 1]) - padding))
    crop_x2 = math.ceil(min(width - 1, np.max(boxes[:, 2]) + padding))
    crop_y2 = math.ceil(min(height -1, np.max(boxes[:, 3]) + padding))

    crop_shape = np.array([crop_x1, crop_y1, crop_x2, crop_y2])
    return crop_shape
