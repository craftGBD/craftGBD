# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox_voting
import argparse
import math
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
import scipy.io as sio
from utils.blob import im_list_to_blob, im_list_to_fixed_spatial_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # blob = im_list_to_blob(processed_ims)
    blob = im_list_to_fixed_spatial_blob(processed_ims, cfg.TEST.MAX_SIZE, cfg.TEST.MAX_SIZE)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect_split(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs_all, im_scales = _get_blobs(im, boxes)
    num_boxes = boxes.shape[0]
    scores = np.zeros((num_boxes, 201), dtype=np.float32)
    box_deltas = np.zeros((num_boxes, 4*201), dtype=np.float32)
    for i in xrange(blobs_all['data'].shape[0]):
        # load blobs
        inds = np.where(blobs_all['rois'][:, 0] == i)[0]
        if inds.shape[0] == 0:
            continue
        blobs = {'data' : None, 'rois' : None}
        blobs['data'] = blobs_all['data'][[i]]
        blobs['rois'] = blobs_all['rois'][inds]
        blobs['rois'][:, 0] = 0

        # reshape network inputs
        net.blobs['data'].reshape(*(blobs['data'].shape))
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

        # do forward
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
        blobs_out = net.forward(**forward_kwargs)

        # use softmax estimated probabilities
        score = blobs_out['cls_prob']
        scores[inds] = score

        box_delta = blobs_out['bbox_pred']
        box_deltas[inds] = box_delta

    # Apply bounding-box regression deltas
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def im_detect(net, im, boxes=None, svm=False, layer_name='cls_prob'):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

   # blobs, unused_im_scale_factors = _get_blobs(im, boxes)
   # rois_image = [0, 0, 0, im.shape[1], im.shape[0]] * unused_im_scale_factors
   # blobs['rois'] = np.vstack((rois_image, blobs['rois']))
   # net.blobs['data'].reshape(*(blobs['data'].shape))
   # net.blobs['rois'].reshape(*(blobs['rois'].shape))
   # blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
   #  rois=blobs['rois'].astype(np.float32, copy=False))

    blobs, im_scales = _get_blobs(im, boxes)
    
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    
    #zwang context code
    #rois_image = [0, 0, 0, im.shape[1], im.shape[0]] * im_scales
    #blobs['rois'] = np.vstack((rois_image, blobs['rois']))
    ####end

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    #print "display shapes:"
    #print blobs['data'].shape
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        #print blobs['im_info'].shape
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))
        #print blobs['rois'].shape

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    #print "net.forward"
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM or svm:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = net.blobs[layer_name].data 

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred'].copy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.TEST.LR_FLIP:
        flip_im = np.fliplr(im)
        im_height, im_width, _ = im.shape
        flip_boxes = boxes.copy()
        flip_boxes[:, 2] = im_width - 1 - boxes[:, 0]
        flip_boxes[:, 0] = im_width - 1 - boxes[:, 2]

        flip_blobs, im_scales = _get_blobs(flip_im, flip_boxes)

        # reshape network inputs
        net.blobs['data'].data[...] = flip_blobs['data']
        net.blobs['rois'].data[...] = flip_blobs['rois']
        flip_blobs_out = net.forward()

        flip_scores = flip_blobs_out['cls_prob']
        flip_box_deltas = flip_blobs_out['bbox_pred']
        flip_box_deltas[:, 0::4] = -flip_box_deltas[:, 0::4]

        scores = (scores + flip_scores) / 2
        box_deltas = (box_deltas + flip_box_deltas) / 2
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return net, scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def scores_doping(scores, bbox_top_n=10):
    if bbox_top_n <= 0:
        return []
    scores_flatten = np.ravel(scores).copy()
    bbox_inds, cls_inds = np.unravel_index(scores_flatten.argsort(), scores.shape)
    top_classes = np.unique(cls_inds[-bbox_top_n:])
    return top_classes

def test_net(net, imdb, layer_name='cls_prob', max_per_image=100, thresh=0.05, boxes_num_per_batch=0, vis=False, startIdx=0, endIdx=-1, saveMat=False, svm=0):
    """Test a Fast R-CNN network on an image database."""
    
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    #print "4"
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    #print "5"
    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
    
    if endIdx==-1:
        endIdx=num_images
    #print "6"
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if i < startIdx or i>=endIdx:
            continue
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            #print "x"
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
            #print "y"
        im_name = imdb.image_path_at(i)
        im_name = im_name.split('/')[-1]
        im_name = im_name.split('.')[0]
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()

        thresh_c = [[] for _ in xrange(4)]
        thresh_c[1] = 0.9
        thresh_c[2] = 0.1
        thresh_c[3] = 0.05
        #print "boxes_num %d"%boxes_num_per_batch
        if boxes_num_per_batch > 0:
            num_boxes = box_proposals.shape[0]
            num_batch = (num_boxes + boxes_num_per_batch -1) / boxes_num_per_batch
            #print "zzz"
            #num_boxes = roidb[i]['boxes'].shape[0]
            #num_batch = math.ceil(num_boxes/boxes_num_per_batch)
            scores_batch = np.zeros((num_batch*boxes_num_per_batch, imdb.num_classes), dtype=np.float32)
            boxes_batch = np.zeros((num_batch*boxes_num_per_batch, 4*imdb.num_classes), dtype=np.float32)
            # replicate the first box num_batch*boxes_num_per_batch times for preallocation
            rois = np.tile(box_proposals[0, :], (num_batch*boxes_num_per_batch, 1))         
            #print "xx"
            # assign real boxes to rois
            rois[:num_boxes, :] = box_proposals
            #print "num_batch: %d"%num_batch
            for j in xrange(int(num_batch)):
               roi = rois[j*boxes_num_per_batch:(j+1)*boxes_num_per_batch, :]
               #print roi.shape

               f_name_str_bb = 'bb' + str(i) + '_' + str(j) + '.pkl'
               fname_bb = os.path.join(output_dir, f_name_str_bb)
               if os.path.isfile(fname_bb):
                   with open(fname_bb, 'rb') as f:
                       scores1 = cPickle.load(f)
                       scores2 = cPickle.load(f)
                       scores3 = cPickle.load(f)
                       #scores4 = cPickle.load(f)
                       box= cPickle.load(f)
                       f.close()
               else:
                   net, score, box = im_detect(net, im, roi, svm, layer_name)
                   scores1 = net.blobs['cls_prob'].data
                   scores2 = net.blobs['cls_prob_192'].data
                   scores3 = net.blobs['cls_prob_128'].data
#                with open(fname_bb, 'wb') as f:
#                    cPickle.dump(scores1, f, cPickle.HIGHEST_PROTOCOL)
#                    cPickle.dump(scores2, f, cPickle.HIGHEST_PROTOCOL)
#                    cPickle.dump(scores3, f, cPickle.HIGHEST_PROTOCOL)
#                    cPickle.dump(box, f, cPickle.HIGHEST_PROTOCOL)
#                    f.close()
               scores1[:, 0] = 0
               scores2[:, 0] = 0
               scores3[:, 0] = 0
               Flag1 = scores1 > thresh_c[1]
               Flag2 = scores2[:, 0] > thresh_c[2]
               Flag3 = scores3[:, 0] > thresh_c[3]

               inds1 = np.where(Flag1)
               #inds2 = np.where(Flag1 & Flag2)
               #inds3 = np.where(Flag1 & Flag2 & Flag3)
               if i < 0:
                   print "scores: "
                   print inds1
#                   print inds1[1]
                   print scores1[inds1]
               score = scores3



#                score, box = im_detect(net, im, roi, svm)
               scores_batch[j*boxes_num_per_batch:(j+1)*boxes_num_per_batch, :] = score# [:,:,0,0]
               boxes_batch[j*boxes_num_per_batch:(j+1)*boxes_num_per_batch, :] = box
               # print "6_%d"%j
            # discard duplicated results
            scores = scores_batch[:num_boxes, :]
            #print "kx"
            boxes = boxes_batch[:num_boxes, :]
        else:
            #print box_proposals.shape[0]
            scores, boxes = im_detect(net, im, box_proposals)
        mat_dir = os.path.join(output_dir, 'stage%s'%startIdx)
        if not os.path.exists(mat_dir):
            os.mkdir(mat_dir)
#        if True:
#            sio.savemat('%s/%s.mat' % (mat_dir,im_name + '_' + str(i) ), {'scores': scores, 'boxes': boxes})
        
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        #print "7"        
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            if cfg.TEST.BBOX_VOTE:
                cls_dets_after_nms = cls_dets[keep, :]
                cls_dets = bbox_voting(cls_dets_after_nms, cls_dets, threshold=cfg.TEST.BBOX_VOTE_THRESH)
            else:
                cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()
  	if svm: 
            print 'svm im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
	else:
            print 'softmax im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    #det_file = os.path.join(output_dir, 'detection_%sto%s.pkl' % (startIdx,endIdx))
    #with open(det_file, 'wb') as f:
    #    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir, startIdx, endIdx)
    print 'Done, saved to'
    print output_dir
    
