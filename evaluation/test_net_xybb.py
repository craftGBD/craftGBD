#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 Sensetime, CUHK
# Written by Ross Girshick, Yang Bin, Wang Kun
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import argparse
import pprint
import time
import os
import sys
import cPickle
import numpy as np
import _init_paths
import caffe
#from fast_rcnn.test_nocontext import test_net
from fast_rcnn.test_xybb import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='ilsvrc_2013_val2', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--num_per_batch', dest='boxes_num_per_batch',
                        help='split boxes to batches',
                        default=0, type=int)
    parser.add_argument('--bbox_mean', dest='bbox_mean',
                        help='the mean of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_std', dest='bbox_std',
                        help='the std of bbox',
                        default=None, type=str)
    parser.add_argument('--svm', dest='svm', 
                        help='svm use or not', default=False, type=int)
    parser.add_argument('--startIdx', dest='startIdx', 
                        help='startIdx', default=0, type=int)
    parser.add_argument('--endIdx', dest='endIdx',
                        help='endIdx', default=-1, type=int)
    parser.add_argument('--thresh', dest='thresh', 
                        help='threshold', default=-10, type=float)
    parser.add_argument('--saveMat', dest='saveMat', 
                        help='save mat file or not', default=False, type=bool)
    parser.add_argument('--usewzctx', dest='usewzctx', 
                        help='use context feature of zwang or not', default=False, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print 'Using config:'
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print 'Waiting for {} to exist...'.format(args.caffemodel)
        time.sleep(10)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        cfg.GPU_ID = args.gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    #pdb.set_trace()
    if args.usewzctx:
        print "use  wzctx!!!"
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    #print "0"
    if args.bbox_mean and args.bbox_std:
        # apply bbox regression normalization on the net weights
        with open(args.bbox_mean, 'rb') as f:
            bbox_means = cPickle.load(f)
        with open(args.bbox_std, 'rb') as f:
            bbox_stds = cPickle.load(f)
       
        #print "0.5"
        net.params['bbox_pred_finetune'][0].data[...] = \
            net.params['bbox_pred_finetune'][0].data * bbox_stds[:, np.newaxis]
        net.params['bbox_pred_finetune'][1].data[...] = \
            net.params['bbox_pred_finetune'][1].data * bbox_stds + bbox_means

    #print "1"
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    #print "2"
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    #print "3"
    #print args.startIdx, args.endIdx
    test_net(net, imdb, max_per_image=args.max_per_image,
             boxes_num_per_batch = args.boxes_num_per_batch, vis=args.vis, startIdx =  args.startIdx, endIdx = args.endIdx, svm=bool(args.svm), thresh=args.thresh, use_wzctx=bool(args.usewzctx))
