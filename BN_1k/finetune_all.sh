#!/bin/bash

GLOG_logtostderr=1 \
/usr/local/openmpi/bin/mpirun -np 4 ../caffe_fast_rcnn_fast/build/install/bin/caffe train \
--weights=./pretrain/bbox_256x256_ctx_32_multi_scale_full_polyak_7215_8933.caffemodel \
--solver=solver_M_region.prototxt  \
2>&1|tee bn_train_4d_multiregion.log


GLOG_logtostderr=1 \
/usr/local/openmpi/bin/mpirun -np 4 ../caffe_fast_rcnn_fast/build/tools/caffe train \
--weights=models/BN_M_region_iter_120000.caffemodel \
--solver=solver_GBD.prototxt \
2>&1|tee bn_train_4d_GBD.log
