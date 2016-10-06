#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
USESVM=0
THRESH=0.01

def=../BN_1k/deploy_GBD.prototxt
net=../BN_1k/models/BN_GBD_iter_120000.caffemodel
batch_size=150

saveMat="False"

GPU=(0 1 2 3 4 5 6 7 8 9 10 11)
GPU_NUM=${#GPU[@]}

Images_per_GPU=$((9920/GPU_NUM))

i=0

for GPU_ID in ${GPU[@]}; do
startIdx=$((i*Images_per_GPU))
endIdx=$((i*Images_per_GPU+Images_per_GPU))

nohup python ./test_net.py --gpu ${GPU_ID} \
     --def $def \
     --net $net \
     --imdb ilsvrc_2013_val2 \
     --cfg experiments/cfgs/ilsvrc_700.yml \
     --num_per_batch $batch_size \
     --startIdx $startIdx \
     --endIdx $endIdx \
     --svm 0 \
     --thresh $THRESH \
     --bbox_mean bbox_means.pkl \
     --bbox_std bbox_stds.pkl \
     --saveMat $saveMat &>  logBN_$((startIdx+1))_${endIdx}_log &

disown -h
i=$((i+1))

done

exit 0
