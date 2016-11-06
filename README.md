#Structure of this README
1. Folder meanings
2. Preparation for Training & Testing
3. Tesing the models on ImageNet val2 data
4. Training the models on ImageNet data
5. Generating proposals
6. Pre-trained models
7. Contacts

## --------------------------- Folder meanings ---------------------------
### BN_1k
The folder that contains the trained GBD-Net based on BN-Net. Training code is also provided.

### ResNet-GBD
The folder that contains the trained GBD-Net based on ResNet-269.

### caffe_fast_rcnn_fast
The caffe code used for learning and testing.

### rois
The folder that contains generated region proposal for training the models.

### evaluation
The python code used for testing. It contains images and region proposals.

### proposal_gen
The matlab code used for generating proposals.

### fetch_data
Scripts used for fetching additional data from cloud drive.

## ----------------- Preparation for Training & Testing ------------------
1. run fetch_data/fetch_eval_data.m to download test images and scripts.
2. run fetch_data/fetch_BN_data.m to download pre-trained models for GBD-Net based on BN-Net.
3. run fetch_data/fetch_ResNet_data.m to download pre-trained models for GBD-Net based on ResNet-269.
4. run fetch_data/fetch_roi_data.m to download the generated proposals for training. You can generate the proposals by yourself, in this case please refer to README in proposal_gen folder.

## -------------- Tesing the models on ImageNet val2 data ----------------
There are two models released, GDB-Net based on BN-Net and GDB-Net based on ResNet-269.
The following steps show how to test GDB-Net based on BN-Net, steps for ResNet-269 is slightly different.

### Test GDB-Net based on BN-Net
1. Go to the "evaluation" folder.
cd evaluation

2. modify the script "run_test_multiGPU_BN_GBD.sh".
you can modify the "GPU" list to adapt your hardware configuration.

3. run the script.
sh ./run_test_multiGPU_BN_GBD.sh

4. Concatnate the results into one res.txt file. Remember to delete former generated res.txt before concatnation.
cat output/craft_ilsvrc/ilsvrc_2013_val2/BN_GBD_iter_120000/*.txt >> res.txt

5. Go to the "ILSVRC2014_devkit" folder.
cd ILSVRC2014_devkit

6. modify the script "demo_eval_det.m", make sure that "pred_file" is pointed to "res.txt".
pred_file = '../res.txt';

7. run "demo_eval_det.m" in matlab to evaluate, the mean AP is 53.5.
\>>run demo_eval_det

### Test the fast version of GBD-Net based on ResNet-269

To Test the fast version of GBD-Net based on ResNet-269, the corresponding script is "run_test_multiGPU_ResNet_GBD_fast.sh", and the results are located in "output/craft_ilsvrc/ilsvrc_2013_val2/ResNet-269-GBD_iter_180000/*.txt"

The mean AP is 60.6.

### Test the accurate version of GBD-Net based on ResNet-269
To Test the accurate version of GBD-Net based on ResNet-269, the corresponding script is "run_test_multiGPU_ResNet_GBD_accurate.sh", and the results are located in "output/craft_ilsvrc/ilsvrc_2013_val2/ResNet-269-GBD_iter_180000/*.txt"

The mean AP is 63.7.

##--------- Training the GBD-Net model using ImageNet data ---------------
1. Go to the folder "BN_1k".
cd BN_1k

2. finetune the model by running the shell:
sh ./finetune_all.sh

Note: Fintuning has two stages.

1. finetuning a multi-region BN-net from a pretrained BN-net, which is
pretrain/bbox_256x256_ctx_32_multi_scale_full_polyak_7215_8933.caffemodel

2. finetuning the GBD-Net from the multi-region BN-net, which is
models/BN_M_region_iter_120000.caffemodel

3. after the above two stages, the final model is
models/BN_GBD_iter_120000.caffemodel

##-------------------------- Generating proposals ------------------------
Please refer to README in proposal_gen folder

##--------------------------- Pre-trained models -------------------------
These are models trained by ourselves with identity mapping & stochastic depth.

|                            | ResNet-101 | ResNet-152 | ResNet-269 |
| -------------------------- |:----------:| :---------:| ----------:|
| Top-1 accuracy (single crop) | 78.21%     | 79.39%     | 80.34%     |
| Top-5 accuracy (single crop) | 93.95%     | 94.62%     | 95.04%     |

**Download**

ResNet-101: [GoogleDrive](https://drive.google.com/drive/folders/0B67_d0rLRTQYd1NTTi1nWE9US2M?usp=sharing)

ResNet-152: [GoogleDrive](https://drive.google.com/drive/folders/0B67_d0rLRTQYX2FMMFg1QU5MYTA?usp=sharing)

ResNet-269: [GoogleDrive](https://drive.google.com/drive/folders/0B67_d0rLRTQYM0FRVk9KT3laSGM?usp=sharing)

**Notice**

These models were trained with a modified caffe(https://github.com/yjxiong/caffe/tree/mem), which is different in BN layer with the offical version.

##------------------------------ Contacts ---------------------------------
For details about GBD, please contact Wanli Ouyang(wlouyang@ee.cuhk.edu.hk)

For details about code usage, please contact Yucong Zhou(zhouyucong@sensetime.com)

For details about pre-trained models, please contact Tong Xiao(xiaotong@ee.cuhk.edu.hk)
