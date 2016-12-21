%% Training RPN for getting proposals in detection task in ILSVRC2016
%   Any questions please contact liuyuisanai@gmail.com
%% Please follow the tips below:
%   1. You need download dependent files (our annotations / pre-trained
%   models / compiled caffemex(on windows)) on cloud drive. The compiled
%   caffemex only support running on windows. Please install MATLAB 2014a
%   or later version and VS2013 for good compatibility.
%   2. Copy annotation file "inds1314old_1500new_flip_*.mat" in ./annotation/
%   3. Copy caffe/ to ./bin/
%   4. Set your own configure in this file, items who need to be customized
%   are marked by "%todo: customize" 
%   5. Run script_start.m
%   6. For testing, run get_proposal_from_list.m.
%   7. For evaluation, run evaluate_recall.m. The first input 'parsed' is
%   produced by 'get_proposal_from_list.m', in which you need save all
%   'parsed_all' in parsed(i) in each loop.


%% load data annotation
param.anno_trainval = 'boxdb';
param.anno_train_dir = 'ilsvrc16_trainval';
param.anno_test_dir = 'ilsvrc16_val2';
param.dataset_root = 'G:\temp\imagenet_2016';%todo: customize
param.downsample = 1;
if param.downsample
    load('annotation/inds1314old_1500new_flip_train.mat');
    load('annotation/inds1314old_1500new_flip_val.mat');
    param.inds_train_down = inds_train;
    param.inds_val_down = inds_val;
    clear inds_train inds_val;
end

%% load net
param.model_name = 'ResNet-101L';
param.solverfile = 'solver_train.prototxt';
param.netnolossfile = 'net_noloss.prototxt';
assert(~isempty(param.netnolossfile));
param.net_def_dir = fullfile(pwd, 'model', 'rpn_prototxts', param.model_name);
param.output_dir = fullfile(pwd, 'output', param.model_name);

%% configure training setting
param.gpu_id = 0; %todo: customize
param.batch_size_per_gpu = 1;
param.speedup = 1; %todo: customize
param.max_img = 1000; %todo: customize
param.min_img = 16;
param.max_target = 512;
param.min_target = 16;
param.val_num = 128;
param.fast_valid = 1;
param.test_interval = 5000;
param.snapshot_interval = 5000;
param.display_interval = 100;
param.anchor_scale = 5;
param.multishape_anchor = true;
param.pos_overlap_ratio = 0.5;
param.gray_augment_ratio = 0.5;
param.validation_ratio = 0.01;

%% configure anchor
param.anchor_box = prepare_anchor_box(param);

%% configure init setting
param.init_model = 'model\pre_trained_models\ResNet-101L\init_model.caffemodel'; % This model is the pre-trained model of resnet-101
param.out_dir = fullfile(pwd, 'output', param.model_name);
param.caffe_log_dir = fullfile(param.out_dir, 'caffe_log/');
if ~exist(param.out_dir)
    mkdir(param.out_dir);
end
if ~exist(param.caffe_log_dir)
    mkdir(param.caffe_log_dir);
end
    
    
%% load data annotation
if ~exist('boxdb', 'var') && ~isempty(param.anno_trainval)
    load(fullfile(pwd, 'annotation', param.anno_trainval));
end
load('annotation\id_val2.mat');
if param.downsample
    boxdb.test_ids = param.inds_val_down;
else
    boxdb.test_ids = val2_inds;
end
clear val2_inds;

%% active mex
addpath(fullfile(pwd, 'bin', 'caffe'));
t = pwd;
cd(fullfile(pwd, 'bin', 'caffe', '+caffe', 'private'));
caffe.init_log(fullfile(param.caffe_log_dir, 'caffe'));
cd(t);
clear t;
script_train_p_rpn;