%% Testing RPN for getting proposals in detection task in ILSVRC2016
%   Any questions please contact liuyuisanai@gmail.com
%% Please follow the tips below:
%   1. Finish training and all things mentioned in "script_start.m"
%   2. Set your own configure in this file, items who need to be customized
%   are marked by "%todo: customize" 
%   3. Images for testing are list in a mat file named "list_xxx.mat",
%   modify it to your own list.
%   4. Run this script, results will be saved in worker*roi.mat, in which
%   '*' is a worker identity number for speed up, which is 1 by default.

%% load param
timestamp = strrep(mat2str(clock), ' ', '');
load(fullfile(pwd, 'output', 'ResNet-101L', 'param.mat'));
diary(fullfile(param.output_dir, ['test-' fullfile(timestamp) '.txt']));
param.init_model = fullfile(pwd, 'output/ResNet-101L/300000');%todo: customize
param.solverfile = 'solver_test.prototxt';
param.max_img = 1400;%todo: customize 
%% load list
% load list mannuly or
load('list_coco15.mat');
list_test = list;
param.dataset_root = '';
if ~exist('work_num', 'var')
    work_num = 1;%todo: customize
    work_id = 1;%todo: customize
end
param.gpu_id = work_id-1;
l = length(list_test)/work_num;
thisid = floor(l * (work_id-1)+1):ceil(l*work_id);
list = list_test(thisid);
fprintf('worker: %d\ / %d\n', work_id, work_num);
%% active mex
addpath(fullfile(pwd, 'bin', 'caffe'));
t = pwd;
cd(fullfile(pwd, 'bin', 'caffe', '+caffe', 'private'));
caffe.init_log(fullfile(param.caffe_log_dir, 'caffe'));
cd(t);
clear t;

%% init caffe solver
caffe.reset_all;
caffe_solver = caffe.get_solver(fullfile(param.net_def_dir, param.solverfile), param.gpu_id);
if ~isempty(param.init_model)
    assert(exist(param.init_model)==2, 'Cannot find caffemodel.');
    caffe_solver.use_caffemodel(param.init_model);
end
num = length(list);
test_num = ceil(num/length(param.gpu_id));
cons = 0;
hit = 0;
sums = 0;
missed = cell(num, 1);
recall = 0;
roi = cell(num, 1);
for i = 1 : test_num
    tic;
    drawnow;
    fprintf('Testing %d/%d...', i, test_num);
    now_id = mod((i-1)*length(param.gpu_id):i*length(param.gpu_id)-1, num) + 1;
    list_t = list(now_id);
    parsed_all = detect_all( list_t, param, caffe_solver, [-4:0], -7 );
    for ii = 1 : length(param.gpu_id)
        score = parsed_all(ii).cls_score;
        box = parsed_all(ii).box;
        box = [box, score];
        id = nms(box, 0.7);
        [~, ss] = sort(score(id), 'descend');
        if length(id) > 2000
            id = id(1:2000);
        end
        roi{now_id(ii)} = box(id,:);
    end
    toc;
end
save('-v7.3', ['worker' num2str(work_id) 'roi.mat'], 'roi', 'list');
diary off