prefetch_num = 8;
if param.speedup
    p = new_parpool(prefetch_num);
    parHandle = cell(prefetch_num, 1);
end
diary(fullfile(param.out_dir, 'diary.txt'));
losses = [];
recalls = [];
loss_weight = [10,1];

%% calculate net attribute
attribute = get_net_attr(fullfile(param.net_def_dir, param.netnolossfile), param.max_img);
param.receptive_field = find(attribute == 1, 1, 'first');
param.stride = find(attribute == 2, 1, 'first') - param.receptive_field;
param.net_response_size = attribute;
fprintf('\n\nreceptive_field:%d\nstride:%d\n\n', param.receptive_field, param.stride);
param.min_img = max(param.min_img, param.receptive_field);
param.anchor_center = (param.stride-1)/2;

%% init caffe solver
caffe.reset_all;
caffe_solver = caffe.get_solver(fullfile(param.net_def_dir, param.solverfile), param.gpu_id);
if ~isempty(param.init_model)
    assert(exist(param.init_model)==2, 'Cannot find caffemodel.');
    caffe_solver.use_caffemodel(param.init_model);
end
history = cell(length(caffe_solver.nets{1}.outputs), 1);
now_iter = caffe_solver.iter();
max_iter = caffe_solver.max_iter();
if isfield(boxdb, 'test_ids')
    val_id_all = boxdb.test_ids;
else
    val_id_all = randperm(length(boxdb.list), round(length(boxdb.list)*param.validation_ratio));
end
if param.downsample
    tr_id = param.inds_train_down;
else
    tr_id = setdiff(1:length(boxdb.list), val_id_all);
end
tr_num = length(tr_id);
val_num = length(val_id_all);
if  ~exist('val_id', 'var')
    if param.fast_valid == 1 
        val_id = val_id_all(randperm(length(val_id_all), param.val_num));
    else
        val_id = val_id_all;
    end
end
if ~exist('remainid', 'var')
    remainid = [];
end

if param.speedup
    for i = 1 : prefetch_num
        [thisid, remainid] = randomid(tr_num, remainid, length(param.gpu_id));
        parHandle{i} = parfeval(p, @prepare_data_for_input, 1, param.dataset_root, boxdb.list(tr_id(thisid)), boxdb.bbox(tr_id(thisid)), ...
            param);
    end
end
save(fullfile(param.output_dir, 'param'), 'param');

%% start train
tic;
for iter = now_iter : max_iter
    drawnow;
    [thisid, remainid] = randomid(tr_num, remainid, length(param.gpu_id));
    if param.speedup 
        input_data = fetchOutputs(parHandle{mod(iter, prefetch_num) + 1});
        parHandle{mod(iter, prefetch_num) + 1} = parfeval(p, @prepare_data_for_input, 1, param.dataset_root, boxdb.list(tr_id(thisid)), boxdb.bbox(tr_id(thisid)), ...
            param);
    else
        input_data = prepare_data_for_input(param.dataset_root, boxdb.list(tr_id(thisid)), boxdb.bbox(tr_id(thisid)), ...
            param);
    end
    caffe_solver.reshape_as_input(input_data);
    caffe_solver.set_input_data(input_data);
    caffe_solver.step(1);
    now_iter = caffe_solver.iter();
    output = fetch_output(caffe_solver);
    for t = 1 : length(output)
        history{t}(end+1) = output(t).value;
    end
    if  mod(iter, param.display_interval) == 0 && iter > 0
        fprintf('Iter %d: ', iter);
        for t = 1 : length(output)
            temp = history{t}(end - param.display_interval + 1 : end);
            temp = temp(temp~=0);
            tt(t) = mean(temp);
            fprintf('%s: %.3f  ', output(t).name, mean(temp));
        end
        losses(end+1) = tt*loss_weight';
        figure(1)
        hold on
        plot(losses, 'b');
        grid on;
        toc;
        tic;
    end
    if  mod(iter, param.snapshot_interval) == 0
        caffe_solver.snapshot(fullfile(param.output_dir, num2str(iter)));
    end
end


diary off;