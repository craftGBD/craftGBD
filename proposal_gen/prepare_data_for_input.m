function [ input ] = prepare_data_for_input( root, list, box, param )
% input{1} : imgs
% input{2} : featuremap label
% input{3} : bbox regression
% input{4} : bbox valid
    input = cell(length(param.gpu_id), 1);
    for i = 1 : length(list)
        img_t = imread(fullfile(root, list{i}));
        box_t = box{i};

        if size(img_t,3) < 3
            img_t = img_t(:,:,[1 1 1]);
        end
        [img_t, bbox_new] = random_crop_img(img_t, box_t, param);
        imsize = size(img_t);
        [label_t, boxreg_t, box_valid_t, ~] = ...
            prepare_label(imsize, bbox_new, param);
        input{i}{1} = single(convert_img2caffe(img_t)) - 127.0;
        input{i}{2} = convert_mat2caffe(label_t);
        input{i}{3} = convert_mat2caffe(boxreg_t);
        input{i}{4} = convert_mat2caffe(box_valid_t);
    end
end
