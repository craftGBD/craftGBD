function parsed = convert_output_to_struct( param, handle, issolver, thresh_cls)
    
    factor = 1;
    if param.multishape_anchor
        factor = 3;
    end
    if nargin < 4
        thresh_cls = 0;
    end
    if issolver
        handle = handle.nets;
    else
        handle = {handle};
    end
    parsed(length(handle)) = struct(...
        'cls_score', zeros(0, 1),...
        'box', zeros(0, 4),...
        'active', []);
    for gpu_id = 1 : length(handle)
        
        %fetch output
        cls = convert_caffe2mat(handle{gpu_id}.blob_vec(handle{gpu_id}.name2blob_index('rpn_cls')).get_data());
        box = convert_caffe2mat(handle{gpu_id}.blob_vec(handle{gpu_id}.name2blob_index('rpn_bbox')).get_data());
        %pts = convert_caffe2mat(handle{gpu_id}.blob_vec(handle{gpu_id}.name2blob_index('rpn_pts')).get_data());
        %pts_valid = convert_caffe2mat(handle{gpu_id}.blob_vec(handle{gpu_id}.name2blob_index('rpn_pts_valid')).get_data());
        fmwidth = size(cls, 2);
        fmheight = size(cls, 1);
        % reshape to [h w anchor]
        cls = reshape(cls, fmheight, fmwidth, param.anchor_scale * factor, 1);
        box = reshape(box, fmheight, fmwidth, 4, param.anchor_scale * factor);
        %pts = reshape(pts, fmheight, fmwidth, length(param.anchor_point), param.anchor_scale);
        %pts_valid = reshape(pts_valid, fmheight, fmwidth, length(param.anchor_point)/2 , param.anchor_scale);
        % for each anchor scale
        for anchor_id = 1 : param.anchor_scale * factor
            anchor_box_len = [param.anchor_box(anchor_id,3) - param.anchor_box(anchor_id,1), param.anchor_box(anchor_id,4) - param.anchor_box(anchor_id,2)];
            [validy, validx] = find(cls(:,:,anchor_id) >= thresh_cls);
            score_t{anchor_id} = [];
            target_box_t{anchor_id} = [];
            score_t{anchor_id} = diag(cls(validy,validx,anchor_id));
            for valid_id = 1 : length(validy)
                anchor_center_now = [(validx(valid_id)-1)*param.stride (validy(valid_id)-1)*param.stride] + param.anchor_center;
                % for box
                box_delta = reshape(box(validy(valid_id), validx(valid_id), :, anchor_id), 1, []);
                target_box_center = anchor_center_now + box_delta(1:2) .* anchor_box_len;
                target_box_length = anchor_box_len .* box_delta(3:4);
                target_box = [  1,  0,  -0.5,   0;
                                0,  1,  0,   -0.5;
                                1,  0,  0.5,    0;
                                0,  1,  0,    0.5   ] * [target_box_center, target_box_length]';
                target_box_t{anchor_id}(valid_id,:) = target_box';
            end
        end
        parsed(gpu_id).active = cls;
        parsed(gpu_id).cls_score = cell2mat(score_t');
        parsed(gpu_id).box = cell2mat(target_box_t');
        %parsed(gpu_id).point = cell2mat(target_pts_t');
        % target_pts = pts_new(match_id(anchor_id),:);
        % anchor_point_now = (ptsmean-0.5) * (anchor_box_now(anchor_id, 3) - anchor_box_now(anchor_id, 1)) + anchor_center_now;
        % ptsdiff = (target_pts - anchor_point_now) / (anchor_box_now(anchor_id, 3) - anchor_box_now(anchor_id, 1));
        
    end
end

