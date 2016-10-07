function [label_t, boxreg_t, box_valid_label_t, mask_t] = prepare_label(imsize, bbox_new, param)
    fmsize = param.net_response_size(imsize(1:2));
    label_t = zeros([fmsize, size(param.anchor_box, 1), 1]);
    boxreg_t = zeros([fmsize, 4 * size(param.anchor_box, 1), 1]);
    box_valid_label_t = zeros([fmsize, 4 * size(param.anchor_box, 1), 1]);
    mask_t = ones(size(label_t));
    
    % pure false sample
    if isempty(bbox_new)
        return;
    end
    
    %best_points = [(bbox_new(:,1) + bbox_new(:,3))/2, (bbox_new(:,2) + bbox_new(:,4))/2];
    %best_points = round((best_points - param.anchor_center) / param.stride) + 1;
    %label_t(best_points(:,2),best_points(:,1),:) = 1;
    for y = 1 : fmsize(1)
        for x = 1 : fmsize(2)
            anchor_center_now = [(x-1)*param.stride (y-1)*param.stride] + param.anchor_center;
            anchor_box_now = param.anchor_box + ...
                    repmat([anchor_center_now(1) anchor_center_now(2) anchor_center_now(1) anchor_center_now(2)], ...
                    [size(param.anchor_box, 1) 1]);
            overlap_ratio = get_overlap_MtoN(bbox_new, anchor_box_now);
            label_now = overlap_ratio >= param.pos_overlap_ratio;
            if size(overlap_ratio, 1) == 1
                match_id = ones(size(overlap_ratio, 2), 1);
            else
                [~, match_id] = max(overlap_ratio);
                label_now = max(label_now);
            end
%             if max(label_now) == 0
%                 continue;
%             end
            label_t(y, x, :, 1) = label_t(y, x, :, 1) | reshape(label_now, 1, 1, [], 1);
            if max(label_t(y, x, :, 1)) == 0
                continue;
            end
            factor = 1;
            if param.multishape_anchor
                factor = 3;
            end
            for anchor_id = 1 : param.anchor_scale * factor
                if label_t(y, x, anchor_id, 1) ~= 1
                    continue;
                end
                % for bbox regress ccwh
                target_box = bbox_new(match_id(anchor_id),:);
                wscal = (target_box(3) - target_box(1)) / (anchor_box_now(anchor_id, 3) - anchor_box_now(anchor_id, 1));
                hscal = (target_box(4) - target_box(2)) / (anchor_box_now(anchor_id, 4) - anchor_box_now(anchor_id, 2));
                xshift = ((target_box(3) + target_box(1)) / 2 - anchor_center_now(1)) / (anchor_box_now(anchor_id, 3) - anchor_box_now(anchor_id, 1));
                yshift = ((target_box(4) + target_box(2)) / 2 - anchor_center_now(2)) / (anchor_box_now(anchor_id, 4) - anchor_box_now(anchor_id, 2));
                
                % set data
                boxreg_t(y, x, (anchor_id-1)*4+1:anchor_id*4, 1) = [xshift yshift wscal hscal];
                box_valid_label_t(y, x, (anchor_id-1)*4+1:anchor_id*4, 1) = 1;
                
            end
        end
    end
    
    % TODO: CONVERT ALL LABEL FOR C
    % Done this outside.
end