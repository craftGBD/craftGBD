function [recall_dis, recall_con, missed, hit, cons, sums] = evaluate_recall(parsed, ptsdb, top_k, ratio, nms_ratio)
    if nargin < 4
        ratio = 0.5;
        nms_ratio = 0.7;
    end
    num = min(length(parsed), length(ptsdb.bbox));
    if nargin < 3
        top_k = 300;
    end
    sums = 0;
    hit = 0;
    cons = 0;
    missed = cell(0);
    for i = 1 : num
        bbox_predicted = parsed(i).box;
        score = parsed(i).cls_score;
        bbox_gt = ptsdb.bbox{i};
        nms_in = [bbox_predicted, score];
        sel_id = nms(nms_in, nms_ratio);
        score = score(sel_id);
        bbox_predicted = bbox_predicted(sel_id,:);
        if size(bbox_gt, 1) == 0
            missed{i} = zeros(0, 4, 'single');
            continue;
        end
        sums = sums + size(bbox_gt, 1);
        [~, id] = sort(score, 'descend');
        id = id(1:min(top_k, length(id)));
        bbox_predicted = bbox_predicted(id, :);
        score = score(id, :);
        overlap = get_overlap_MtoN(bbox_predicted, bbox_gt);
        label_used = zeros(size(bbox_gt, 1), 1);
        overlap = max(overlap);
        t = find(overlap>=ratio);
        label_used(t) = 1;
        cons = cons + sum(overlap(t));
        hit = hit + sum(label_used);
        missed{i} = bbox_gt(label_used==0,:);
    end
    recall_dis = hit / sums;
    recall_con = cons / sums;
end