function anchor_box = prepare_anchor_box( param )
    scale = (param.max_target / param.min_target)^(1/(param.anchor_scale*2));
    length = param.min_target * scale.^(1:2:2*param.anchor_scale);
    anchor_box = [-(length-1)'/2, -(length-1)'/2, (length-1)'/2, (length-1)'/2];
    if param.multishape_anchor
        slength = length / sqrt(2);
        llength = length * sqrt(2);
        anchor_box_fat = [-(llength-1)'/2, -(slength-1)'/2, (llength-1)'/2, (slength-1)'/2];
        anchor_box_thin = [-(slength-1)'/2, -(llength-1)'/2, (slength-1)'/2, (llength-1)'/2];
        anchor_box = cat(1, anchor_box, anchor_box_fat, anchor_box_thin);
    end
end

