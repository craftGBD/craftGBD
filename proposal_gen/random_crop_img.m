function [img_new, box_new] = random_crop_img(img_t, box_t, param)

tic
%dump_img(img_t, box_t, pts_t, valid_t, 'ori.jpg');

%% augmentation
[img_t box_t] = augmentation(img_t, box_t, 0, 0.5);
%dump_img(img_t, box_t, pts_t, valid_t, 'aug.jpg');


if isempty(box_t)
    if max(size(img_t)) > param.max_img
        startx = max(round(rand*size(img_t,2)), 1);
        starty = max(round(rand*size(img_t,1)), 1);
        endx = round(min(size(img_t,2), startx + param.max_img * 0.8));
        endy = round(min(size(img_t,1), starty + param.max_img * 0.8));
        img_t = img_t(starty:endy, startx:endx,:);
    end
	img_new = img_t;
	box_new = box_t;
	return;
end

%% select target
n_box = size(box_t,1);
the_one = randi(n_box);

%% compute ratio
gap = min(param.max_target, param.max_img)/param.min_target;
now_scale = max(box_t(the_one,3)-box_t(the_one,1)+1, ...
				box_t(the_one,4)-box_t(the_one,2)+1);
tar_scale = max(min(max(2*rand(),0.5) * now_scale, param.max_target), param.min_target);
ratio = tar_scale/now_scale;

assert(tar_scale <= param.max_img);

img_new = imresize(img_t, ratio);
box_new = box_t * ratio;

box_new = clip_box(box_new, size(img_new,2), size(img_new,1));
the_box = box_new(the_one,:);

%% mask out bad objects
box_h = box_new(:,4) - box_new(:,2);
box_w = box_new(:,3) - box_new(:,1);
keep = true(n_box,1);
longer = max(box_h, box_w);
keep = (longer <= param.max_target) & (longer >= param.min_target);
keep(the_one) = true; %% needed for precision error 
for i = 1:n_box
	if keep(i)
		keep_box{i} = img_new(box_new(i,2):box_new(i,4),box_new(i,1):box_new(i,3),:);
	end
end
for i = 1:n_box
	if ~keep(i)
		img_new(box_new(i,2):box_new(i,4), box_new(i,1):box_new(i,3), :) = 0;
	end
end
for i = 1:n_box
	if keep(i)
		img_new(box_new(i,2):box_new(i,4), box_new(i,1):box_new(i,3), :) = keep_box{i};
	end
end

%% see if the img_new need crop
img_h = size(img_new, 1);
img_w = size(img_new, 2);

if max(img_h, img_w) < param.min_img
	pad_h = floor((param.min_img - img_h)/2);
	pad_w = floor((param.min_img - img_w)/2);
	tmp_img = zeros(param.min_img, param.min_img, size(img_new,3), 'uint8');
	tmp_img(pad_h+1:pad_h+img_h,pad_w+1:pad_w+img_w,:) = img_new;
	img_new = tmp_img;


	box_new = box_new(keep,:);


	box_new = align_box(box_new, pad_w, pad_h);

	%dump_img(img_new, box_new, pts_new, valid_new, 'crop.jpg');

	%display('small');

	assert(all(box_new(:,1)>=1));
	assert(all(box_new(:,2)>=1));
	assert(all(box_new(:,3)<=size(img_new,2)));
	assert(all(box_new(:,4)<=size(img_new,1)));

	return;
end

if img_h > param.max_img
	up_beg = max(1, the_box(4)-param.max_img+1);
	up_end = min(the_box(2), img_h-param.max_img+1);
	assert(up_end>=up_beg);
	up = rand*(up_end-up_beg)+up_beg;
	down = up + param.max_img - 1;
	inds = find(box_new(:,2)>=down | box_new(:,4)<=up);
	keep(inds) = false;
	box_new(:,2) = max(box_new(:,2), up);
	box_new(:,4) = min(box_new(:,4), down);
	%display('large h');
else
	up = 1;
	down = img_h;
end
if img_w > param.max_img
	left_beg = max(1, the_box(3)-param.max_img+1);
	left_end = min(the_box(1), img_w-param.max_img+1);
	assert(left_end>=left_beg);
	left = rand*(left_end-left_beg)+left_beg;
	right = left + param.max_img - 1;
	inds = find(box_new(:,1)>=right | box_new(:,3)<=left);
	keep(inds) = false;
	box_new(:,1) = max(box_new(:,1), left);
	box_new(:,3) = min(box_new(:,3), right);
	%display('large w');
else
	left = 1;
	right = img_w;
end

frame = clip_box([left up right down], img_w, img_h);
left  = frame(1);
up    = frame(2);
right = frame(3);
down  = frame(4);

img_new = img_new(up:down, left:right, :);

box_new = box_new(keep,:);
box_new = align_box(box_new, 1-left, 1-up);

%dump_img(img_new, box_new, pts_new, valid_new, 'crop.jpg');

function box = clip_box(box, w, h)
box(:,1) = ceil(box(:,1));
box(:,2) = ceil(box(:,2));
box(:,3) = floor(box(:,3));
box(:,4) = floor(box(:,4));
box(:,1) = min(max(box(:,1), 1), w);
box(:,2) = min(max(box(:,2), 1), h);
box(:,3) = min(max(box(:,3), 1), w);
box(:,4) = min(max(box(:,4), 1), h);


function box = align_box(box, off_x, off_y)
if isempty(box)
	return;
end
box(:,1) = box(:,1) + off_x;
box(:,2) = box(:,2) + off_y;
box(:,3) = box(:,3) + off_x;
box(:,4) = box(:,4) + off_y;



function [img box] = augmentation(img, box, rotate_ratio, flip_ratio)
% rotate or/and flip
h = size(img, 1);
w = size(img, 2);
ctr_x = w/2;
ctr_y = h/2;
theta = rotate_ratio*(rand*2-1);
%color 
if rand() < 0.5
    img = mean(img, 3);
    img = img(:,:, [1 1 1]);
end


%%%% flip
if rand < flip_ratio
	img = flipdim(img, 2);
	if size(box,1) > 0
		box(:,1) = w - box(:,1) + 1;
		box(:,3) = w - box(:,3) + 1;
        box = box(:, [3 2 1 4]);
    end
end
if rotate_ratio > 0
    img = imrotate(img,theta,'bilinear');
    h_aft = size(img, 1);
    w_aft = size(img, 2);
    ctr_x_aft = w_aft/2;
    ctr_y_aft = h_aft/2;
    %%%% rotate box
    for i = 1:size(box,1)
        x1 = box(i,1) - ctr_x;
        y1 = box(i,2) - ctr_y;
        x2 = box(i,3) - ctr_x;
        y2 = box(i,4) - ctr_y;
        pp = [x1, y1; x1, y2; x2, y1; x2, y2];
        pp = pp * [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
        pp(:,1) = pp(:,1) + ctr_x_aft; 
        pp(:,2) = pp(:,2) + ctr_y_aft; 
        l = min(pp(:,1));
        r = max(pp(:,1));
        t = min(pp(:,2));
        b = max(pp(:,2));
        box(i,:) = int16([l,t,r,b]);
        box(i,1) = min(max(box(i,1),1),w_aft);
        box(i,2) = min(max(box(i,2),1),h_aft);
        box(i,3) = min(max(box(i,3),1),w_aft);
        box(i,4) = min(max(box(i,4),1),h_aft);
    end
end

function dump_img(img, box, name)
if size(box, 1) > 0
	box(:,3) = box(:,3) - box(:,1);
	box(:,4) = box(:,4) - box(:,2);
	img = insertShape(img, 'Rectangle', box);
end
imwrite(img, name);

