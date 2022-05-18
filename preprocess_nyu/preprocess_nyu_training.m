save_dir='../preprocess_nyu/testing/';

if ~ exist(save_dir,'dir')
    mkdir(save_dir);
end


flag = 'Test'; % 'Train' or 'Test'

fFocal_x = 588.03;	% mm
fFocal_y = 587.07;

cube = 250.

if strcmp(flag,'Train')
    dataset_dir='/workspace/Hand-Pointnet/preprocess_nyu/dataset/train/';
    centersID = fopen('/workspace/Hand-Pointnet/preprocess_nyu/center_train_refined.txt');
    labels = load([dataset_dir 'joint_data.mat']);
    mkdir([save_dir 'Training/']);
    recordID = fopen([save_dir 'Training/record.txt'], 'a');
elseif strcmp(flag,'Test')
    dataset_dir='/workspace/Hand-Pointnet/preprocess_nyu/dataset/test/';
    centersID = fopen('/workspace/Hand-Pointnet/preprocess_nyu/center_test_refined.txt');
    
    labels = load([dataset_dir 'joint_data.mat']);

    mkdir([save_dir 'Testing/']);
    recordID = fopen([save_dir 'Testing/record.txt'], 'a');
end

centers = textscan(centersID, '%s %s %s');

%labels.joint_uvd
size(labels.joint_uvd(1,:,:,:))
labels = squeeze(labels.joint_uvd(1,:,:,:));
size(labels)
f = dir([dataset_dir 'depth_1*.png']);
flist = {f.name};

JOINT_NUM = 36;
SAMPLE_NUM = 2048;
sample_num_level1 = 512;
sample_num_level2 =  128;                                                                                                                                                                                          
%for file_idx = 1:length(labels)
for file_idx = 1000:1100

    img_width = 640;
    img_height = 480;
 
    if (strcmp(centers{1}(file_idx), 'invalid') ||strcmp(centers{2}(file_idx), 'invalid')||strcmp(centers{3}(file_idx), 'invalid'))
        continue;
    end

    display(file_idx);
    display(flist{file_idx});
    
    if strcmp(flag,'Train')
        save_pos = [save_dir 'Training/' ]
        mkdir(save_pos);
    elseif strcmp(flag,'Test')
        save_pos = [save_dir 'Testing/']
        mkdir(save_pos);
    end
    
    depth_img = imread([dataset_dir flist{file_idx}]); 
    depth_img = double(depth_img);
    depth = depth_img(:,:,3)*1.0 + depth_img(:,:,2) * 256.0;

    file_names = strsplit(flist{file_idx}, {'/','.'});

    joints = squeeze(labels(file_idx,:,:));
    
    coms_x = str2double(centers{1}(file_idx));
    coms_y = str2double(centers{2}(file_idx));
    coms_z = str2double(centers{3}(file_idx));

    zstart = coms_z - cube / 2.0;
    zend = coms_z + cube / 2.0;

    xstart = floor((coms_x - cube / 2.0) / coms_z * fFocal_x + img_width/2.0);
    xend = ceil((coms_x + cube / 2.0) / coms_z * fFocal_x + img_width/2.0);
    ystart = floor(img_height/2.0 - (coms_y + cube / 2.0) / coms_z * fFocal_y);
    yend = ceil(img_height/2.0 - (coms_y - cube / 2.0) / coms_z * fFocal_y);

    
    left = max(xstart, 1);
    right = min(xend, 640);
    top =  max(ystart, 1);
    bottom = min(yend, 480);
    back = zend;
    front =  zstart;

    depth = depth(top:bottom, left:right);

    % 1. read ground truth

    %  gt_wld(3,:,:) = -gt_wld(3,:,:);
    % gt_wld=permute(gt_wld, [3 2 1]);

    % 2. get point cloud and surface normal

    Point_Cloud_FPS = zeros(SAMPLE_NUM,3);
    Volume_rotate = zeros(3,3);
    Volume_length = zeros(1);
    Volume_offset = zeros(3);
    Volume_GT_XYZ = zeros(JOINT_NUM,3);
    Volume_rotate = zeros(3,3);
%     valid = msra_valid{sub_idx, ges_idx};


    %% 2.1 read binary file

    bb_width = right - left;
    bb_height = bottom - top;

    valid_pixel_num = bb_width*bb_height;


    %% 2.2 convert depth to xyz
    hand_3d = zeros(valid_pixel_num,3);
    for ii=1:bb_height
        for jj=1:bb_width
            if  (depth(ii,jj) < front ||  depth(ii,jj) > back)
                continue;
            end
            idx = (jj-1)*bb_height+ii;
            hand_3d(idx, 1) = -(img_width/2.0 - (jj+left-1))*double(depth(ii,jj))/fFocal_x;
            hand_3d(idx, 2) = (img_height/2.0 - (ii+top-1))*double(depth(ii,jj))/fFocal_y;
            hand_3d(idx, 3) = depth(ii,jj);
        end
    end

    valid_idx = 1:valid_pixel_num;
    valid_idx = valid_idx(hand_3d(:,1)~=0 | hand_3d(:,2)~=0 | hand_3d(:,3)~=0);
    hand_points = hand_3d(valid_idx,:);
    
    if(length(hand_points) <= 0)
        continue;
    end
    
    [coeff,score,latent] = pca(hand_points);
    if coeff(2,1)<0
        coeff(:,1) = -coeff(:,1);
    end
    if coeff(3,3)<0
        coeff(:,3) = -coeff(:,3);
    end
    coeff(:,2)=cross(coeff(:,3),coeff(:,1));

    jnt_xyz = zeros(length(joints),3);
    for j=1:length(joints)
            jnt_xyz(j, 1) = -(img_width/2 - joints(j,1))*joints(j,3)/fFocal_x;
            jnt_xyz(j, 2) = (img_height/2 - joints(j,2))*joints(j,3)/fFocal_y;
            jnt_xyz(j, 3) = joints(j,3);
    end

    %% 2.4 sampling
    if size(hand_points,1)<SAMPLE_NUM
        tmp = floor(SAMPLE_NUM/size(hand_points,1));
        rand_ind = [];
        for tmp_i = 1:tmp
            rand_ind = [rand_ind 1:size(hand_points,1)];
        end
        rand_ind = [rand_ind randperm(size(hand_points,1), mod(SAMPLE_NUM, size(hand_points,1)))];
    else
        rand_ind = randperm(size(hand_points,1),SAMPLE_NUM);
    end
    hand_points_sampled = hand_points(rand_ind,:);

    %% 2.6 Normalize Point Cloud

    max_bb3d_len = 275.0;

    hand_points_sampled = hand_points_sampled/max_bb3d_len;
    %if size(hand_points,1)<SAMPLE_NUM
    %    offset = mean(hand_points)/max_bb3d_len;
    %else
    %    offset = mean(hand_points_sampled);
    %end
    offset = [coms_x, coms_y, coms_z]/max_bb3d_len;
    hand_points_sampled = hand_points_sampled - repmat(offset,SAMPLE_NUM,1);

    %% 2.8 ground truth
    jnt_xyz_normalized = jnt_xyz/max_bb3d_len;
    jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);

    Point_Cloud_FPS = hand_points_sampled;
    Volume_length = max_bb3d_len;
    Volume_offset = offset;
    Volume_GT_XYZ = jnt_xyz_normalized;
    Volume_rotate(:,:) = coeff;

    % 3. save files

    if strcmp(flag,'Train')
        save([save_pos file_names{1} '_Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        fprintf(recordID, '%s ', file_names{1} );
    elseif strcmp(flag,'Test')
        save([save_pos file_names{1} '_Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        fprintf(recordID, '%s ', file_names{1} );    
    end
    
    fprintf(recordID, '%.6f ', Volume_length);
    
    for ri=1:JOINT_NUM
        for rj=1:3
             fprintf(recordID, '%.6f ', Volume_GT_XYZ(ri,rj));
        end
    end
    
    for i = 1:3
        fprintf(recordID, '%.6f ', Volume_offset(i));
    end

    for ri=1:3
        for rj =1:3
             fprintf(recordID, '%.6f ', Volume_rotate(ri,rj));
        end
    end

    fprintf(recordID, '\r\n');
    
end
fclose(recordID);
