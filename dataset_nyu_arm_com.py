
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import pdb
from tqdm import tqdm

SAMPLE_NUM = 1024
JOINT_NUM = 14

class HandPointDatasetArm(data.Dataset):
    def __init__(self, root_path, opt, sample=2048, output_num=1024, train=True, shuffle=False):
        self.root_path = root_path
        self.train = train

        self.SAMPLE_NUM = sample
        self.OUTPUT_NUM = output_num
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.JOINT_NUM = opt.JOINT_NUM
        self.restrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.record_file, self.record_data = self.__fileToNumpy(os.path.join(root_path, 'record.txt'))

        self.total_frame_num = len(self.record_file)
        print(self.total_frame_num)
        
        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, 36*3], dtype=np.float32)
        self.offset = np.empty(shape=[self.total_frame_num, 3], dtype=np.float32)
        self.rotate = np.empty(shape=[self.total_frame_num, 3, 3], dtype=np.float32)


        self.start_index = 0
        self.end_index = 0
        
        print("Loading Dataset..........")

        for i in tqdm(range(self.total_frame_num)):
            cur_data_dir = os.path.join(self.root_path, self.record_file[i] + '_Point_Cloud_FPS.mat')
            # print("Training: " + cur_data_dir)
            self.__loaddata(cur_data_dir)


        self.gt_xyz = self.record_data[:, 1:109].astype(np.float32)
        self.volume_length = self.record_data[:, 0].astype(np.float32)
        self.offset = self.record_data[:, 109:112].astype(np.float32)
        self.rotate = self.record_data[:, 112:].astype(np.float32).reshape(-1,3,3)

       
        if shuffle:
            idx_shuffle = np.random.permutation(len(self.point_clouds))
            self.point_clouds = self.point_clouds[idx_shuffle]
            self.volume_length = self.volume_length[idx_shuffle]
            self.gt_xyz = self.gt_xyz[idx_shuffle]
            self.offset = self.offset[idx_shuffle]
            self.rotate = self.rotate[idx_shuffle]

        self.point_clouds = torch.from_numpy(self.point_clouds)
        self.volume_length = torch.from_numpy(self.volume_length).view(self.total_frame_num, 1)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)
        self.offset = torch.from_numpy(self.offset)
        self.rotate = torch.from_numpy(self.rotate)

        self.total_frame_num = self.point_clouds.size(0)


    def __getitem__(self, index):
        # sample_idx = np.random.choice(self.SAMPLE_NUM, 1024)
        # return self.point_clouds[index, sample_idx, :], self.volume_length[index], self.gt_xyz.view(-1,36,3), self.offset[index],

        if self.OUTPUT_NUM != self.SAMPLE_NUM:
            pts = self.__random_cut_arm(self.point_clouds[index, :, :], self.rotate[index], self.gt_xyz[index, :].view(-1, 3), self.OUTPUT_NUM)
            # pts = self.__random_cut_arm(self.point_clouds[index, :, :], self.rotate[index], self.gt_xyz[index, :].view(-1, 3), self.OUTPUT_NUM)
            gt_xyz = self.gt_xyz[index, :].view(-1, 3) 
            offset = self.offset[index]
        else:
            pts = self.point_clouds[index, :, :]
            offset =  self.offset[index]
            gt_xyz = self.gt_xyz[index, :].view(-1, 3)
        return pts, self.volume_length[index], gt_xyz[self.restrictedJointsEval,:].view(14*3), offset

    def __len__(self):
        return self.point_clouds.size(0)

    def __loaddata(self, data_dir):
        point_cloud = sio.loadmat(data_dir)

        self.start_index = self.end_index + 1
        self.end_index = self.end_index + 1

        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud['Point_Cloud_FPS'].astype(np.float32)[:,:3]

    def __random_cut_arm(self, points, rot, output_num):

        points = torch.matmul(points, rot)

        max_ = torch.max(points, 0)[0]
        min_ = torch.min(points, 0)[0]

        box = max_ - min_

        if box[0] > box[1]:
            mask = points[:,0] > (max_[0] - 0.8 - np.random.rand() * (max_[0] - min_[0] - 0.8))
            sampled_points = points[mask]
            idx = np.random.choice(len(sampled_points), output_num)
            points = sampled_points[idx]
        elif box[1] > box[0]:
            mask1 = points[:,1] > (-0.4 - np.random.rand() * (-0.4 - min_[1]))
            mask2 = points[:,1] < (0.4 + np.random.rand() * (max_[1] - 0.4))
            mask = mask1 & mask2
            sampled_points = points[mask]
            idx = np.random.choice(len(sampled_points), output_num)
            points = sampled_points[idx]
        else:
            idx = np.random.choice(len(points), output_num)
            points = points[idx]

        points = torch.matmul(points, rot.T)

        return points

    def __random_cut_arm(self, points, rot, joints, output_num):

        points = torch.matmul(points, rot)
        joints = torch.matmul(joints.view(-1, 3), rot).view(-1)

        anchors = joints[[30, 31]]

        max_anchor = torch.min(anchors, 0)[0]

        max_ = torch.max(points, 0)[0]
        min_ = torch.min(points, 0)[0]

        box = max_ - min_

        if box[0] > box[1]:
            mask = points[:,0] > (max_anchor - np.random.rand() * (max_anchor - min_[0]))
            sampled_points = points[mask]
            if len(sampled_points) > output_num:
                idx = np.random.choice(len(sampled_points), output_num)
                points = sampled_points[idx]
            else:
                idx = np.random.choice(len(points), output_num)
                points = points[idx]
        elif box[1] > box[0]:
            mask1 = points[:,1] > (-0.3 - np.random.rand() * (-0.3 - min_[1]))
            mask2 = points[:,1] < (0.3 + np.random.rand() * (max_[1] - 0.3))
            mask = mask1 & mask2
            sampled_points = points[mask]
            idx = np.random.choice(len(sampled_points), output_num)
            points = sampled_points[idx]
        else:
            idx = np.random.choice(len(points), output_num)
            points = points[idx]

        points = torch.matmul(points, rot.T)

        return points

    def __get_frmae_num(self, data_dir):
        volume_length = sio.loadmat(os.path.join(data_dir, "Volume_length.mat"))
        return len(volume_length['Volume_length'])

    def __fileToNumpy(self, filename):
        file = open(filename)
        file_lines = file.readlines()
        numberOfLines = len(file_lines)
        dataArray = np.zeros((numberOfLines, 121))
        labels = []
        index = 0
        for line in file_lines:
            line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
            formLine = line.split(' ')
            dataArray[index,:] = formLine[1:122]
            labels.append((formLine[0]))
            index += 1
        return labels, dataArray