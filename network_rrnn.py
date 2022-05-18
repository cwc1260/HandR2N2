import torch
import torch.nn as nn
import math
import numpy as np
from pointutil import Conv1d,  Conv2d, BiasConv1d, PointNetSetAbstraction, GRUMappingGCN0
import torch.nn.functional as F


class HandModel(nn.Module):
    def __init__(self, joints=21, iters=3, graph_bias=True):
        super(HandModel, self).__init__()
        
        self.encoder_1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=64, in_channel=3, mlp=[32,32,128])
        
        self.encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128, mlp=[64,64,256])

        self.encoder_3 = nn.Sequential(Conv1d(in_channels=256+3, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                       nn.MaxPool1d(128,stride=1))

        self.fold1 = nn.Sequential(BiasConv1d(bias_length=joints, in_channels=512, out_channels=256, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=256, out_channels=256, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=256, out_channels=256, bn=True))
        self.regress_1 = nn.Conv1d(in_channels=256, out_channels=3, kernel_size=1)

        self.gru = GRUMappingGCN0(nsample=48, in_channel=128, latent_channel=256, graph_width=joints, mlp=[256, 256, 256], mlp2=None) #original
        self.regress = nn.Conv1d(in_channels=256, out_channels=3, kernel_size=1)

        self.iters = iters
        self.joints = joints

    def forward(self, pc, feat):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        pc1, feat1 = self.encoder_1(pc, feat)# B, 3, 512; B, 64, 512
        
        pc2, feat2 = self.encoder_2(pc1, feat1)# B, 3, 256; B, 128, 256
        
        code = self.encoder_3(torch.cat((pc2, feat2),1))# B, 3, 128; B, 256, 128
        
        code = code.expand(code.size(0),code.size(1), self.joints)

        joints = []

        embed = self.fold1(code)
        joint = self.regress_1(embed)
        joints.append(joint)

        for _ in range(self.iters):
            embed_ = self.gru(joint, pc1, embed, feat1)
            joint = self.regress(embed_ - embed) + joint
            embed = embed_
            joints.append(joint) # B, 3, 21
        
        return  [j.transpose(1,2).contiguous().view(-1, self.joints*3) for j in joints]


from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    import argparse
    # from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default = 3, help='start epoch')
    parser.add_argument('--joints', type=int, default = 21, help='start epoch')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,3,1024)).float().cuda()
    model = HandModel(opt.joints, opt.iters).cuda()
    print(model)