import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils
# import pytorch3d.ops as torch3d

LEAKY_RATE = 0.1
use_bn = False


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def joint2offset(joints, points, theta):

    device = joints.device
    B = joints.size(0)
    J = joints.size(-1)
    N = points.size(-1)
    joints_feature = joints.view(B,-1,1).repeat(1,1,N) #B, Jx3, N
    points_repeat = points.repeat(1, J, 1) #B, Jx3, N
    offset = joints_feature - points_repeat
    offset = offset.view(B,J,3,N)
    dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8) #B, J, N
    offset_norm = (offset / (dist.unsqueeze(2))) #B, J, 3, N
    heatmap = theta - dist
    mask = heatmap.ge(0).float() #B, J, N
    offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(B,-1,N)
    heatmap_mask = heatmap * mask.float()

    return torch.cat((offset_norm_mask,heatmap_mask),dim=1)

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class BiasConv1d(nn.Module):
    def __init__(self, bias_length, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(BiasConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.randn((out_channels, bias_length)),requires_grad=True)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)+self.bias.unsqueeze(0).repeat(x.size(0),1,1)))
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class GCN1d(nn.Module):
    def __init__(self, in_channels, out_channels, graph_width, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(GCN1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.graph_a = nn.Parameter(torch.randn(1, in_channels, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

    def forward(self, x):
        point1_graph = self.graph_w(torch.matmul(x.unsqueeze(-2),self.graph_a).squeeze(-2))
        return x

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2=None, group_all=False, use_fps=True,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False, bn=True, knn=False):
        super(PointConv, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        # self.act = act
        self.act = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.use_fps = use_fps
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.knn = knn
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if bn:
                if use_instance_norm:
                    self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
                else:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        self.weightnet = WeightNet(3, 8)
        self.linear = nn.Sequential(nn.Conv1d(8 * last_channel, last_channel, 1, bias=False),
                                    nn.BatchNorm1d(out_channel) if bn else nn.Identity())

        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    if use_instance_norm:
                        self.mlp2_bns.append(nn.InstanceNorm1d(out_channel, affine=True))
                    else:
                        self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel
        if knn is False:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, new_xyz = None, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.contiguous()

        if new_xyz == None:
            if (self.group_all == False) and (self.npoint != -1) and self.use_fps:
                if fps_idx == None:
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
                new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)  # [B, C, N]
            elif self.use_fps is False:
                # fps_idx = torch.arange(self.npoint,dtype=torch.int).view(1, -1).repeat(xyz.size(0),1)
                # new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)
                new_xyz = xyz[...,:self.npoint]
            else:
                new_xyz = xyz

        if self.knn:
            sqrdists = square_distance(new_xyz.transpose(2, 1).contiguous(), xyz_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz_t, knn_idx)
            direction_xyz = neighbor_xyz - new_xyz.transpose(1,2).view(B, self.npoint, 1, C)

            grouped_points = index_points_group(points.transpose(1,2), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points], dim = -1)
            new_points = new_points.permute(0, 3, 1, 2)
        else:
            new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        grouped_xyz = new_points[:,:3,...]
        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                if self.bn:
                    bn = self.mlp_bns[i]
                    new_points = self.act(bn(conv(new_points)))
                else:
                    new_points = self.act(conv(new_points))
            else:
                new_points = conv(new_points)

        weights = self.weightnet(grouped_xyz)# B, K, N, S
        # B, N, C, S * B, N, S, K
        new_points = torch.matmul(input=new_points.permute(0, 2, 1, 3), other = weights.permute(0, 2, 3, 1))
        new_points = new_points.view(B, self.npoint, -1).permute(0, 2, 1)
        new_points = self.act(self.linear(new_points))

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.use_act:
                    if self.bn:
                        bn = self.mlp2_bns[i]
                        new_points = self.act(bn(conv(new_points)))
                    else:
                        new_points = self.act(conv(new_points))
                else:
                    new_points = conv(new_points)

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points

class PointConvCentroid(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(PointConvCentroid,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 8)
        self.linear = nn.Sequential(nn.Conv1d(8 * last_channel, last_channel, 1, bias=False),
                                    nn.BatchNorm1d(out_channel) if bn else nn.Identity())
        
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        # _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        weights = self.weightnet(direction_xyz.permute(0, 3, 2, 1))# B, K, S, N
        # B, N, C, S * B, N, S, K
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1))
        new_points = new_points.view(B, N1, -1).permute(0, 2, 1)
        new_points = self.relu(self.linear(new_points))

        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))
        if self.return_inter:
            return new_points, inter
        return new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2=None, group_all=False, use_fps=True,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False, bn=True, knn=False, bias=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        # self.act = act
        self.act = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.use_fps = use_fps
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.knn = knn
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                if use_instance_norm:
                    self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
                else:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    if use_instance_norm:
                        self.mlp2_bns.append(nn.InstanceNorm1d(out_channel, affine=True))
                    else:
                        self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel
        if knn is False:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, new_xyz = None, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.contiguous()

        if new_xyz == None:
            if (self.group_all == False) and (self.npoint != -1) and self.use_fps:
                if fps_idx == None:
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
                new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)  # [B, C, N]
            elif self.use_fps is False:
                # fps_idx = torch.arange(self.npoint,dtype=torch.int).view(1, -1).repeat(xyz.size(0),1)
                # new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)
                new_xyz = xyz[...,:self.npoint]
            else:
                new_xyz = xyz

        if self.knn:
            sqrdists = square_distance(new_xyz.transpose(2, 1).contiguous(), xyz_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz_t, knn_idx)
            direction_xyz = neighbor_xyz - new_xyz.transpose(1,2).view(B, self.npoint, 1, C)

            grouped_points = index_points_group(points.transpose(1,2), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points], dim = -1)
            new_points = new_points.permute(0, 3, 1, 2)
        else:
            new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                if self.bn:
                    bn = self.mlp_bns[i]
                    new_points = self.act(bn(conv(new_points)))
                else:
                    new_points = self.act(conv(new_points))
            else:
                new_points = conv(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.use_act:
                    if self.bn:
                        bn = self.mlp2_bns[i]
                        new_points = self.act(bn(conv(new_points)))
                    else:
                        new_points = self.act(conv(new_points))
                else:
                    new_points = conv(new_points)

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points

class PointNetSetAbstractionSeparate(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2=None, group_all=False, use_fps=True,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False, bn=True, knn=False, bias=False):
        super(PointNetSetAbstractionSeparate, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        # self.act = act
        self.act = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.use_fps = use_fps
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.knn = knn
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                if use_instance_norm:
                    self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
                else:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    if use_instance_norm:
                        self.mlp2_bns.append(nn.InstanceNorm1d(out_channel, affine=True))
                    else:
                        self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel
        if knn is False:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, new_xyz = None, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.contiguous()

        # print(points.shape)
        # print(self.mlp_convs[0].weight.shape)
        points = F.conv1d(points, self.mlp_convs[0].weight[:,3:,...].squeeze(-1))

        if new_xyz == None:
            if (self.group_all == False) and (self.npoint != -1) and self.use_fps:
                if fps_idx == None:
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
                new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)  # [B, C, N]
            elif self.use_fps is False:
                # fps_idx = torch.arange(self.npoint,dtype=torch.int).view(1, -1).repeat(xyz.size(0),1)
                # new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)
                new_xyz = xyz[...,:self.npoint]
            else:
                new_xyz = xyz

        if self.knn:
            sqrdists = square_distance(new_xyz.transpose(2, 1).contiguous(), xyz_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz_t, knn_idx)
            direction_xyz = neighbor_xyz - new_xyz.transpose(1,2).view(B, self.npoint, 1, C)

            grouped_points = index_points_group(points.transpose(1,2), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points], dim = -1)
            new_points = new_points.permute(0, 3, 1, 2)
        else:
            new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = F.conv2d(new_points[:,:3,...], self.mlp_convs[0].weight[:,:3,...])+ new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.use_act:
                if self.bn:
                    bn = self.mlp_bns[i]
                    new_points = self.act(bn(new_points))
                else:
                    new_points = self.act(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.use_act:
                    if self.bn:
                        bn = self.mlp2_bns[i]
                        new_points = self.act(bn(conv(new_points)))
                    else:
                        new_points = self.act(conv(new_points))
                else:
                    new_points = conv(new_points)

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class PointNetSetAbstractionLightFirst(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2=None, group_all=False, use_fps=True,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False, bn=True, knn=False, bias=False):
        super(PointNetSetAbstractionLightFirst, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        # self.act = act
        self.act = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.use_fps = use_fps
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.knn = knn
        last_channel = 3
        self.mlp_convs_first = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                if use_instance_norm:
                    self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
                else:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    if use_instance_norm:
                        self.mlp2_bns.append(nn.InstanceNorm1d(out_channel, affine=True))
                    else:
                        self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel
        if knn is False:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, new_xyz = None, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.contiguous()

        points = self.mlp_convs_first(points)

        if new_xyz == None:
            if (self.group_all == False) and (self.npoint != -1) and self.use_fps:
                if fps_idx == None:
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
                new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)  # [B, C, N]
            elif self.use_fps is False:
                # fps_idx = torch.arange(self.npoint,dtype=torch.int).view(1, -1).repeat(xyz.size(0),1)
                # new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)
                new_xyz = xyz[...,:self.npoint]
            else:
                new_xyz = xyz

        if self.knn:
            sqrdists = square_distance(new_xyz.transpose(2, 1).contiguous(), xyz_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz_t, knn_idx)
            direction_xyz = neighbor_xyz - new_xyz.transpose(1,2).view(B, self.npoint, 1, C)

            grouped_points = index_points_group(points.transpose(1,2), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points], dim = -1)
            new_points = new_points.permute(0, 3, 1, 2)
        else:
            new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = conv(new_points[:,:3,...]) + new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.use_act:
                if self.bn:
                    bn = self.mlp_bns[i]
                    new_points = self.act(bn(new_points))
                else:
                    new_points = self.act(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.use_act:
                    if self.bn:
                        bn = self.mlp2_bns[i]
                        new_points = self.act(bn(conv(new_points)))
                    else:
                        new_points = self.act(conv(new_points))
                else:
                    new_points = conv(new_points)

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points

class PointNetSetAbstractionCentroid(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(PointNetSetAbstractionCentroid,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        # _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))
        if self.return_inter:
            return new_points, inter
        return new_points

class Mapping(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(Mapping,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  + latent_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))
        if self.return_inter:
            return new_points, inter
        return new_points

class FuseMappingLight(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(FuseMappingLight,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        self.fuse = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        # grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            new_points =  conv(new_points)
            if i == 0:
                grouped_points1 = self.fuse(points1)
                new_points = new_points + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(new_points))
            else:
                new_points =  self.relu(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))
        if self.return_inter:
            return new_points, inter
        return new_points

class GraphMapping(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(GraphMapping,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
            self.fuse_bns = nn.ModuleList()
        last_channel = in_channel  + latent_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.fuse_convs.append(nn.Conv2d(out_channel*2, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                self.fuse_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel * 2
        last_channel = out_channel
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        grouped_xyz = xyz1.view(B, N1, 1, 3).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3+3, nsample, N1]

        #
        new_embeddings = None
        for i, conv in enumerate(self.mlp_convs):
            if new_embeddings is not None:
                new_points = torch.cat((new_points, new_embeddings.unsqueeze(-2).repeat(1,1,self.nsample,1)), 1)

            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
            embeddings = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2) # [B, D, N1]

            #dynamic graph
            embeddings_fuse = torch.cat((embeddings.unsqueeze(-1).repeat(1,1,1,N1),
                                        embeddings.unsqueeze(-2).repeat(1,1,N1,1)), 1)

            new_embeddings = self.fuse_convs[i](embeddings_fuse)
            if self.bn:
                bn = self.fuse_bns[i]
                new_embeddings = self.relu(bn(new_embeddings))
            else:
                new_embeddings =  self.relu(new_embeddings)
            new_embeddings = torch.mean(new_embeddings, -1)

        new_points = new_embeddings
        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
        if self.return_inter:
            return new_points, inter
        return new_points

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.bias = nn.Parameter(torch.randn((21, 21)),requires_grad=True)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) + self.bias.unsqueeze(0).repeat(energy.size(0),1,1)
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
 
class TransMapping(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False, stack=False):
        super(TransMapping,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.stack = stack
        self.mlp_convs = nn.ModuleList()

        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
            self.fuse_bns = nn.ModuleList()
        last_channel = in_channel  + latent_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        last_channel = out_channel
        if mlp2:
            self.pos = nn.Conv1d(3, last_channel, 1)
            self.trans = nn.ModuleList() 
            for out_channel in mlp2:
                # self.trans.append(SelfAttention(4, last_channel, last_channel, out_channel//8 ,out_channel//4))
                self.trans.append(SelfAttention(last_channel))
                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        inter = new_points

        stack = []
        if self.mlp2:
            new_points = new_points + self.pos(xyz1.permute(0, 2, 1))
            for _,trans in enumerate(self.trans):
                new_points =  trans(new_points)
                stack.append(new_points)
        if self.stack:
            new_points = torch.cat(stack, 1)

        if self.return_inter:
            return new_points, inter
        return new_points

class GRUMapping(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(GRUMapping,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn

        last_channel = in_channel + 3
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        self.graph_i = nn.Linear(graph_width, graph_width, bias=False)
        self.graph_r = nn.Linear(graph_width, graph_width, bias=False)
        self.graph_h = nn.Linear(graph_width, graph_width, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

        point1_graph_i = self.graph_i(points1)

        # z
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph_i)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph_i)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_r = self.fuse_r_o(self.graph_r(points1))
        point1_graph_r_expand = point1_graph_r.view(B, point1_graph_r.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_r_expand = r * point1_graph_r_expand

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_r_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        point1_graph_h = self.graph_h(points1)
        
        new_points = (1 - z) * point1_graph_h + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingChl(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(GRUMappingChl,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn

        last_channel = in_channel + 3
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        self.graph_i = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_r = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_h = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]

        point1_graph_i = torch.matmul(points1.unsqueeze(-2),self.graph_i).squeeze(-2)

        # z
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph_i)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph_i)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_r = self.fuse_r_o(torch.matmul(points1.unsqueeze(-2),self.graph_r).squeeze(-2))
        point1_graph_r_expand = point1_graph_r.view(B, point1_graph_r.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_r_expand = r * point1_graph_r_expand

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_r_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        point1_graph_h = torch.matmul(points1.unsqueeze(-2),self.graph_h).squeeze(-2)
        
        new_points = (1 - z) * point1_graph_h + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCN(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingGCN,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        errrrrrrrrrrr, da wu yu, fuse_r_o lost
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCN0(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False, bias=True):
        super(GRUMappingGCN0,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1, bias=True),#bias=False 2022.03.31 01:55
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        if self.bypass_gcn:
            new_points = (1 - z) * points1 + z * h
        else:
            new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingNoGCN(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingNoGCN,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = points1

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        new_points = (1 - z) * points1 + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points


class GRUMappingGCN1(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingGCN1,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.graph_a2 = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w2 = nn.Sequential(nn.Conv1d(latent_channel, latent_channel*2, 1),
                                     nn.BatchNorm1d(latent_channel*2) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.graph_a3 = nn.Parameter(torch.randn(1, latent_channel*2, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w3 = nn.Sequential(nn.Conv1d(latent_channel*2, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))
        point1_graph = self.graph_w2(torch.matmul(point1_graph.unsqueeze(-2),self.graph_a2).squeeze(-2))
        point1_graph = self.graph_w3(torch.matmul(point1_graph.unsqueeze(-2),self.graph_a3).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCN2(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False):
        super(GRUMappingGCN2,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.graph_ah = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_wh = nn.Sequential(nn.Conv1d(latent_channel, latent_channel*2, 1),
                                     nn.BatchNorm1d(latent_channel*2) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        h_graph = self.graph_w(torch.matmul(h.unsqueeze(-2),self.graph_ah).squeeze(-2))

        
        if self.bypass_gcn:
            new_points = (1 - z) * points1 + z * h_graph
        else:
            new_points = (1 - z) * point1_graph + z * h_graph

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCN3(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingGCN3,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        new_points = h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points


class GRUMappingGCN0Bias(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingGCN0Bias,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp_r_bias = []
        self.mlp_z_bias = []
        self.mlp_h_bias = []
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_r_bias.append(nn.Parameter(torch.randn((1, out_channel, 1, graph_width)).cuda(),requires_grad=True))
            self.mlp_z_bias.append(nn.Parameter(torch.randn((1, out_channel, 1, graph_width)).cuda(),requires_grad=True))
            self.mlp_h_bias.append(nn.Parameter(torch.randn((1, out_channel, 1, graph_width)).cuda(),requires_grad=True))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            r = r + self.mlp_r_bias[i]
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            z = z + self.mlp_z_bias[i]
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            h = h + self.mlp_h_bias[i]
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)
        
        new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points


class GRUMappingGCNAttenPool(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None):
        super(GRUMappingGCNAttenPool,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn

        last_channel = in_channel

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        self.mlp_r_pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp_z_pos = nn.Sequential(nn.Conv2d(3, mlp[-2], 1),
                                     nn.BatchNorm2d(mlp[-2]) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.mlp_h_pos = nn.Sequential(nn.Conv2d(3, mlp[-2], 1),
                                     nn.BatchNorm2d(mlp[-2]) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.mlp_z_atten = nn.Conv2d(mlp[-2]*2, mlp[-2], 1)
        self.mlp_h_atten = nn.Conv2d(mlp[-2]*2, mlp[-2], 1)

        self.softmax = nn.Softmax(-2)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points2], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points[:,3:,...]
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                r_pos = self.mlp_r_pos(new_points[:,:3,...])
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + r_pos + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)

        # z
        z = new_points[:,3:,...]
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            elif i != len(self.mlp_z_convs) - 2:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z_pos = self.mlp_z_pos(new_points[:,:3,...])
                z_softmax = self.softmax(self.mlp_z_atten(torch.cat([z_pos, z], 1)))
                z = torch.sum(z * z_softmax, -2, keepdim=True)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand

        h = new_points[:,3:,...]
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h_pos = self.mlp_h_pos(new_points[:,:3,...])
                h_softmax = self.softmax(self.mlp_h_atten(torch.cat([h_pos, h], 1)))
                h = torch.sum(h * h_softmax, -2, keepdim=True)

        h = h.squeeze(-2)
        
        new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCNStep(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False):
        super(GRUMappingGCNStep,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)

            # print('relu', r.shape)
        # print('r', r.shape)
        # r = torch.max(r, -2)[0]
        # print('rmax', r.shape)
        # r = self.sigmoid(r)

        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            z = self.relu(z)
            # print('relu', z.shape)
        # print('z', z.shape)
        z = torch.max(z, -2)[0]
        # print('zmax', z.shape)
        z = self.sigmoid(z)

        # point1_graph_expand = self.fuse_r_o(r * point1_graph).view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_graph_expand = r * point1_graph_expand
        point1_graph_expand = self.fuse_r_o(point1_graph_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_graph_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            h = self.relu(h)

        h = torch.max(h, -2)[0]
        h = self.tanh(h)

        new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingFuseGCN0(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False, bias=True):
        super(GRUMappingFuseGCN0,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn

        last_channel = mlp[0] + latent_channel

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1, bias=True),#bias=False 2022.03.31 01:55
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channel + 3, mlp[0], 1, bias=bias),
            nn.BatchNorm2d(mlp[0]) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True),
            nn.Conv2d(mlp[0], mlp[0], 1, bias=bias),
            nn.BatchNorm2d(mlp[0]) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=bias))
            self.mlp_z_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=bias))
            self.mlp_h_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=bias))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm1d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm1d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        new_points =  torch.max(self.fuse(new_points), -2)[0]

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = torch.cat([new_points, point1_graph], 1)
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
            else:
                r = self.relu(r)

        # z
        z = torch.cat([new_points, point1_graph], 1)
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

        h = torch.cat([new_points, r * point1_graph], 1)
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)

        h = h.squeeze(-2)

        if self.bypass_gcn:
            new_points = (1 - z) * points1 + z * h
        else:
            new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points

class GRUMappingGCN0Separate(nn.Module):
    # first conv decomposed
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False, bias=True, graph_bias=True):
        super(GRUMappingGCN0Separate,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1, bias=graph_bias),#bias=False 2022.03.31 01:55
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        # self.graph_w[0].bias = torch.nn.Parameter(torch.zeros_like(self.graph_w[0].bias))
        # torch.nn.init.zeros_(self.graph_w[0].bias.data)

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)
        points_r = F.conv1d(points2, self.mlp_r_convs[0].weight[:,:-3,...].squeeze(-1))
        points_h = F.conv1d(points2, self.mlp_h_convs[0].weight[:,:-3,...].squeeze(-1))
        points_z = F.conv1d(points2, self.mlp_z_convs[0].weight[:,:-3,...].squeeze(-1))

        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            points_r = index_points_group(points_r.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            points_h = index_points_group(points_h.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            points_z = index_points_group(points_z.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            direction_xyz = direction_xyz.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_r = points_r.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_h = points_h.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_z = points_z.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = direction_xyz
        for i, conv in enumerate(self.mlp_r_convs):
           
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = F.conv2d(r, self.mlp_r_convs[0].weight[:,-3:,...], self.mlp_r_convs[0].bias ) + points_r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            else:
                r = conv(r)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = direction_xyz
        for i, conv in enumerate(self.mlp_z_convs):
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = F.conv2d(z, self.mlp_z_convs[0].weight[:,-3:,...], self.mlp_z_convs[0].bias) + points_z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            else:
                z = conv(z)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = direction_xyz
        for i, conv in enumerate(self.mlp_h_convs):
            
            if i == 0:
                h = F.conv2d(h, self.mlp_h_convs[0].weight[:,-3:,...], self.mlp_h_convs[0].bias)  + point1_expand + points_h
            else:
                h = conv(h)
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        if self.bypass_gcn:
            new_points = (1 - z) * points1 + z * h
        else:
            new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points


class GRUMappingGCNLightFirst(nn.Module):
    # first conv decomposed
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False, bias=True):
        super(GRUMappingGCNLightFirst,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn


        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1, bias=True),#bias=False 2022.03.31 01:55
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        self.r_first = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.z_first = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
        self.h_first = nn.Conv1d(in_channel, mlp[0], 1, bias=False)

        last_channel = last_channel_r = 3
        for i, out_channel in enumerate(mlp):
            # self.mlp_r_convs.append(nn.Conv2d(last_channel_r, out_channel if i != 1 else out_channel//2, 1, bias=bias))
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            if bn:
                # self.mlp_r_bns.append(nn.BatchNorm2d(out_channel if i != 1 else out_channel//2))
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            # last_channel_r = out_channel if i != 1 else out_channel//2
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)
        points_r = self.r_first(points2)
        points_h = self.h_first(points2)
        points_z = self.z_first(points2)
  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            points_r = index_points_group(points_r.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            points_h = index_points_group(points_h.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            points_z = index_points_group(points_z.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            direction_xyz = direction_xyz.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_r = points_r.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_h = points_h.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
            points_z = points_z.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # r
        r = direction_xyz
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + points_r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
                # print('sigmoid', r.shape)
            else:
                r = self.relu(r)
                # print('relu', r.shape)


        # z
        z = direction_xyz
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + points_z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)
        
        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = direction_xyz
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand + points_h
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                # 
                if self.relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        if self.bypass_gcn:
            new_points = (1 - z) * points1 + z * h
        else:
            new_points = (1 - z) * point1_graph + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return new_points


class UpsampleFeat(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_feat):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_feat = sparse_feat.permute(0, 2, 1) # B S 3

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        knn_idx = knn_point(8, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_feat = index_points_group(sparse_feat, knn_idx)
        dense_feat = torch.sum(weight.view(B, N, 8, 1) * grouped_feat, dim = 2).permute(0, 2, 1)
        return dense_feat 

class PointNetSetUpConv(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn = True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) is not 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if not knn:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)

        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape
        if self.knn:
            sqrdists = square_distance(pos1_t, pos2_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(pos2_t, knn_idx)
            direction_xyz = neighbor_xyz - pos1_t.view(B, N, 1, C)

            grouped_points = index_points_group(feature2.transpose(1,2), knn_idx) # B, N1, nsample, D2
            feat_new = torch.cat([direction_xyz, grouped_points], dim = -1)
            feat_new = feat_new.permute(0, 3, 1, 2)
        else:
            feat_new = self.queryandgroup(pos2_t, pos1_t, feature2)
            # (self.radius, self.nsample, pos2_t, pos1_t)
        
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]   # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)
        
        return feat_new

