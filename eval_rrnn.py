'''
evaluation
'''
import argparse
import os
import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from thop import profile, clever_format
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
parser.add_argument('--iters', type=int, default = 3, help='start epoch')
parser.add_argument('--test_iters', type=int, default = 5, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='./pretrained_model',  help='output folder')
parser.add_argument('--test_index', type=int, default = 0,  help='test index for cross validation, range: 0~8')
parser.add_argument('--model', type=str, default = 'best_model.pth',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '/workspace/HandFoldDynGraph/data/NYU/process_nyu_center_com/Testing',  help='model name for training resume')

parser.add_argument('--dataset', type=str, default = 'nyu', help='optimizer name for training resume')
parser.add_argument('--model_name', type=str, default = 'rrnn',  help='')
parser.add_argument('--gpu', type=str, default = '1',  help='gpu')


opt = parser.parse_args()
print (opt)

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset == 'msra':
	from dataset_msra import subject_names
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+str(opt.iters)+'iters', subject_names[opt.test_index])
	opt.JOINT_NUM = 21
else:
	if opt.dataset == 'icvl':
		from dataset_icvl import HandPointDataset
		save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+str(opt.iters)+'iters')
		opt.JOINT_NUM = 16
		opt.test_iters = 3
	else:
		from dataset_nyu import HandPointDataset
		save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+str(opt.iters)+'iters')
		opt.JOINT_NUM = 14


# 1. Load data                                         
test_data = HandPointDataset(root_path=opt.test_path, opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers), pin_memory=False)
                                          
print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, iters=opt.test_iters)

if opt.ngpu > 1:
    model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
    model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
    model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':
    model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    
model.cuda()
print(model)

criterion = nn.MSELoss(size_average=True).cuda()

# 3. evaluation
torch.cuda.synchronize()

model.eval()
test_mse = 0.0
test_wld_err = np.zeros(opt.test_iters + 1)
test_class_err = torch.zeros(opt.JOINT_NUM, 1).cuda()
timer = 0

saved_points = []
saved_gt = []
saved_folds = [None for _ in range(opt.test_iters+1)]
saved_final = None

dump_input = torch.randn((1,3,1024)).float().cuda()
traced_model = torch.jit.trace(model, (dump_input, dump_input))


for i, data in enumerate(tqdm(test_dataloader, 0)):
	torch.cuda.synchronize()
	# 3.1 load inputs and targets
	with torch.no_grad():
		points, volume_length, gt_xyz, offset = data
		points, volume_length, gt_xyz, offset = points.cuda(), volume_length.cuda(), gt_xyz.cuda(), offset.cuda()

		t = time.time()
		folds= model(points.transpose(1,2), points.transpose(1,2))
		estimation = folds[-1]
		timer += time.time() - t

	torch.cuda.synchronize()

	# 3.3 compute error in world cs
	for it in range(opt.test_iters+1):
		outputs_xyz = folds[it]
		diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
		diff_sum = torch.sum(diff,2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
		diff_mean_wld = torch.mul(diff_mean,volume_length)

		test_wld_err[it] = test_wld_err[it] + diff_mean_wld.sum()
 
		fold = outputs_xyz.view(-1,opt.JOINT_NUM,3)
		if saved_folds[it] is None:
			saved_folds[it] = np.array(fold.cpu().numpy())
		else:
			saved_folds[it] = np.concatenate((saved_folds[it],fold.cpu().numpy()), 0)

# time taken
torch.cuda.synchronize()
# timer = time.time() - timer
timer = timer / len(test_data)
print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

# print mse
print('average estimation error in world coordinate system: ')
for it in range(opt.test_iters+1):
	print(' %f (mm)' %(test_wld_err[it] / len(test_data)))
