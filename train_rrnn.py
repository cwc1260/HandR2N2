'''
training
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

from dataset_msra import HandPointDataset

from utils import rotate_point_cloud_by_angle_flip, rotate_point_cloud_by_angle_nyu, rotate_point_cloud_by_angle_xyz

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=160, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (SGD only)')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
parser.add_argument('--iters', type=int, default = 3, help='start epoch')

parser.add_argument('--start_epoch', type=int, default = 0, help='start epoch')
parser.add_argument('--test_index', type=int, default = 0, help='test index for cross validation, range: 0~8')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '', help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '', help='optimizer name for training resume')

parser.add_argument('--dataset', type=str, default = 'nyu', help='optimizer name for training resume')
parser.add_argument('--dataset_path', type=str, default = '/workspace/HandFoldDynGraph/data/NYU/process_nyu_center_rot0_2048_com/Training',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '/workspace/HandFoldDynGraph/data/NYU/process_nyu_center_com/Testing',  help='model name for training resume')


parser.add_argument('--model_name', type=str, default = 'rrnn',  help='')
parser.add_argument('--gpu', type=str, default = '0',  help='gpu')

opt = parser.parse_args()

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset == 'msra':
	from dataset_msra import subject_names
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+str(opt.iters)+'iters', subject_names[opt.test_index])
	opt.JOINT_NUM = 21
else:
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+str(opt.iters)+'iters')
	if opt.dataset == 'icvl':
		from dataset_icvl import HandPointDataset
		from dataset_icvl_arm_com import HandPointDatasetArm
		opt.JOINT_NUM = 16
	else:
		from dataset_nyu import HandPointDataset
		from dataset_nyu_arm_com import HandPointDatasetArm
		opt.JOINT_NUM = 14


def _debug(model):
	model = model.netR_1
	print(model.named_paramters())
try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
if opt.dataset == 'icvl' or  opt.dataset == 'nyu':	
	train_data = HandPointDatasetArm(root_path=opt.dataset_path, opt=opt, train = True, sample=2048, output_num=1024)

else:									  
	train_data = HandPointDataset(root_path=opt.dataset_path, opt=opt, train = True)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
										shuffle=True, num_workers=int(opt.workers), pin_memory=False)

test_data = HandPointDataset(root_path=opt.test_path, opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, iters=opt.iters)

if opt.ngpu > 1:
	model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
	model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
	model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':
	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	
model.cuda()
print(model)

parameters = model.parameters()

def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer
criterion = smooth_l1_loss

optimizer = optim.AdamW(parameters, lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06, weight_decay=opt.weight_decay)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.1)
if opt.dataset == 'icvl':
	scheduler = lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
if opt.dataset == 'nyu':
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

test_best_error = np.inf

# 3. Training and testing
for epoch in range(opt.start_epoch, opt.nepoch):
	scheduler.step(epoch)
	if opt.dataset == 'msra':
		print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %(epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
	else:
		print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))

	# 3.1 switch to train mode
	torch.cuda.synchronize()
	model.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()

	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		points, volume_length, gt_xyz , _= data
		# gt_pca = Variable(gt_pca, requires_gr,ad=False).cuda()
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()

		permutation = torch.randperm(points.size(1))
		points = points[:,permutation,:]

		points, gt_xyz = rotate_point_cloud_by_angle_xyz(points, gt_xyz.view(-1, opt.JOINT_NUM , 3), False)

		shift = ((torch.rand((points.size(0),3)).cuda() * 40. - 20.) / volume_length).view(-1, 1, 3)
		points[:,:,0:3] = points[:,:,0:3] + shift
		gt_xyz = gt_xyz + shift

		scale = (torch.rand(points.size(0)).cuda() * 0.4 + 0.8).view(-1, 1, 1)
		points = points * scale
		gt_xyz = gt_xyz * scale		
		
		gt_xyz = gt_xyz.view(-1, opt.JOINT_NUM * 3)

		# print(gt_xyz.size())
		# points: B * 1024 * 6; target: B * 42
		# 3.1.2 compute output
		optimizer.zero_grad()

		folds= model(points.transpose(1,2), points.transpose(1,2))
		estimation = folds[-1]
		loss = criterion(estimation, gt_xyz)*1
		for i in range(len(folds) - 1):

			loss += criterion(folds[i], gt_xyz) * (0.8**(len(folds)-i))
		loss = loss*opt.JOINT_NUM * 3

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		train_mse = train_mse + loss.item()*len(points)
		
		# 3.1.5 compute error in world cs      
		outputs_xyz = estimation
		diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
		diff_sum = torch.sum(diff,2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
		diff_mean_wld = torch.mul(diff_mean,volume_length)
		train_mse_wld = train_mse_wld + diff_mean_wld.sum().item()

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	# print mse
	train_mse = train_mse / len(train_data)
	train_mse_wld = train_mse_wld / len(train_data)


	print('mean-square error of 1 sample: %f, #train_data = %d' %(train_mse, len(train_data)))
	print('average estimation error in world coordinate system: %f (mm)' %(train_mse_wld))

	if (epoch % 10) == 0:
		torch.save(model.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
		torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))

	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	model.eval()
	test_mse = 0.0
	test_wld_err = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, 0)):
		torch.cuda.synchronize()
		with torch.no_grad():
			# 3.2.1 load inputs and targets
			points, volume_length, gt_xyz, _ = data
			points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()

			folds= model(points.transpose(1,2), points.transpose(1,2))
			estimation = folds[-1]
			loss = criterion(estimation, gt_xyz)*1
			for i in range(len(folds) - 1):
				loss += criterion(folds[i], gt_xyz) * (0.8**(len(folds)-i))
				# loss += criterion(folds[i], gt_xyz) * 1
			loss = loss*opt.JOINT_NUM * 3

		torch.cuda.synchronize()
		test_mse = test_mse + loss.item()*len(points)

		# 3.2.3 compute error in world cs        
		outputs_xyz = estimation
		diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
		diff_sum = torch.sum(diff,2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
		diff_mean_wld = torch.mul(diff_mean,volume_length)
		test_wld_err = test_wld_err + diff_mean_wld.sum().item()

	if test_best_error > test_wld_err:
		test_best_error = test_wld_err
		torch.save(model.state_dict(), '%s/best_model.pth' % (save_dir))
		torch.save(optimizer.state_dict(), '%s/best_optimizer.pth' % (save_dir))
				
	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	# print mse
	test_mse = test_mse / len(test_data)
	print('mean-square error of 1 sample: %f, #test_data = %d' %(test_mse, len(test_data)))
	test_wld_err = test_wld_err / len(test_data)
	print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))
	# log
	logging.info('Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, best wld error = %f, lr = %f' %(epoch, train_mse, train_mse_wld, test_mse, test_wld_err, test_best_error / len(test_data), scheduler.get_lr()[0]))