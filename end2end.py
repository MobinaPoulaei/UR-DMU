import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

class FrozenBN(nn.Module):
	def __init__(self, num_channels, momentum=0.1, eps=1e-5):
		super(FrozenBN, self).__init__()
		self.num_channels = num_channels
		self.momentum = momentum
		self.eps = eps
		self.params_set = False

	def set_params(self, scale, bias, running_mean, running_var):
		self.register_buffer('scale', scale)
		self.register_buffer('bias', bias)
		self.register_buffer('running_mean', running_mean)
		self.register_buffer('running_var', running_var)
		self.params_set = True

	def forward(self, x):
		assert self.params_set, 'model.set_params(...) must be called before the forward pass'
		return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum, self.eps, torch.backends.cudnn.enabled)

	def __repr__(self):
		return 'FrozenBN(%d)'%self.num_channels

def freeze_bn(m, name):
	for attr_str in dir(m):
		target_attr = getattr(m, attr_str)
		if type(target_attr) == torch.nn.BatchNorm3d:
			frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
			frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean, target_attr.running_var)
			setattr(m, attr_str, frozen_bn)
	for n, ch in m.named_children():
		freeze_bn(ch, n)

#-----------------------------------------------------------------------------------------------#

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		outplanes = planes * 4
		self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		if self.nl is not None:
			out = self.nl(out)

		return out

class NonLocalBlock(nn.Module):
	def __init__(self, dim_in, dim_out, dim_inner):
		super(NonLocalBlock, self).__init__()

		self.dim_in = dim_in
		self.dim_inner = dim_inner
		self.dim_out = dim_out

		self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

		self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
		self.bn = nn.BatchNorm3d(dim_out)

	def forward(self, x):
		residual = x

		batch_size = x.shape[0]
		mp = self.maxpool(x)
		theta = self.theta(x)
		phi = self.phi(mp)
		g = self.g(mp)

		theta_shape_5d = theta.shape
		theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)

		theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
		theta_phi_sc = theta_phi * (self.dim_inner**-.5)
		p = F.softmax(theta_phi_sc, dim=-1)

		t = torch.bmm(g, p.transpose(1, 2))
		t = t.view(theta_shape_5d)

		out = self.out(t)
		out = self.bn(out)

		out = out + residual
		return out

#-----------------------------------------------------------------------------------------------#

class I3Res50(nn.Module):

	def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
		self.inplanes = 64
		super(I3Res50, self).__init__()
		self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
		self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

		nonlocal_mod = 2 if use_nl else 1000
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.drop = nn.Dropout(0.5)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
				nn.BatchNorm3d(planes * block.expansion)
				)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i%nonlocal_mod==nonlocal_mod-1))

		return nn.Sequential(*layers)

	def forward_single(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool1(x)

		x = self.layer1(x)
		x = self.maxpool2(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		return x

	def forward(self, batch):
		if batch['frames'].dim() == 5:
			feat = self.forward_single(batch['frames'])
		return feat

#-----------------------------------------------------------------------------------------------#

def i3_res50(num_classes, pretrainedpath):
		net = I3Res50(num_classes=num_classes, use_nl=False)
		state_dict = torch.load(pretrainedpath, weights_only=True)
		net.load_state_dict(state_dict)
		print("Received Pretrained model..")
		# freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
		return net

def i3_res50_nl(num_classes, pretrainedpath):
		net = I3Res50(num_classes=num_classes, use_nl=True)
		state_dict = torch.load(pretrainedpath, weights_only=False)
		net.load_state_dict(state_dict)
		# freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
		return net


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.autograd import Variable

def load_frame(frame_file):
    data = Image.open(frame_file[0])
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)
    return data


def load_rgb_batch(rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))  # Shape: (batch_size, chunk_size, 256, 340, 3)
    for i in range(frame_indices.shape[0]):  # Iterate over batches
        for j in range(frame_indices.shape[1]):  # Iterate over frames in each batch
            batch_data[i, j, :, :, :] = load_frame(rgb_files[frame_indices[i][j]])  # Use full paths
    return batch_data

def oversample_data(data):
    # 19, 16, 256, 340, 3
    data_flip = torch.flip(data, dims=[3])

    data_1 = data[:, :, :224, :224, :]
    data_2 = data[:, :, :224, -224:, :]
    data_3 = data[:, :, 16:240, 58:282, :]
    data_4 = data[:, :, -224:, :224, :]
    data_5 = data[:, :, -224:, -224:, :]

    data_f_1 = data_flip[:, :, :224, :224, :]
    data_f_2 = data_flip[:, :, :224, -224:, :]
    data_f_3 = data_flip[:, :, 16:240, 58:282, :]
    data_f_4 = data_flip[:, :, -224:, :224, :]
    data_f_5 = data_flip[:, :, -224:, -224:, :]

    return [data_1, data_2, data_3, data_4, data_5,
      data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]
      
import pickle
import torch
import re
import sys

c2_weights = "pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl"
pth_weights_out = "pretrained/i3d_r50_kinetics.pth"

c2 = pickle.load(open(c2_weights, 'rb'), encoding='latin')['blobs']
c2 = {k:v for k,v in c2.items() if 'momentum' not in k}

downsample_pat = re.compile('res(.)_(.)_branch1_.*')
conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
nl_pat = re.compile('nonlocal_conv(.)_(.)_(.*)_.*')

m2num = dict(zip('abc',[1,2,3]))
suffix_dict = {'b':'bias', 'w':'weight', 's':'weight', 'rm':'running_mean', 'riv':'running_var'}

key_map = {}
key_map.update({'conv1.weight':'conv1_w',
			'bn1.weight':'res_conv1_bn_s',
			'bn1.bias':'res_conv1_bn_b',
			'bn1.running_mean':'res_conv1_bn_rm',
			'bn1.running_var':'res_conv1_bn_riv',
			'fc.weight':'pred_w',
			'fc.bias':'pred_b',
			})

for key in c2:

	conv_match = conv_pat.match(key)
	if conv_match:
		layer, block, module = conv_match.groups()
		layer, block, module = int(layer), int(block), m2num[module]
		name = 'bn' if 'bn_' in key else 'conv'
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s%d.%s'%(layer-1, block, name, module, suffix)
		key_map[new_key] = key

	ds_match = downsample_pat.match(key)
	if ds_match:
		layer, block = ds_match.groups()
		layer, block = int(layer), int(block)
		module = 0 if key[-1]=='w' else 1
		name = 'downsample'
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s.%d.%s'%(layer-1, block, name, module, suffix)
		key_map[new_key] = key

	nl_match = nl_pat.match(key)
	if nl_match:
		layer, block, module = nl_match.groups()
		layer, block = int(layer), int(block)
		name = 'nl.%s'%module
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s.%s'%(layer-1, block, name, suffix)
		key_map[new_key] = key

pth = I3Res50(num_classes=400, use_nl=True)
state_dict = pth.state_dict()

new_state_dict = {key: torch.from_numpy(c2[key_map[key]]) for key in state_dict if key in key_map}
torch.save(new_state_dict, pth_weights_out)
torch.save(key_map, pth_weights_out+'.keymap')

# check if weight dimensions match
for key in state_dict:

	if key not in key_map:
		continue

	c2_v, pth_v = c2[key_map[key]], state_dict[key]
	assert str(tuple(c2_v.shape))==str(tuple(pth_v.shape)), 'Size Mismatch'
	print ('{:23s} --> {:35s} | {:21s}'.format(key_map[key], key, str(tuple(c2_v.shape))))
	

def folder_to_list(folder_path, reference_list_file, output_list_file):
    # Read reference folder names (without paths)
    with open(reference_list_file, 'r') as ref_file:
        ref_folders = [line.strip() for line in ref_file]
    ref_folders = [os.path.basename(line).replace('_i3d.npy', '') for line in ref_folders]
    # print("Processed reference folder names:", ref_folders, "\n")

    # Get all folder paths in the given directory
    folder_paths = {}
    for root, subdirs, _ in os.walk(folder_path):
        # print(f"Root: {root}, Subdirs: {subdirs}\n")
        for subdir in subdirs:
            if subdir in ref_folders:
                # print(f"Match found: {subdir}\n")
                folder_paths[subdir] = os.path.join(root, subdir)

    # Order folder paths based on reference list
    ordered_folders = [folder_paths[folder] for folder in ref_folders if folder in folder_paths]

    # Write the ordered folder paths to the output .list file
    with open(output_list_file, 'w') as f:
        f.write('\n'.join(ordered_folders) + '\n')
    #print(f"List file created at '{output_list_file}' with {len(ordered_folders)} entries.")
	
folder_path = '/home/mnafez/msad_dataset/test_frames'
reference_list_file = '/home/mnafez/RTFM/MSAD/RTFM/list/msad-i3d-test.list'
output_list_file = './msad-frame-test.list'
folder_to_list(folder_path, reference_list_file, output_list_file)

import os
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self,
                 order_list="./msad-frame-test.list",
                 frames_dir="/home/mnafez/msad_dataset/test_frames",
                 gt_dir="/home/mnafez/msad_dataset/gt_new.npy"):
        self.order_list = order_list
        self.frames_dir = frames_dir
        self.gt_dir = gt_dir
        
        self.video_list = [line.strip() for line in open(self.order_list)]
        self.gt_data = np.load(self.gt_dir, allow_pickle=True)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # Get the video ID from the list
        video_id = self.video_list[index]
        video_folder = os.path.join(self.frames_dir, video_id)

        # Get sorted list of frame files
        frame_files = natsorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
        
        # Get full paths of the frames
        frame_paths = [os.path.join(video_folder, frame_file) for frame_file in frame_files]

        # Get the corresponding labels (same length as frame_files)
        start_idx = sum(((len(natsorted(os.listdir(os.path.join(self.frames_dir, v)))))//16)*16 
                        for v in self.video_list[:index])
        end_idx = start_idx + (len(frame_files)//16)*16
        labels = self.gt_data[start_idx:end_idx]

        return frame_paths, labels


import torch
import numpy as np
import torch.nn as nn
from natsort import natsorted
from torch.autograd import Variable


import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import L1Loss
import torch.optim as optim
from torch.nn import MSELoss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve

class WrapperModel(nn.Module):
    def __init__(self, pretrained_i3d, pretrained_net, num_classes, n_features, batch_size):
        super().__init__()
        self.i3d = I3Res50(num_classes=num_classes, use_nl=False)
        state_dict_i3d = torch.load(pretrained_i3d, weights_only=True)
        self.i3d.load_state_dict(state_dict_i3d)
        
        self.net = WSAD(n_features, flag = "Test", a_nums = 60, n_nums = 60)
        # self.net = self.net.cuda()
        # self.rtfm = RTFM(n_features=n_features, batch_size=batch_size)
        state_dict_net = torch.load(pretrained_net, weights_only=True)
        self.net.load_state_dict(state_dict_net)
    
    def run(self, batch_data):
        batch_data = batch_data.unsqueeze(0) # [1, 16, 256, 340, 3]
        batch_data_ten_crop = oversample_data(batch_data)  # 10 ta [1, 16, 224, 224, 3]
        batch_data_ten_crop = [batch_data_ten_crop[i].float() for i in range(10)]
        full_features = [torch.empty((0,), dtype=torch.float32) for _ in range(10)]

        for i in range(10):
            b_data = batch_data_ten_crop[i].cuda().permute(0, 4, 1, 2, 3) # [1, 3, 16, 224, 224]
            inp = {'frames': b_data}
            features = self.i3d(inp).cpu()  # [1, 2048, 1, 1, 1]
            full_features[i] = torch.cat((full_features[i], features.unsqueeze(0)), dim=1)
            
        full_features = torch.stack(full_features).squeeze()  # Shape: [10, 2048]
        
        return full_features

    def forward(self, batch_data, device):
        feature = self.run(batch_data)
        feature = feature.unsqueeze(0).unsqueeze(2).to(device) # [1, 10, 1, 2048]
        # feature = np.expand_dims(feature, axis=0)
        # feature = torch.from_numpy(feature).to(device)
        # print('feature.grad', feature.grad)
        logits = self.net(feature)
        return logits
        

def pgd_attack(dataloader, model, device, frequency, batch_size, epsilon=8.0/255, num_steps=10):
    
        step_size = 2.5 * (epsilon / num_steps)
        model.eval()
        chunk_size = 16

        all_preds = torch.zeros(0, device=device)
        all_labels = torch.tensor(np.load('/home/mnafez/msad_dataset/gt_new.npy'))
        print(all_labels.shape)        
        video_num_ = 0 
        for frame_paths, labels in dataloader:
            print("video_number:", video_num_)
            with open('output.txt', "a") as f:  # Open in append mode
                f.write(f"video_number: { video_num_}\n")
            labels = labels[0].to(device)
            frame_cnt = len(frame_paths)
            assert frame_cnt > chunk_size, "Not enough frames to process"
            clipped_length = frame_cnt - chunk_size
            clipped_length = (clipped_length // frequency) * frequency
            frame_indices = [
                [j for j in range(i * frequency, i * frequency + chunk_size)]
                for i in range(clipped_length // frequency + 1)
            ] # chunke aval, az frame 0 ta 15, chunke dovom, az frame 16 ta 31 ...
        
            frame_indices = np.array(frame_indices)
            chunk_num = frame_indices.shape[0]
            batch_num = int(np.ceil(chunk_num / batch_size))
            frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    
            idx = 0
            final_logits = []
            for batch_id in range(batch_num):
                batch_datas = load_rgb_batch(frame_paths, frame_indices[batch_id]) # [17, 16, 256, 340, 3]
                for batch_data in batch_datas:
                    batch_data = torch.tensor(batch_data).to(device) # [16, 256, 340, 3]
                    original_batch_data = batch_data.clone().detach()
                    adv_batch_data = batch_data.clone().detach()
            
                    num_snippets = 1
                    snippet_labels = labels[idx:idx + num_snippets*16]
                    snippet_gt_list = [snippet_labels[j * 16] for j in range(num_snippets)]
                    snippet_gt = torch.tensor(snippet_gt_list, device=device).float()  # Shape: [num_snippets]
                    idx += num_snippets*16
                    
                    for step in range(num_steps):
                        adv_batch_data.requires_grad_() # [16, 256, 340, 3]
                        logits = model(adv_batch_data, device)['frame']  # shape: [10, 1, 1]
                        #print(logits.shape)
                        logits = logits.view(-1) # [10, 1]
                        #print(f'===={logits.shape}=====')
                        # logits = torch.mean(logits, 0) # [1, 1]
                        model.zero_grad()
                        coef = torch.where(snippet_gt==0, torch.ones_like(snippet_gt), -torch.ones_like(snippet_gt))
                        #print(coef.shape)
                        cost = torch.dot(coef, logits)
                        cost.backward()
                        
                        grad_sign = adv_batch_data.grad.sign()
                        adv_batch_data = adv_batch_data.detach() + step_size * grad_sign
                        perturbation = torch.clamp(adv_batch_data - original_batch_data, min=-epsilon, max=epsilon)
                        adv_batch_data = torch.clamp(original_batch_data + perturbation, 0, 1)
        
                    with torch.no_grad():
                        final_logit = model(adv_batch_data, device)['frame']
                        #final_logit = torch.squeeze(final_logit, 1)
                        #final_logit = torch.mean(final_logit, 0)
                    final_logits.append(final_logit)
            video_num_ += 1
                    
            final_logits = torch.cat(final_logits, dim=0)
            all_preds = torch.cat((all_preds, final_logits))
    
        all_labels = all_labels.cpu().detach().numpy()
        all_preds = all_preds.cpu().detach().numpy()
        all_preds = np.repeat(np.array(all_preds), 16)
        print(all_labels.shape, all_preds.shape)
        fpr, tpr, threshold = roc_curve(all_labels, all_preds)
        rec_auc = auc(fpr, tpr)
        return rec_auc

test_loader = DataLoader(VideoDataset(),
                          batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=False)

pretraine_i3d = "/home/mnafez/UR-DMU/pretrained/i3d_r50_kinetics.pth"
pretrained_net = "/home/mnafez/UR-DMU/models/msad_trans_2022.pkl"
model = WrapperModel(pretraine_i3d, pretrained_net, num_classes=400, n_features=2048, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(pgd_attack(dataloader=test_loader, model=model, device=device, frequency=16, batch_size=20, epsilon=1.0/255, num_steps=10))
