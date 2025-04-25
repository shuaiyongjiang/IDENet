import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.IDENet import IDENet
from data import test_dataset
import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=320, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/dataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = IDENet(opt)
model.cuda()

epoch = 'best'
model_name = "IDENet"
model.load_state_dict(torch.load('./ucpts/' + model_name + '_epoch_' + epoch + '.pth'))
model.eval()


test_datasets = ['usod10k']

t_all =[]

for dataset in test_datasets:
    save_path = './results/'+ model_name + '/' + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/TE/RGB/'
    gt_root = dataset_path + dataset + '/TE/GT/'
    depth_root = dataset_path + dataset + '/TE/depth/'

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1,3,1,1).cuda()

        t1 = time.time()
        res, res2, res3, res4, edge1, _, _, _ = model(image,depth)
        t2 = time.time()
        t_all.append(t2-t1)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)

    print('Test Done!')

print("fps:", 1/np.mean(t_all))