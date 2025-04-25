import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.IDENet import IDENet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
import logging
import torch.backends.cudnn as cudnn
from options import opt

# set loss function
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

BCEwL = torch.nn.BCEWithLogitsLoss()
BCE = torch.nn.BCELoss()

class EIULoss(nn.Module):
    def __init__(self):
        super(EIULoss, self).__init__()

        self.alp = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, s1, s2, s3, s4, e1, e2, e3, e4, edges, gts):
        s1, s2, s3, s4 = torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4)
        e1, e2, e3, e4 = torch.sigmoid(e1), torch.sigmoid(e2), torch.sigmoid(e3), torch.sigmoid(e4)

        aoloss1 = self.alp * BCE(e1 * s1, edges) + (1 - self.alp) * BCE(e1 + s1 - e1 * s1, gts)
        aoloss2 = self.alp * BCE(e2 * s2, edges) + (1 - self.alp) * BCE(e2 + s2 - e2 * s2, gts)

        aoloss = aoloss1 + aoloss2
        return aoloss

eiu_loss = EIULoss().cuda()

if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
edge_root = opt.edge_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
val_edge_root = opt.val_edge_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'IDENet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("IDENet-Train")
model = IDENet(opt)
model.cuda()

num_parms = 0
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root,depth_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
val_loader = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_pre, save_path,
        opt.decay_epoch))


step = 0
best_mae = 1
best_em = 0
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path, modelname=""):
    global step
    model.train()


    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, edges, depth) in enumerate(train_loader, start=1):
            if(images.shape[0] == 1):
                continue
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()
            depth = depth.cuda().repeat(1, 3, 1, 1)

            s1,s2,s3,s4,e1,e2,e3,e4 = model(images,depth)

            bce_iou1 = BCEwL(s1, gts) + iou_loss(s1, gts)
            bce_iou2 = BCEwL(s2, gts) + iou_loss(s2, gts)
            bce_iou3 = BCEwL(s3, gts) + iou_loss(s3, gts)
            bce_iou4 = BCEwL(s4, gts) + iou_loss(s4, gts)
            bce_iou_deep_supervision = bce_iou1+0.8*bce_iou2+0.6*bce_iou3+0.4*bce_iou4
            
            bce_iou1_e = BCEwL(e1, edges) + iou_loss(e1, edges)
            bce_iou2_e = BCEwL(e2, edges) + iou_loss(e2, edges)
            bce_iou3_e = BCEwL(e3, edges) + iou_loss(e3, edges)
            bce_iou4_e = BCEwL(e4, edges) + iou_loss(e4, edges)
            bce_iou_deep_supervision_e = bce_iou1_e+0.8*bce_iou2_e+0.6*bce_iou3_e+0.4*bce_iou4_e

            eiuloss = eiu_loss.forward(s1, s2, s3, s4, e1, e2, e3, e4, edges, gts)

            loss = bce_iou_deep_supervision + bce_iou_deep_supervision_e + eiuloss
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f}, sal:{:4f}, edge:{:4f}, iu:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data, bce_iou_deep_supervision.data, bce_iou_deep_supervision_e.data, eiuloss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,memory_used))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if (epoch) % 5 == 0 and epoch>=40:
            torch.save(model.state_dict(), save_path + modelname + '_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + modelname + '_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def AlignmentTerm(dFM,dGT):
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    align_Matrix = 2. * (align_GT * align_FM)/ (align_GT* align_GT + align_FM* align_FM + 1e-8)
    return align_Matrix

def EnhancedAlignmentTerm(align_Matrix):
    enhanced = np.power(align_Matrix + 1,2) / 4
    return enhanced

def em_evl(pred, gt):
    th = 2 * pred.mean()
    if th > 1:
        th = 1
    FM = np.zeros(gt.shape)
    FM[pred >= th] = 1
    FM = np.array(FM,dtype=bool)
    GT = np.array(gt,dtype=bool)
    dFM = np.double(FM)
    if (sum(sum(np.double(GT)))==0):
        enhanced_matrix = 1.0-dFM
    elif (sum(sum(np.double(~GT)))==0):
        enhanced_matrix = dFM
    else:
        dGT = np.double(GT)
        align_matrix = AlignmentTerm(dFM, dGT)
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
    [w, h] = np.shape(GT)
    score = sum(sum(enhanced_matrix))/ (w * h - 1 + 1e-8)
    return score

# test function
def val(val_loader, model, epoch, save_path, modelname=""):
    global best_mae, best_em, best_epoch
    model.eval()
    with torch.no_grad():
        # mae_sum = 0
        em_sum = 0
        for i in range(val_loader.size):
            image, gt, depth, name, img_for_post = val_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda().repeat(1, 3, 1, 1)
            res,res2,res3,_,_,_,_,_ = model(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            em_sum += em_evl(res, gt)
        em = em_sum / val_loader.size

        print('Epoch: {} EM: {} ####  bestEM: {} bestEpoch: {}'.format(epoch, em, best_em, best_epoch))
        if epoch == 1:
            best_em = em
        else:
            if em > best_em:
                best_em = em
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + modelname + '_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} EM:{} bestEpoch:{} bestEM:{}, seed:{}'.format(epoch, em, best_epoch, best_em, random.seed()))


if __name__ == '__main__':
    print("Start train...")
    model_name = "IDENet"

    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, save_path, model_name)
        if epoch > 10:
            val(val_loader, model, epoch, save_path, model_name)
    
