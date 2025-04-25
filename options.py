import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=320, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=70, help='every n epochs decay learning rate')
parser.add_argument('--load_pre', type=str, default='.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--save_path', type=str, default='./ucpts/', help='the path to save models and logs')

# usod10k
parser.add_argument('--rgb_root', type=str, default='/home/dataset/usod10k/TR/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='/home/dataset/usod10k/TR/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='/home/dataset/usod10k/TR/GT/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='/home/dataset/usod10k/TR/Boundary/', help='the training edge images root')

parser.add_argument('--val_rgb_root', type=str, default='/home/dataset/usod10k/VAL/RGB/', help='the val gt images root')
parser.add_argument('--val_depth_root', type=str, default='/home/dataset/usod10k/VAL/depth/', help='the val gt images root')
parser.add_argument('--val_gt_root', type=str, default='/home/dataset/usod10k/VAL/GT/', help='the val gt images root')


opt = parser.parse_args()
