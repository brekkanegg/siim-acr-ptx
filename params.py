import os
import argparse
import collections
from datetime import datetime

def str2bool(v):
    if v is None:
        return None
    return v.lower() in ('true')


# control here
parser = argparse.ArgumentParser()


# GPU
parser.add_argument('--gpu', '--g', type=str, default='0')


# CPU
parser.add_argument('--num_workers', '--nw', type=int, default=0)


# Dataset
parser.add_argument('--num_classes', '--nc', type=int, default=2)
parser.add_argument('--image_size', '--in', type=int, default=512)  # IMAGENET Pretrained
parser.add_argument('--valset', '--val', type=int, default=1)
parser.add_argument('--use', '--u', type=str, default='all')



# Augmentations
parser.add_argument('--augmentation', '--aug', type=str2bool, default=False)
parser.add_argument('--hflip', type=str2bool, default=False)
parser.add_argument('--vflip', type=str2bool, default=False)
parser.add_argument('--rotate', type=str2bool, default=True)
parser.add_argument('--scale', type=str2bool, default=True)
parser.add_argument('--translate', type=str2bool, default=True)
parser.add_argument('--elastic', type=str2bool, default=False)
parser.add_argument('--contrast', type=str2bool, default=True)
parser.add_argument('--gamma_contrast', type=str2bool, default=True)
parser.add_argument('--gaussian_blur', '--gau', type=str2bool, default=True)



# Optimization
parser.add_argument('--optimizer', '--opt', type=str, default='sgd')
parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4)
parser.add_argument('--beta1', '--b1', type=float, default=0.9)
parser.add_argument('--beta2', '--b2', type=float, default=0.99)
parser.add_argument('--gradient_clip', '--gdc', type=str2bool, default=False)


# Loss
parser.add_argument('--loss_type', '--loss', type=str, default='wce')

parser.add_argument('--dice_weight', '--dw', type=float, default=0.)
parser.add_argument('--ce_weight', '--cw', type=float, default=1.)
parser.add_argument('--focal_weight', '--fw', type=float, default=0.)
parser.add_argument('--class_weight', '--clw', type=float, default=0.1)

# parser.add_argument('--boundary_weight', '--bw', type=float, default=0.)



# Training
parser.add_argument('--is_train', '--train', type=str2bool, default=True)
parser.add_argument('--resume_epoch', '--rep', type=int, default=0)
parser.add_argument('--max_epoch', '--mep', type=int, default=100)
parser.add_argument('--warmup_epoch', '--wep', type=int, default=4)

parser.add_argument('--batch_size', '--bs', type=int, default=10)
parser.add_argument('--display_interval', '--dp', type=int, default=4)


# Inference
parser.add_argument('--inference_epoch', '--iep', type=int, default=0)



# Network
parser.add_argument('--network', '--n', type=str, default='UNetRes50') # UNetRes50
parser.add_argument('--backbone', '--bb', type=str, default='resnet50')
parser.add_argument('--use_pretrain', '--p', type=str2bool, default=True)
parser.add_argument('--act', '--a', type=str, default='sigmoid')
parser.add_argument('--up', type=str, default='bi')




# Options
parser.add_argument('--seed', '--s', type=int, default=None)  #






# Utils
parser.add_argument('--draw', '--dfn', type=str2bool, default=False)  #
parser.add_argument('--debug_mode', '--dm', type=str2bool, default=False)  #


# Ray - Tune
parser.add_argument('--smoke_test', '--st', type=str2bool, default=False)  #
parser.add_argument('--tune', type=str, default='hb')  #

params = parser.parse_args()


if params.resume_epoch:
    assert params.seed is not None


if params.seed is None:
    params.seed = datetime.now().strftime('%Y%m%d%H%M%S')[4:]


if not params.augmentation:
    params.hflip = False
    params.vflip = False
    params.rotate = False
    params.scale = False
    params.translate = False
    params.elastic = False



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(params.gpu)



