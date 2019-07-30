import collections
import os
import torch
import numpy as np
import time
import glob

import torch.nn.functional as F
from scipy import ndimage

def set_save_dir(params):
    if not params.is_train:
        save_dir = glob.glob('./ckpt/seed-{}*'.format(params.seed))[0]

    else:
        save_dict = collections.OrderedDict()
        save_dict['seed'] = params.seed
        save_dict['val'] = params.valset
        save_dict['size'] = params.image_size

        save_dict['net'] = params.network
        save_dict['opt'] = params.optimizer
        save_dict['lr'] = params.learning_rate
        if params.loss_type == 'wce':
            save_dict['w'] =  str(int(params.dice_weight)) \
                              + 'w' + str(int(params.ce_weight)) \
                              + str(int(params.focal_weight))\
                              + 'c' +str(params.class_weight)
        else:
            save_dict['w'] = str(int(params.dice_weight)) \
                             + str(int(params.ce_weight)) \
                             + str(int(params.focal_weight))\
                             + 'c' +str(params.class_weight)

        save_dict['use'] = params.use

        save_dir = ['{}-{}'.format(key, save_dict[key]) for key in save_dict.keys()]
        save_dir = './ckpt/' + '_'.join(save_dir)
        save_dir = save_dir.replace('True', 'true').replace('False', 'false')
        print('Save Directory: ', save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    return save_dir


def calc_dice_info(pid, logit, lbl_logit, target, eps=1e-7, aux=False):
    t1 = time.time()

    batch_size = logit.shape[0]

    infos = []

    for bi in range(batch_size):

        bi_logit = logit[bi]
        pred = (torch.argmax(F.softmax(bi_logit, dim=0), dim=0)).type(torch.float32)

        if aux:
            bi_lbl_logit = lbl_logit[bi]
            if torch.argmax(bi_lbl_logit, dim=0) == 0:
                pred = torch.zeros_like(pred)

        bi_target = target[bi]
        gt = bi_target.type(logit.type())

        TP2 = 2 * torch.sum(pred * gt).item()
        TP_FP = (torch.sum(pred) + torch.sum(gt)).item()
        if (TP2 == 0) and (TP_FP == 0):
            dice = 1.
        else:
            dice = TP2 / (TP_FP+eps)

        infos.append([pid[bi],
                      dice, TP2, TP_FP,
                      ])


    return infos
