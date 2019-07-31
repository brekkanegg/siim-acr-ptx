import collections
import os
import torch
import numpy as np
import time
import glob

import torch.nn.functional as F
from scipy import ndimage

def set_save_dir(params):
    if (~params.is_train) or (params.resume_epoch !=0):
        save_dir = glob.glob('./ckpt/seed-{}*'.format(params.seed))[0]
        # todo: check params matching

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


def calc_dice_info(pid, lbl, lbl_logit, mask, mask_logit, eps=1e-7, follow_aux=False, th=0.9):
    t1 = time.time()

    batch_size = mask_logit.shape[0]

    infos = []
    for bi in range(batch_size):

        bi_lbl_logit = lbl_logit[bi]
        if torch.softmax(bi_lbl_logit, dim=0)[1] > th:
            bi_pred_lbl = 1
        else:
            bi_pred_lbl = 0


        bi_mask_logit = mask_logit[bi]

        # pred = (torch.argmax(F.softmax(bi_logit, dim=0), dim=0)).type(torch.float32)
        bi_pred_mask = (torch.softmax(bi_mask_logit, dim=0)[1, :, :] > th).type(torch.float32)

        if follow_aux:
            if not bi_pred_lbl:
                bi_pred_mask = torch.zeros_like(bi_pred_mask)


        bi_gt_mask = mask[bi]
        bi_gt_mask = bi_gt_mask.type(bi_mask_logit.type())

        bi_2TP = 2 * torch.sum(bi_pred_mask * bi_gt_mask).item()
        bi_TP_FP = (torch.sum(bi_pred_mask) + torch.sum(bi_gt_mask)).item()
        if (bi_2TP == 0) and (bi_TP_FP == 0):
            bi_dice = 1.
        else:
            bi_dice = bi_2TP / (bi_TP_FP+eps)

        infos.append([pid[bi], lbl[bi], bi_pred_lbl, bi_dice, bi_2TP, bi_TP_FP])


    return infos
