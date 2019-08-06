import collections
import os
import torch
import numpy as np
import time
import glob

import torch.nn.functional as F
from scipy import ndimage
from skimage import measure

def set_save_dir(params):
    if (not params.is_train) or (params.resume_epoch !=0) or (params.submit):
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

        save_dict['w'] =  'd' + str(int(params.dice_weight)) \
                          + format(params.loss_type) + str(params.ce_weight) \
                          + 'f' + str(params.focal_weight)\
                          + 'cl' + str(params.class_weight)

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
        bi_pred_lbl = torch.softmax(bi_lbl_logit, dim=0)[1].cpu().numpy()


        bi_mask_logit = mask_logit[bi]

        # pred = (torch.argmax(F.softmax(bi_logit, dim=0), dim=0)).type(torch.float32)
        bi_pred_softmax_mask = torch.softmax(bi_mask_logit, dim=0)[1, :, :].cpu().numpy()
        bi_pred_mask = bi_pred_softmax_mask > th
        bi_pred_mask = post_process(bi_pred_softmax_mask, bi_pred_mask, min_size=350//4, max_num=3)

        if follow_aux:
            if not bi_pred_lbl:
                bi_pred_mask = torch.zeros_like(bi_pred_mask)


        bi_gt_mask = mask[bi].cpu().numpy()
        # bi_gt_mask = bi_gt_mask.type(bi_mask_logit.type())

        bi_2TP = 2 * np.sum(bi_pred_mask * bi_gt_mask)
        bi_TP_FP = (np.sum(bi_pred_mask) + np.sum(bi_gt_mask))
        if (bi_2TP == 0) and (bi_TP_FP == 0):
            bi_dice = 1.
        else:
            bi_dice = bi_2TP / (bi_TP_FP+eps)

        infos.append([pid[bi], lbl[bi].cpu().numpy(), bi_pred_lbl, bi_dice, bi_2TP, bi_TP_FP])


    return infos


# check post_process

def post_process(softmax_outputs, mask_outputs, min_size=350, max_num=3):
    # todo: cut by min_size, max_num by probabiliy
    # cut by min_size
    image_labels, num_features = ndimage.label(mask_outputs)

    if num_features > 0:
        region_probs_dict = {}
        regions = measure.regionprops(image_labels)
        for region in regions:
            if region.area < min_size:
                image_labels[image_labels == region.label] = 0
                num_features -= 1
            else:
                r_prob = np.mean(softmax_outputs[image_labels == region.label])
                region_probs_dict[region.label] = r_prob

        if num_features > max_num:
            region_probs_dict = [(v, k) for k, v in region_probs_dict.items()]
            region_probs_dict.sort()
            i = 0
            while num_features > max_num:
                remove_idx = region_probs_dict[i][1]
                image_labels[image_labels == remove_idx] = 0
                num_features -= 1
                i += 1

        mask_outputs[image_labels >= 1] = 1

    return mask_outputs
