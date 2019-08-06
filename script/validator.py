import os
import sys
import time

import torch

import numpy as np
import pandas as pd

sys.path.append('..')

from utils import tools, visualize, losses

from sklearn import metrics


def validator(model, it, epoch, val_loader, save_dir, device, writer, past_record, follow_aux=False, th=0.9):



    print('===Validation===')
    print(save_dir)


    tot_val_info_df = []

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        vt1 = time.time()
        for val_i, (val_fp, val_img, val_seg, val_lbl) in enumerate(val_loader):
            val_img = val_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            val_seg = torch.squeeze(val_seg, -1)
            val_img = val_img.to(device, dtype=torch.float)
            val_seg = val_seg.to(device, dtype=torch.long)
            val_lbl = val_lbl.to(device, dtype=torch.long)

            val_outputs, val_lbl_outputs = model(val_img)

            val_dice_info = tools.calc_dice_info(val_fp, val_lbl, val_lbl_outputs, val_seg, val_outputs,
                                                 follow_aux=follow_aux, th=th)
            tot_val_info_df.extend(val_dice_info)


        # save fig
        visualize.summary_fig(it, save_dir, val_img, val_seg, val_outputs, writer=writer, type='val')


        # Vessel, Lumen Dice

        tot_val_info_df = pd.DataFrame(tot_val_info_df, columns=['fp', 'lbl', 'pred_lbl', 'dice', '2TP', 'TP_FP'])
        tot_val_info_df.to_csv(save_dir + '/tot_val_info_df_{}'.format(epoch + 1), index=False)


        avg_val_dice = np.sum(tot_val_info_df['2TP']) / np.sum(tot_val_info_df['TP_FP'])
        print('avg val dice: {:.4f}'.format(avg_val_dice))
        avg_val_kaggle_dice = np.mean(tot_val_info_df['dice'])
        print('avg val kaggle dice: {:.4f}'.format(avg_val_kaggle_dice))


        val_lbl_fpr, val_lbl_tpr, val_lbl_threshold = metrics.roc_curve(list(tot_val_info_df['lbl']),
                                                                        list(tot_val_info_df['pred_lbl']), pos_label=1)
        val_lbl_auc = metrics.auc(val_lbl_fpr, val_lbl_tpr)
        val_lbl_acc = np.sum((tot_val_info_df['pred_lbl']>th) == tot_val_info_df['lbl']) / len(tot_val_info_df)
        print('avg val lbl AUC: {:.4f}'.format(val_lbl_auc))
        # print('avg val lbl recall: ', (val_lbl_tpr))
        # print('avg val lbl specificity: ', (1-val_lbl_fpr))
        print('avg val lbl Acc: {:.4f}'.format(val_lbl_acc))

        # print('val time: {:.4f}'.format(time.time() - vt1))

        # Logging

        (best_metric, that_dice, that_k_dice, that_acc, that_epoch) = past_record

        choose_metric = avg_val_dice
        if choose_metric > best_metric:
            best_metric = choose_metric
            model_name = os.path.join(save_dir, 'epoch_' + repr(epoch + 1) + '.pth.tar')
            torch.save(model.state_dict(), model_name)

            that_dice = avg_val_dice
            that_k_dice = avg_val_kaggle_dice
            that_epoch = epoch + 1
            that_acc = val_lbl_acc

        print('Best Metric {:.4f}, Dice {:.4f} KDice {:.4f}, Acc {:.4f} at epoch {}'.
              format(best_metric, that_dice, that_k_dice, that_acc, that_epoch))

        print('================')


        if epoch == 0 :
            training_log = pd.DataFrame(columns=['epoch', 'train_dice', 'val_dice'])
        else:
            training_log = pd.read_csv(os.path.join(save_dir, 'training_log.csv'))
        start_row = training_log.shape[0] + 1

        training_log.loc[start_row + epoch, 'epoch'] = epoch + 1
        training_log.loc[start_row + epoch, 'val_loss'] = 1 - avg_val_dice
        training_log.to_csv(os.path.join(save_dir, 'training_log.csv'))


        past_record = (best_metric, that_dice, that_k_dice, that_acc, that_epoch)

        return past_record, avg_val_dice, val_lbl_acc


