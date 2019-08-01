import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import cv2

sys.path.append('..')

from utils import tools, visualize
from mask_functions import mask2rle




def tester(model, val_loader, save_dir, device, writer, best_epoch, submit=False, th=0.9):

    print('\n\nInference\n')


    tot_val_info_df = []

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    if not submit:
        with torch.no_grad():
            for val_fp, val_img, val_seg, val_lbl in tqdm(val_loader):
                val_img = val_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
                val_seg = torch.squeeze(val_seg, -1)
                val_img = val_img.to(device, dtype=torch.float)
                val_seg = val_seg.to(device, dtype=torch.long)

                val_outputs, val_lbl_outputs = model(val_img)

                val_dice_info = tools.calc_dice_info(val_fp, val_lbl, val_lbl_outputs, val_seg, val_outputs, follow_aux=False, th=th)
                tot_val_info_df.extend(val_dice_info)


            visualize.summary_fig(0, save_dir, val_img.detach(), val_seg.detach(), val_outputs.detach(), writer=writer,
                                       type='inf', draw_num=len(val_fp), save=True, fps=val_fp)


            # save fig
            print(save_dir)

            tot_val_info_df = pd.DataFrame(tot_val_info_df, columns=['fp', 'lbl', 'pred_lbl', 'dice', '2TP', 'TP_FP'])
            tot_val_info_df.to_csv(save_dir + '/tot_val_info_df_{}'.format(best_epoch), index=False)

            avg_val_dice = np.sum(tot_val_info_df['2TP']) / np.sum(tot_val_info_df['TP_FP'])
            print('avg val dice: {:.4f}'.format(avg_val_dice))
            avg_val_kaggle_dice = np.mean(tot_val_info_df['dice'])
            print('avg val kaggle dice: {:.4f}'.format(avg_val_kaggle_dice))
            val_acc = np.sum(tot_val_info_df['pred_lbl'] == tot_val_info_df['lbl']) / len(tot_val_info_df)
            print('avg val acc: {:.4f}'.format(val_acc))



    else:
        print('Make Submission CSV')
        submission_csv = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

        with torch.no_grad():
            i = 0
            for val_fp, val_img, _, _ in tqdm(val_loader):
                val_img = val_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
                val_img = val_img.to(device, dtype=torch.float)

                val_outputs, val_lbl_outputs = model(val_img)


                test_outputs = (torch.softmax(val_outputs, dim=1)[0, 1, :, :] > th).detach().cpu().numpy()
                test_outputs = cv2.resize(test_outputs, (1024, 1024), interpolation=cv2.INTER_CUBIC)



                rle = mask2rle(test_outputs*255, 1024, 1024)
                if len(rle) == 0:
                    rle = -1

                imgid = val_fp[0].split('/')[-1][:-4]
                submission_csv.loc[i, 'ImageId'] = imgid
                submission_csv.loc[i, 'EncodedPixels'] = rle
                i += 1

        submission_csv.to_csv('./submission/{}_th{}_submit.csv'.format(th, save_dir.split('/ckpt/')[1]), index=False)