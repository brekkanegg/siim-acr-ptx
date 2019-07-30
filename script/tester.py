import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

sys.path.append('..')

from utils import tools, visualize



def tester(model, val_loader, save_dir, device, writer, best_epoch):

    print('\n\nInference\n')

    tot_val_info_df = []

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        for val_fp, val_img, val_seg, val_lbl in tqdm(val_loader):
            val_img = val_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            val_seg = torch.squeeze(val_seg, -1)
            val_img = val_img.to(device, dtype=torch.float)
            val_seg = val_seg.to(device, dtype=torch.long)

            val_outputs = model(val_img)

            val_dice_info = tools.calc_dice_info(val_fp, val_outputs, val_seg)
            tot_val_info_df.extend(val_dice_info)

            visualize.summary_fig(0, save_dir, val_img.detach(), val_seg.detach(), val_outputs.detach(), writer=writer,
                                       type='inf', draw_num=len(val_fp), save=True, fps=val_fp)


        # save fig
        print(save_dir)


        tot_val_info_df = pd.DataFrame(tot_val_info_df,
                                       columns=['fp',
                                                'dice', '2TP', 'TP_FP'

                                                ])

        tot_val_info_df.to_csv(save_dir + '/tot_val_info_df_{}.csv'.format(best_epoch), index=False)
        avg_val_dice = np.mean(tot_val_info_df['dice'])

        print('avg val dice: {:.4f}'.format(avg_val_dice))

