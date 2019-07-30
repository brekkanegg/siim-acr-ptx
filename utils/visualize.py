import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import os
import warnings

def summary_fig(it, save_dir, img, seg, outputs, writer, type='train', draw_num=1, save=False, fps=None):
    warnings.filterwarnings("ignore")

    fig_label_l = seg.cpu().numpy()[:draw_num, :, :,]  # [:, np.newaxis, :, :]
    fig_out_l = torch.argmax(F.softmax(outputs, dim=1)[:draw_num, :, :, :], dim=1).cpu().numpy()  # [:, np.newaxis, : :, :]

    fig = plt.figure(figsize=(8*3, 8))
    plt.axis('off')

    for bi in range(draw_num):

        gray = img.cpu().numpy()[bi, 0, :, :]

        mask_label = fig_label_l[bi, :, :]
        mask_pred = fig_out_l[bi, :, :]

        ax00 = fig.add_subplot(1, 3, 1)
        ax01 = fig.add_subplot(1, 3, 2)
        ax02 = fig.add_subplot(1, 3, 3)

        # 원본 gray
        ax00.imshow(gray, cmap='gray')


        # gt seg: vessel(분홍) + mask(파랑)
        ax01.imshow(gray, cmap='gray')
        mask_masked_label = np.ma.masked_where(mask_label == 0, mask_label)
        ax01.imshow(mask_masked_label, cmap='winter', alpha=0.3)  # 파랑, 겹쳐지면 보라

        # pred seg: vessel(분홍) + mask(파랑)
        ax02.imshow(gray, cmap='gray')
        mask_masked_pred = np.ma.masked_where(mask_pred == 0, mask_pred)
        ax02.imshow(mask_masked_pred, cmap='autumn', alpha=0.3)  # 파랑, 겹쳐지면 보라

        if not save:
            writer.add_figure(type, fig, it, close=True)
        else:
            fp = fps[bi].split('-train/')[1]
            seed = save_dir.split('seed-')[1].split('_val-')[0]
            inf_img_dir = './inf_images/{}'.format(seed)
            if not os.path.exists(inf_img_dir):
                os.makedirs(inf_img_dir)

            plt.savefig(os.path.join(inf_img_dir, fp), bbox_inches='tight')

