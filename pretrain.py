import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


import torch.nn.functional as F

sys.path.append('..')

from params import params
from inputs import chest14
from models import unet



device = torch.device('cuda:0')
torch.cuda.set_device(0)



model = getattr(unet, 'UNet')(n_classes=1, pretrained=True).to(device)

optimizer = torch.optim.SGD(model.parameters(), #filter(lambda p: p.requires_grad, model.parameters())
                            lr=params.learning_rate,
                            momentum=params.beta1,
                            weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                 T_0=5,
                                                                 T_mult=1)
train_loader, val_loader = chest14.get_dataloader(params)
save_dir = './ckpt/pretrain/chest14'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if params.resume_epoch != 0 :
    resume_model_dir = os.path.join(save_dir, 'epoch_{}.pth.tar'.format(params.resume_epoch))
    print('\nresume model dir:', resume_model_dir, '\n')
    state_dict = torch.load(resume_model_dir)
    model.load_state_dict(state_dict)




writer = SummaryWriter(log_dir=save_dir)





def summary_fig(it, img, outputs, writer, type='train'):


    fig = plt.figure(figsize=(8*2, 8))
    plt.axis('off')


    img = img.cpu().numpy()[0, 0, :, :]
    out = outputs.cpu().numpy()[0, 0, :, :]

    ax00 = fig.add_subplot(1, 2, 1)
    ax01 = fig.add_subplot(1, 2, 2)

    ax00.imshow(img, cmap='gray')
    ax01.imshow(out, cmap='gray')

    writer.add_figure(type, fig, it, close=True)





start_time = time.time()
total_step = len(train_loader)
display_step = len(train_loader) // params.display_interval# // 10

it = params.resume_epoch*len(train_loader)
for epoch in range(params.resume_epoch, params.max_epoch):
    scheduler.step(epoch)  # sgdr

    model.train()
    for i, img in enumerate(train_loader):


        img = img.permute(0, 3, 1, 2)  # NXY1 -> N1XY
        img = img.to(device, dtype=torch.float)

        outputs, _ = model(img)
        optimizer.zero_grad()

        l2_loss = nn.MSELoss()(outputs, img)


        loss = l2_loss
        loss.backward()

        optimizer.step()



        it += 1

        if it % display_step == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Time: [{:.4f}], l2: {:.4f}'
                .format(epoch + 1, params.max_epoch, i + 1, total_step, time.time() - start_time,
                        l2_loss.item()))

            summary_fig(it, img.detach(), outputs.detach(), writer=writer, type='train', )

    print('Validation')
    model.eval()
    with torch.no_grad():
        tot_vimgs = []
        tot_voutputs = []
        for vi, vimg in enumerate(val_loader):
            vimg = vimg.permute(0, 3, 1, 2)  # NXY1 -> N1XY
            vimg = vimg.to(device, dtype=torch.float)

            voutput, _ = model(vimg)
            tot_vimgs.append(vimg)
            tot_voutputs.append(voutput)

        summary_fig(it, vimg, voutput, writer=writer, type='val')

        tot_vimgs = torch.cat(tot_vimgs, dim=0)
        tot_voutputs = torch.cat(tot_voutputs, dim=0)

        vl2_loss = nn.MSELoss()(tot_voutputs, tot_vimgs)
        print('val l2 loss: {:.4f}'.format(vl2_loss.item()))

        model_name = os.path.join(save_dir, 'epoch_' + repr(epoch + 1) + '.pth.tar')
        torch.save(model.state_dict(), model_name)



