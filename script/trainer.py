import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append('..')
from utils import tools, visualize, losses
from script import validator



def trainer(model, optimizer, scheduler, train_loader, val_loader, params, save_dir, device, writer,
            it=0, freeze_bn=False, use='bal', th=0.9):

    if params.resume_epoch != 0:
        assert it != 0

    past_record = (-1, -1, -1, -1, -1)

    start_time = time.time()

    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for epoch in range(params.resume_epoch, params.max_epoch):
        scheduler.step(epoch)  # sgdr

        if use == 'bal':
            train_normal_loader, train_abnormal_loader = train_loader
            total_step = len(train_abnormal_loader)
            display_step = len(train_abnormal_loader) // params.display_interval

            for i, ((n_fp, n_img, n_seg, n_lbl), (ab_fp, ab_img, ab_seg, ab_lbl)) in enumerate(
                    zip(train_normal_loader, train_abnormal_loader)):

                fp = list(n_fp) + list(ab_fp)
                img = torch.cat((n_img, ab_img), dim=0)
                seg = torch.cat((n_seg, ab_seg), dim=0)
                lbl = torch.cat((n_lbl, ab_lbl), dim=0)

                it = train(fp, img, seg, lbl, device, model, optimizer, params, writer, it, display_step, epoch,
                          save_dir, i, total_step, start_time)



        else:
            total_step = len(train_loader)
            display_step = len(train_loader) // params.display_interval

            for i, (fp, img, seg, lbl) in enumerate(train_loader):
                it = train(fp, img, seg, lbl, device, model, optimizer, params, writer, it, display_step, epoch,
                          save_dir, i, total_step, start_time)


        # fixme
        past_record, avg_val_dice, val_acc = \
            validator.validator(model, it, epoch, val_loader, save_dir, device, writer, past_record, th)

        writer.add_scalars('dice', {'val dice': avg_val_dice}, it)
        writer.add_scalars('accuracy', {'val accuracy': val_acc}, it)



def calc_weight(seg, ub=0.8):
    if np.sum(seg.cpu().numpy()==1) == 0:
        weights = None
    else:
        weights = [np.sum(seg.cpu().numpy()==1), np.sum(seg.cpu().numpy()==0)]
        tot_size = np.sum(weights)
        weights = [w/tot_size for w in weights]
        if weights[1] > ub:  #
            weights = [1-ub, ub]
        weights = torch.Tensor(weights)
    return weights


def train(fp, img, seg, lbl, device, model, optimizer, params, writer, it, display_step, epoch, save_dir, i, total_step, start_time):

    img = img.permute(0, 3, 1, 2)  # NXY1 -> N1XY
    seg = torch.squeeze(seg, -1)
    img = img.to(device, dtype=torch.float)
    seg = seg.to(device, dtype=torch.long)
    lbl = lbl.to(device, dtype=torch.long)

    outputs, preds = model(img)
    optimizer.zero_grad()

    batch_weight = None
    # batch_weight = calc_batch_weight(seg).to(device, dtype=torch.float)

    # Dice Loss
    dice_criterion = losses.SegmentationLosses(weight=None, batch_weight=batch_weight).build_loss(
        mode='dice')
    dice_loss = dice_criterion(outputs, seg)

    # BCE
    ce_pos_weight = None
    if params.loss_type == 'wce':
        try:
            ce_pos_weight = calc_weight(seg).to(device)
        except AttributeError:
            ce_pos_weight = None
    ce_criterion = losses.SegmentationLosses(weight=ce_pos_weight, batch_weight=batch_weight).build_loss(
        mode='bce')
    ce_loss = ce_criterion(outputs, seg)

    # Focal
    focal_criterion = losses.SegmentationLosses(weight=None, batch_weight=batch_weight).build_loss(
        mode='focal')
    focal_loss = focal_criterion(outputs, seg)

    #################################
    # todo: multi-task learning + header data

    # Class BCE Loss
    class_pos_weight = torch.Tensor([0.2, 0.8]).to(device)
    class_criterion = losses.SegmentationLosses(weight=class_pos_weight).build_loss(mode='bce')
    class_loss = class_criterion(preds, lbl)
    class_accuracy = torch.sum((torch.argmax(preds, dim=1) == lbl)).item() / len(fp)

    loss = params.dice_weight * dice_loss + params.ce_weight * ce_loss + params.focal_weight * focal_loss
    loss += params.class_weight * class_loss
    loss.backward()

    optimizer.step()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalars('dice', {'train dice': 1 - dice_loss.item()}, it)
    writer.add_scalars('ce', {'train ce': ce_loss.item()}, it)
    writer.add_scalars('class ce', {'train class ce': class_loss.item()}, it)
    writer.add_scalars('focal', {'train focal': focal_loss.item()}, it)
    writer.add_scalars('accuracy', {'train accuracy': class_accuracy}, it)

    it += 1

    if it % display_step == 0:
        print(
            'Epoch [{}/{}], Step [{}/{}], Time: [{:.4f}], Dice: {:.4f}, Acc: {:.4f}, '
            'CE Loss: {:.4f}, Focal Loss: {:.4f}, Class Loss: {:.4f}'
                .format(epoch + 1, params.max_epoch, i + 1, total_step, time.time() - start_time,
                        1 - dice_loss.item(), class_accuracy,
                        ce_loss.item(), focal_loss.item(), class_loss.item()))

        visualize.summary_fig(it, save_dir, img.detach(), seg.detach(), outputs.detach(), writer=writer,
                              type='train', )

    return it