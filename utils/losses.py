import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, batch_weight=None): #, size_average=True, batch_average=True):
        # self.ignore_index = ignore_index
        self.weight = weight
        self.batch_weight = batch_weight
        # self.size_average = size_average
        # self.batch_average = batch_average
        # self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BCELoss
        elif mode == 'dice':
            return self.DICELoss
        elif mode == 'boundary':
            return self.BoundaryLoss

        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        # logit = torch.sigmoid(logit)

        criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')
        loss = criterion(logit, target.long())

        # if self.batch_average:
        #     loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        # if self.batch_average:
        #     loss /= n

        return loss

    def BCELoss(self, logit, target): # sigmoid

        # if self.weight is not None:
        #     pos_weight = self.weight[1]/self.weight[0]

        if self.batch_weight is not None:
            criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
            loss = torch.mean(torch.mean(criterion(logit, target), dim=(1,2))*self.batch_weight)

        else:
            criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')
            loss = criterion(logit, target)


        return loss



    def DICELoss(self, logit, target, smooth=1, is_train=True):

        if is_train:
            # logit = torch.sigmoid(logit)
            logit = F.softmax(logit, dim=1)[:, 1, :, :]
            target = target.type(logit.type())

            if self.batch_weight is not None:
                intersection = torch.sum(logit*target, dim=(1,2))
                dice = (2*intersection + smooth)/\
                       (torch.sum(target, dim=(1,2)) +
                        torch.sum(logit, dim=(1,2)) + smooth)
                dice = torch.mean(dice*self.batch_weight)

            else:
                intersection = torch.sum(logit*target)
                if (intersection.item() == 0) and ((torch.sum(target) + torch.sum(logit)).item()==0):
                    dice = torch.tensor(1).cuda()
                else:
                    dice = (2*intersection + smooth)/(torch.sum(target) + torch.sum(logit) + smooth)

            return (1-dice)

        else:
            # logit = (torch.sigmoid(logit) > 0.5).type(torch.float32)
            logit = torch.argmax(F.softmax(logit, 1), dim=1).type(torch.float32)
            target = target.type(logit.type())
            intersection = torch.sum(logit * target)
            dice = (2 * intersection + smooth) / (torch.sum(target) + torch.sum(logit) + smooth)

            return (1 - dice)


    def BoundaryLoss(self, logit, target):

        # logit = torch.sigmoid(logit)
        logit = F.softmax(logit, dim=1)[:, 1, :, :]
        target = target.type(logit.type())

        if self.batch_weight is not None:
            loss = torch.mean(logit*target, dim=(1,2,3))
            loss = torch.mean(loss*self.batch_weight)
        else:
            loss = torch.mean(logit*target)

        return loss



