import matplotlib
matplotlib.use('Agg')  # tensorboardX
import os
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tensorboardX import SummaryWriter

from params import params
device = torch.device('cuda:0')
torch.cuda.set_device(0)

from script import trainer, validator, tester
from inputs import pneumothorax
from utils import visualize, tools
# from models import unet_2 as unet
from models import unet

########################################################################################################################


print('Pytorch version: ', torch.__version__)

params.image_size = 'curri'

save_dir = tools.set_save_dir(params)
writer = SummaryWriter(log_dir=save_dir)



########################################################################################################################
########################################################################################################################
# Model
model = getattr(unet, params.network)(n_classes=params.num_classes).to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in model_parameters])
print('\nModel Parameter Numbers: ', parameters)




########################################################################################################################
########################################################################################################################
# Optimizer, lr Scheduler

if params.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), #filter(lambda p: p.requires_grad, model.parameters())
                                lr=params.learning_rate,
                                momentum=params.beta1,
                                weight_decay=params.weight_decay)

elif params.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params.learning_rate,
                                 betas=(params.beta1, params.beta2),
                                 weight_decay=params.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                 T_0=10,
                                                                 T_mult=1)


########################################################################################################################
########################################################################################################################
# Train

print('\nStart Training!\n')

print('\nPhase 1 - 512\n')
# phase 1
params.image_size = 512
params.max_epoch = 30
params.batch_size = 10
train_loader, val_loader = pneumothorax.get_dataloader(params)

trainer.trainer(model, optimizer, scheduler, train_loader, val_loader, params, save_dir, device, writer,
                it=0)

phase_1_it = len(train_loader) * params.max_epoch


print('\nPhase 2 - 1024\n')
# phase 2
params.image_size = 1024
params.resume_epoch = params.max_epoch
params.max_epoch = 100
params.batch_size = 2
train_loader, val_loader = pneumothorax.get_dataloader(params)

trainer.trainer(model, optimizer, scheduler, train_loader, val_loader, params, save_dir, device, writer,
                it=phase_1_it, freeze_bn=True)




