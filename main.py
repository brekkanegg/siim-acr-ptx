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

from script import trainer, validator, tester, btrainer
from inputs import pneumothorax
from utils import visualize, tools
# from models import unet_2 as unet
from models import unet

########################################################################################################################


print('Pytorch version: ', torch.__version__)

save_dir = tools.set_save_dir(params)
writer = SummaryWriter(log_dir=save_dir)


########################################################################################################################
# Load Data


train_loader, val_loader = pneumothorax.get_dataloader(params)







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

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
#                                                                  T_0=10,
#                                                                  T_mult=1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=100,
                                                       eta_min=1e-4)


########################################################################################################################
########################################################################################################################

# Inference
if (not params.is_train) or (params.submit):

    if not params.inference_epoch:
        history = pd.read_csv(save_dir + '/training_log.csv')
        print('\nhistory: ', history)
        best_index = np.argmin(history["val_loss"])
        best_dice = 1-np.min(history["val_loss"])
        best_epoch = int(history['epoch'][best_index])
        print('Loading model with dice {:.4f}, epoch {}'.format(best_dice, best_epoch))

    else:
        best_epoch = params.inference_epoch

    best_model_dir = os.path.join(save_dir, 'epoch_{}.pth.tar'.format(best_epoch))
    print('\nbest model dir:', best_model_dir, '\n')

    state_dict = torch.load(best_model_dir)
    model.load_state_dict(state_dict)

    tester.tester(model, val_loader, save_dir, device, writer, best_epoch, submit=params.submit,
                  follow_aux=params.follow_aux, th=params.threshold)



########################################################################################################################
########################################################################################################################
# Train
else:

    if params.resume_epoch:
        print('\nRe-Start Training!\n')
        it = len(train_loader) * params.resume_epoch

        best_model_dir = os.path.join(save_dir, 'epoch_{}.pth.tar'.format(params.resume_epoch))
        print('\nFrom Model Dir:', best_model_dir, '\n')

        state_dict = torch.load(best_model_dir)
        model.load_state_dict(state_dict)


    else:
        print('\nStart Training!\n')
        it = 0

    ######################################################
    # Train the model


    trainer.trainer(model, optimizer, scheduler, train_loader, val_loader, params, save_dir, device, writer,
                    use=params.use, it=it, follow_aux=params.follow_aux, th=params.threshold, accumulation=params.accumulation)

