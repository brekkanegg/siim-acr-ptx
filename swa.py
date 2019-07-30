import matplotlib
matplotlib.use('Agg')  # tensorboardX

import os
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tensorboardX import SummaryWriter

from params import params
device = torch.device('cuda:0')
torch.cuda.set_device(0)

from collections import OrderedDict as odict


from inputs import pneumothorax
from utils import visualize, tools
from models import unet


from script import tester


print('Pytorch version: ', torch.__version__)

save_dir = tools.set_save_dir(params)
writer = SummaryWriter(log_dir=save_dir)




#######
# assert params.seed == 1906271601

train_loader, val_loader = pneumothorax.get_dataloader(params)



##########################

tot_val_info_df = []


# obtain average of swa results
swa_state_dict_dict = {}

val_segs = []
val_fps = []
val_zs = []


print('\nSWA')

input_string = input("Enter a list of swa epoch separated by space: ")
swa_epochs  = input_string.split()

# swa_epochs = [82, 29, 41, 48, 56]
for swa_epoch in (swa_epochs):

    swa_model_dir = os.path.join(save_dir, 'epoch_{}.pth.tar'.format(swa_epoch))
    print('SWA model dir:', swa_model_dir)

    swa_state_dict = torch.load(swa_model_dir)
    swa_state_dict_dict[swa_epoch] = swa_state_dict

avg_weight_dict = odict()
for k in swa_state_dict_dict[swa_epochs[0]].keys():
    avg_weight_dict[k] = torch.mean(torch.stack([swa_state_dict_dict[se][k].float() for se in swa_epochs]), dim=0)

swa_state_dict_dict.clear()

model = getattr(unet, params.network)(n_channels=params.num_channels, n_classes=params.num_classes,
                  down_ratio=int(params.input_size/params.output_size)).to(device)
model.load_state_dict(avg_weight_dict)

# bn update
model.train()
for i, (fp, img, seg, lbl) in tqdm(enumerate(train_loader)):
    img = img.permute(0, 3, 1, 2)  # NXYZ1 -> N1XYZ
    img = img.to(device, dtype=torch.float)
    seg = seg.to(device, dtype=torch.long)
    outputs = model(img)
    if i == 1000:
        break

tester.tester(model, val_loader, save_dir, device, writer, best_epoch='swa')


