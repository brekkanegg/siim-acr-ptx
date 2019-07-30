import torch
from torchvision.transforms import functional as tff

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
import pickle
import time

from utils import tools
from inputs import augmentations

import cv2




DATA_DIR = '/home/minki/chestxray/data/chestxray14'
print('\nImages 수 : ', len(glob.glob(DATA_DIR + '/*.png')))



def filter_none_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x is not None, batch))
    out = torch.utils.data.dataloader.default_collate(batch)

    return out


# todo: weighted sampling

class my_dataset(Dataset):
    def __init__(self, valset=0, num_classes=1, image_size=1024, transform=None,
                 debug_mode=False, is_train=True, use=True):


        self.valset = valset
        self.num_classes = num_classes
        self.image_size = image_size

        self.is_train = is_train
        self.debug_mode = debug_mode

        self.train_img_dirs, self.val_img_dirs = self.get_train_val_image_dirs(valset)
        if is_train:
            print('\nTrain')
            self.img_dirs = self.train_img_dirs
        else:
            print('\nValidation')
            self.img_dirs = self.val_img_dirs

        if self.debug_mode:
            self.img_dirs = self.img_dirs[:100]



        print('Imgs Num: ', len(self.img_dirs))

        # Augmentation
        self.transform = transform


    def get_train_val_image_dirs(self, valset):
        img_dirs = sorted(glob.glob(DATA_DIR + '/*.png'))

        vlen = len(img_dirs) // 5
        valset_dict = {k: img_dirs[k * vlen:(k + 1) * vlen] for k in range(5)}

        train_img_dirs = []
        for i in [x for x in range(5) if x != valset]:
            train_img_dirs += valset_dict[i]
        val_img_dirs = valset_dict[valset][:1000]

        return train_img_dirs, val_img_dirs


    def __getitem__(self, index):
        t1 = time.time()

        gfp = self.img_dirs[index]
        gray = cv2.imread(gfp, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((1024, 1024), dtype=np.uint8)


        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC) / 255

        if self.transform is not None:
            gray, _ = self.transform(gray, mask)  # caution: 'L' -> 8bit
        else:
            gray, _ = tff.to_tensor(gray).permute(1, 2, 0), tff.to_tensor(mask).permute(1, 2, 0)

        gray = (gray - torch.mean(gray)) / torch.max(torch.tensor(0.1), torch.std(gray))


        return gray


    def __len__(self):
        return len(self.img_dirs)





def get_dataloader(params):

    aug = augmentations.torch_augmentations(hflip=(params.hflip, 0.1),
                                            rotate=(params.rotate, 0.4, 4),
                                            scale=(params.scale, 0.4, 0.04),
                                            translate=(params.translate, 0.4, 0.04),
                                            contrast=(params.contrast, 0.4, 0.1),
                                            gamma_contrast=(params.gamma_contrast, 0.4, 0.1),
                                            gaussian_blur=(params.gaussian_blur, 0.4, 2),
                                            )
    transform = aug.augment

    train_loader = DataLoader(dataset=my_dataset(valset= params.valset,
                                                 num_classes=params.num_classes,
                                                 image_size=params.image_size,
                                                 transform=transform,
                                                 debug_mode=params.debug_mode,
                                                 is_train=True,
                                                 use=params.use
                                                ),
                              batch_size=params.batch_size,
                              shuffle=True,
                              num_workers=params.num_workers,
                              collate_fn=filter_none_collate)

    val_loader = DataLoader(dataset=my_dataset(valset= params.valset,
                                               num_classes=params.num_classes,
                                               image_size=params.image_size,
                                               transform=None,
                                               debug_mode=params.debug_mode,
                                               is_train=False,
                                               use=params.use
                                               ),
                            batch_size=params.batch_size,
                            shuffle=(params.is_train == False),
                            num_workers=params.num_workers,
                            collate_fn=filter_none_collate)


    return train_loader, val_loader

