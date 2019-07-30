import numpy as np
import random

import scipy.ndimage
import skimage.exposure
import time
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch

# todo: contrast, brightness, sharpness, #solarize, #posterize, #cutout
import warnings
warnings.filterwarnings('ignore', '.*output shape of scale.*')


from torchvision import transforms
from torchvision.transforms import functional as tff


import PIL

# scipy.ndimage
# The order of the spline interpolation, default is 3. The order has to be in the range 0-5.


# todo: add more augmentation - contrast types
# gaussian noise, contrast

class torch_augmentations():  # apply for each intersection image

    def __init__(self,
                 hflip=(False, 0.1),
                 vflip=(False, 0.1),
                 rotate=(True, 0.5, 5),
                 scale=(True, 0.5, 0.03),
                 translate=(True, 0.5, 0.03),
                 elastic=(False, 0.5, 2, 0.08, 0.08),
                 contrast=(True, 0.5, 0.1),
                 gamma_contrast=(True, 0.5, 0.1),
                 gaussian_blur=(True, 0.5, 2)


                 ):

        self.hflip_params = hflip
        self.vflip_params = vflip
        self.rotate_params = rotate
        self.scale_params = scale
        self.translate_params = translate
        self.contrast_params = contrast
        self.gamma_contrast_params = gamma_contrast
        self.gaussian_blur_params = gaussian_blur
        # self.elastic_transform_params = elastic



    def augment(self, img, seg):
        # print(1)
        img, seg = self.to_pil(img, seg)

        img, seg = self.hflip(img, seg, *self.hflip_params)
        # img, seg = self.vflip(img, seg, *self.vflip_params)
        img, seg = self.rotate(img, seg, *self.rotate_params)
        img, seg = self.scale(img, seg, *self.scale_params)
        img, seg = self.translate(img, seg, *self.translate_params)
        # img, seg = self.elastic_transform(img, seg, *self.elastic_transform_params)

        img = self.adjust_contrast(img, *self.contrast_params)
        img = self.adjust_gamma(img, *self.gamma_contrast_params)
        img = self.gaussian_blur(img, *self.gaussian_blur_params)


        aimg, aseg = self.pil_2_tensor(img, seg)

        return aimg, aseg


    def to_pil(self, img, seg):

        pil_img = tff.to_pil_image(img, 'L')
        pil_seg = tff.to_pil_image(seg.astype('float32'), 'F')


        return pil_img, pil_seg


    def pil_2_tensor(self, img, seg):
        img = tff.to_tensor(img).permute(1, 2, 0)
        seg = tff.to_tensor(seg).permute(1, 2, 0)

        return img, seg

    def hflip(self, img, seg, do, prob):
        if do:
            if random.random() < prob:
                img = tff.hflip(img)
                seg = tff.hflip(seg)

        return img, seg


    def vflip(self, img, seg, do, prob):
        if do:
            if random.random() < prob:
                img = tff.vflip(img)
                seg = tff.vflip(seg)

        return img, seg


    def rotate(self, img, seg, do, prob, rot_angle):
        if do:
            if random.random() < prob:
                rot_angle = random.uniform(-rot_angle, rot_angle)  # -7,7
                img = tff.rotate(img, rot_angle, resample=PIL.Image.BICUBIC, expand=False)
                seg = tff.rotate(seg, rot_angle, resample=PIL.Image.BICUBIC, expand=False)

        return img, seg


    def scale(self, img, seg, do, prob, scale_factor):
        img_size = img.size[0]
        seg_size = seg.size[0]

        if do:
            if random.random() < prob:
                scale_ratio = random.uniform(1, 1+scale_factor)
                img_cs = int(img_size/scale_ratio)
                img_x1, img_y1 = (img_size-img_cs) // 2, (img_size-img_cs) // 2

                seg_cs = int(seg_size/scale_ratio)
                seg_x1, seg_y1 = (seg_size-seg_cs) // 2, (seg_size-seg_cs) // 2

                img = tff.resized_crop(img, i=img_x1, j=img_y1, h=img_cs, w=img_cs, size=img_size,
                                       interpolation=PIL.Image.BICUBIC)

                seg = tff.resized_crop(seg, i=seg_x1, j=seg_y1, h=seg_cs, w=seg_cs, size=seg_size,
                                             interpolation=PIL.Image.BICUBIC)

        return img, seg

    def translate(self, img, seg, do, prob, translate_factor):
        img_size = img.size[0]
        seg_size = seg.size[0]

        img_seg_ratio = img_size//seg_size


        if do:
            if random.random() < prob:
                translate_factor_x = random.uniform(-translate_factor, translate_factor)
                translate_factor_y = random.uniform(-translate_factor, translate_factor)
                img_tx, img_ty = int(img_size * translate_factor_x), int(img_size * translate_factor_y)
                seg_tx, seg_ty = int(img_tx / img_seg_ratio), int(img_ty / img_seg_ratio)

                img_x1, img_y1 = int(img_size*translate_factor) + img_tx, int(img_size*translate_factor) + img_ty
                seg_x1, seg_y1 = int(seg_size*translate_factor) + seg_tx, int(seg_size*translate_factor) + seg_ty


                img_pad = tff.pad(img, padding=int(img_size*translate_factor), fill=0, padding_mode='constant')
                img = tff.crop(img_pad, img_x1, img_y1, img_size, img_size)

                seg_pad = tff.pad(seg, padding=int(seg_size*translate_factor), fill=0, padding_mode='constant')
                seg = tff.crop(seg_pad, seg_x1, seg_y1, seg_size, seg_size)

        return img, seg


    def adjust_contrast(self, img, do, prob, cf):
        if do:
            if random.random() < prob:
                c = random.uniform(1-cf, 1+cf)  # -7,7
                img = tff.adjust_contrast(img, contrast_factor=c)

        return img



    def adjust_gamma(self, img, do, prob, g):
        if do:
            if random.random() < prob:
                gm = random.uniform(1-g, 1+g)  # -7,7
                img = tff.adjust_gamma(img, gamma=gm)

        return img



    def gaussian_blur(self, img, do, prob, radius):
        if do:
            if random.random() < prob:
                r = random.uniform(0, radius)
                img = img.filter(PIL.ImageFilter.GaussianBlur(r))

        return img




