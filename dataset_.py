import json
from functools import partial

import albumentations as albu
import albumentations.pytorch.transforms as albu_torch
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import fractal
from utils import global_rng, read_image, resize_rotate, resize_within

ImgComp = albu.transforms.ImageCompression
ImageFile.LOAD_TRUNCATED_IMAGES = True

with open("./global_params.json", "r") as f:
    parameters = json.load(f)
size = parameters["size"]

DATA_TRANSFORMS = {
    'train': albu.Compose([
        albu.PadIfNeeded(
            min_height=size[0],
            min_width=size[1],
            position=albu.PadIfNeeded.PositionType.CENTER,
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf(
            [
                albu.Affine(
                    rotate=(-15, 15),
                    translate_percent=(0.1, 0.1),
                    scale=None,
                    shear=(-10, 10),
                    mode=cv2.BORDER_CONSTANT,
                    cval=(255, 255, 255),
                    p=1.),
                albu.Perspective(
                    keep_size=True,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=[255, 255, 255],
                    p=1.),
                albu.ElasticTransform(
                    alpha=2.,
                    sigma=10,
                    alpha_affine=0,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.),
                albu.GridDistortion(
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.),
                albu.OpticalDistortion(
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.)
            ],
            p=0.67,
            ),
        albu.OneOf(
            [
                albu.CLAHE(p=1.),
                albu.RandomBrightnessContrast(p=1.),
                albu.RandomGamma(p=1.),
            ],
            p=0.33,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1.),
                albu.GlassBlur(max_delta=1, sigma=0.1, iterations=1, p=1.),
                albu.MotionBlur(blur_limit=(3, 5), p=1.),
            ],
            p=0.33,
        ),
        albu.OneOf(
            [
                albu.RandomToneCurve(p=1.),
                albu.HueSaturationValue(hue_shift_limit=10,
                                        sat_shift_limit=20,
                                        val_shift_limit=20,
                                        p=1.),
                albu.FancyPCA(p=1)
            ],
            p=0.33,
        ),
        albu.OneOf(
            [
                albu.GaussNoise(p=1.),
                albu.ISONoise(p=1.),
                albu.ImageCompression(
                    quality_lower=3,
                    quality_upper=30,
                    compression_type=ImgComp.ImageCompressionType.WEBP,
                    p=1.),
                albu.ImageCompression(
                    quality_lower=10,
                    quality_upper=40,
                    compression_type=ImgComp.ImageCompressionType.JPEG,
                    p=1.)
            ],
            p=0.33,
        ),
        albu.Normalize(0., 1., max_pixel_value=255.)
    ]),
    'val': albu.Compose([
        albu.PadIfNeeded(
            min_height=size[0],
            min_width=size[1],
            position=albu.PadIfNeeded.PositionType.CENTER,
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]),
        albu.Normalize(0., 1., max_pixel_value=255.)
    ]),
}


class AugmentedTrainDataset(data.Dataset):
    def __init__(self, img_paths, mask_paths, size, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size
        self.transform = transform
        self.resize_func_img = partial(
            resize_within, height=size[0], width=size[1],
            return_PIL=False, resampling=Image.Resampling.BILINEAR
        )  # resize to pad
        self.resize_func_mask = partial(
            resize_within, height=size[0], width=size[1],
            return_PIL=False, resampling=Image.Resampling.NEAREST
        )  # resize to pad

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = read_image(self.img_paths[index], return_PIL=True)
        mask = read_image(self.mask_paths[index], return_PIL=True)
        img = self.resize_func_img(img)
        mask = self.resize_func_mask(mask)
        if self.transform:
            ret = self.transform(image=img, mask=mask)
            img = ret["image"]
            mask = ret["mask"]

        ret = albu_torch.ToTensorV2()(image=img, mask=mask)

        return ret["image"], ret["mask"]


class BGAlteringTrainDataset(data.Dataset):
    def __init__(
        self, images, masks, backgrounds, size, nsamples,
        angle_mean=0., angle_std=8., transform=None
    ):
        self.images = images
        self.masks = masks
        self.backgrounds = backgrounds
        self.size = size
        self.nsamples = nsamples
        self.angle_mean = angle_mean
        self.angle_std = angle_std
        self.transform = transform
        self.bg_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomResizedCrop(height=self.size[0], width=self.size[1],
                                   scale=(0.1, 0.6), p=1.),
            albu.OneOf(
                [
                    albu.CLAHE(p=1.),
                    albu.RandomBrightnessContrast(p=1.),
                    albu.RandomGamma(p=1.),
                ],
                p=0.33,
            ),
            albu.OneOf(
                [
                    albu.RandomToneCurve(p=1.),
                    albu.HueSaturationValue(p=1.),
                    albu.FancyPCA(p=1)
                ],
                p=0.33,
            ),
            albu.ElasticTransform(
                alpha=2., sigma=10, alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT101, p=0.67
            )
        ])

        self.bg_gen = fractal.TerrainGenerator()
        self.bg_cmaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
            'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
            'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
            'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
            'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
            'turbo', 'nipy_spectral', 'gist_ncar'
        ]

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        nimages = len(self.images)
        ind = index % nimages
        img = np.asarray(self.images[ind])
        mask = np.asarray(self.masks[ind])

        angle = global_rng.normal(self.angle_mean, self.angle_std)
        lim = global_rng.uniform(0.5, 1.)
        img, mask = resize_rotate(img, mask, angle, self.size,
                                  scale_upperlimit=lim)

        background = self._make_background()
        img, mask = self._put_image_on_background(img, mask, background)

        if self.transform:
            ret = self.transform(image=img, mask=mask)
            img = ret["image"]
            mask = ret["mask"]

        ret = albu_torch.ToTensorV2(transpose_mask=True)(image=img, mask=mask)

        return ret["image"], ret["mask"]

    def _make_background(self):
        val = global_rng.uniform(0., 1.)
        if val < .6:
            background =\
                self.backgrounds[global_rng.choice(len(self.backgrounds))]
            background = np.asarray(background, dtype=np.uint8)
            background = self.bg_transform(image=background)["image"]
        else:
            height = self.bg_gen.generate(size=(64, 64), roughness=0.2)
            cmap = self.bg_cmaps[global_rng.choice(len(self.bg_cmaps))]
            background = fractal.render(
                height, cmap=cmap,
                shading_ellipticity=.3,
                light_entrance_angle=global_rng.choice(360),
                shading_intensity=.3,
                shading_ksize=9
            )
            background = self.bg_transform(image=background)["image"]

        return background

    def _put_image_on_background(self, img, mask, background):
        coords = mask.nonzero()
        upperleft = coords.min(axis=-1)
        lowerright = coords.max(axis=-1)
        positive_size = lowerright - upperleft
        self.size[0]
        range_h = self.size[0] - positive_size[0]
        offset_h = global_rng.choice(range_h)
        range_w = self.size[1] - positive_size[1]
        offset_w = global_rng.choice(range_w)
        translated_mask = np.zeros_like(mask)
        translated_img = np.zeros_like(img)
        minus_h = self.size[0] - offset_h
        minus_w = self.size[1] - offset_w
        translated_mask[offset_h:, :minus_w] =\
            mask[:minus_h, offset_w:]
        translated_img[offset_h:, :minus_w] =\
            img[:minus_h, offset_w:]

        bg_mask = translated_mask == 0
        translated_img[bg_mask] = background[bg_mask]


class TestDatasetMemory(data.Dataset):
    def __init__(self, img_paths, mask_paths, size, transform=None):
        self.size = size
        self.transform = transform

        images = [read_image(img, return_PIL=True) for img in img_paths]
        images_pad = [resize_within(img,
                                    height=size[0],
                                    width=size[1],
                                    return_PIL=False)
                      for img in images]
        self.images_list = images_pad

        masks = [read_image(m, return_PIL=True) for m in mask_paths]
        masks_pad = [resize_within(mask,
                                   height=size[0],
                                   width=size[1],
                                   return_PIL=False)
                     for mask in masks]
        self.masks_list = masks_pad

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img = self.images_list[index]
        mask = self.masks_list[index]
        if self.transform:
            ret = self.transform(image=img, mask=mask)
            img = ret["image"]
            mask = ret["mask"]

        ret = albu_torch.ToTensorV2()(image=img, mask=mask)

        return ret["image"], ret["mask"]
