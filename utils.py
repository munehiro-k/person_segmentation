import os
import random
import sys
from enum import IntEnum

import cv2
import numpy as np
from PIL import Image


class MaskVal(IntEnum):
    BACKGROUND = 0
    KEY_TIP = 1
    KEY_CENTER = 2
    KEY_BODY = 3


global_rng = np.random.default_rng()


def read_image(img_path, return_PIL=False):
    img = Image.open(img_path)
    if img.mode == "L":
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img.load()  # needed for split()
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    elif img.mode == "1" or img.mode == "P":  # for masks
        pass
    elif img.mode != "RGB":
        print(f"warning: unknown image mode {img.mode} in {img_path}")

    if return_PIL:
        return img
    else:
        return np.asarray(img, dtype=np.uint8)


def load_images(df, background_dir=None):
    image_paths = [f"{d}/{b}.jpg" for d, b
                   in zip(df["image_dir"], df["filename_base"])]
    mask_paths = [f"{d}/{b}.png" for d, b
                  in zip(df["mask_dir"], df["filename_base"])]
    if background_dir is not None and not os.path.isdir(background_dir):
        print(f"background directory {background_dir} does not exist")
        sys.exit(-1)

    if background_dir is not None:
        background_paths = [f"{background_dir}/{img}"
                            for img in os.listdir(background_dir)]
        background_paths = [path for path in background_paths
                            if os.path.isfile(path)]
        background_paths.sort()

    if len(image_paths) != len(mask_paths):
        print("the numbers of images and masks do not match")
        sys.exit(-1)

    for img, mask in zip(image_paths, mask_paths):
        if not os.path.isfile(img):
            print("image file {img} does not exist")
            sys.exit(-1)

        if not os.path.isfile(mask):
            print("mask file {mask} does not exist")
            sys.exit(-1)

        img_base = os.path.splitext(os.path.basename(img))[0]
        mask_base = os.path.splitext(os.path.basename(mask))[0]
        if img_base != mask_base:
            print(f"warning: filenames do not match with {img} and {mask}")

    if background_dir is not None:
        backgrounds = [read_image(background_path, return_PIL=True)
                       for background_path in background_paths]
    else:
        backgrounds = None

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if img.size != mask.size:
            print(f"warning: image shapes do not match with {img} and {mask}")

    return image_paths, mask_paths, backgrounds


def resize_ignoring_aspect_ratio(image, width, height, return_PIL=False):
    if type(image).__name__ == 'ndarray':
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    pil_image = pil_image.resize((width, height), Image.Resampling.BICUBIC)
    if return_PIL:
        return pil_image
    else:
        return np.asarray(pil_image, dtype=np.uint8)


def resize_within(image, width, height,
                  return_PIL=False, resampling=Image.Resampling.BICUBIC):
    if type(image).__name__ == 'ndarray':
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    w, h = pil_image.size
    w_scale = width / w
    h_scale = height / h
    if w_scale < h_scale:
        size = (width, int(round(h * w_scale)))
    else:
        size = (int(round(w * h_scale)), height)

    pil_image = pil_image.resize(size, resampling)
    if return_PIL:
        return pil_image
    else:
        return np.asarray(pil_image, dtype=np.uint8)


def resize_rotate(image, mask, angle,
                  result_shape=None, scale_upperlimit=1.):
    height, width = mask.shape[:2]
    image_size = np.array((width, height), dtype=np.int64)
    image_center = image_size / 2

    rad_angle = np.radians(angle)
    abs_cos = abs(np.cos(rad_angle))
    abs_sin = abs(np.sin(rad_angle))

    # determine maximum scale ratio including nonzero region of mask
    bound_w = height * abs_sin + width * abs_cos
    bound_h = height * abs_cos + width * abs_sin
    rotated_size = np.ceil(np.array((bound_w, bound_h))).astype(np.int64)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    rotated_center = rotated_size / 2
    rotated_origin_diff = image_center - rotated_center
    rotation_mat[:, 2] -= rotated_origin_diff

    rotated_mat = cv2.warpAffine(
        mask, rotation_mat, rotated_size,
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    coords = rotated_mat.nonzero()
    upperleft = coords.min(axis=-1)
    lowerright = coords.max(axis=-1)
    box_h, box_w = lowerright - upperleft
    y, x = upperleft
    bb_upperleft_offset = np.array((x, y)) - rotated_center
    box_center = np.array((box_w, box_h)) / 2.

    if result_shape is not None:
        out_scale = min(result_shape[0] / height, result_shape[1] / width)
        expanded_size = np.array(result_shape, dtype=np.float64)
        expanded_size /= out_scale
        scale = np.min((expanded_size[0] / box_h,
                        expanded_size[1] / box_w,
                        scale_upperlimit))
        scale *= out_scale
        out_size = np.array((result_shape[1], result_shape[0]))
        out_center = out_size / 2.
        out_origin_diff = image_center - out_center
    else:
        out_scale = 1.
        scale = np.min((width / box_w, height / box_h, scale_upperlimit))
        out_size = image_size
        out_origin_diff = np.zeros((2,), dtype=np.float64)

    # rotate and scale mask and image
    translated_origin_diff = scale * (bb_upperleft_offset + box_center)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    rotation_mat[:, 2] -= translated_origin_diff + out_origin_diff

    image_result = cv2.warpAffine(
        image, rotation_mat, out_size, flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    mask_result = cv2.warpAffine(
        mask, rotation_mat, out_size, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    return image_result, mask_result
