"""Yol maskesinden tek piksel kalinliginda iskelet (skeleton) cikarimi."""
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


def clean_mask(mask, min_object_size=64, closing_kernel=3):
    binary = mask > 127
    if closing_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    if min_object_size > 0:
        binary = remove_small_objects(binary, max_size=min_object_size)
    return binary


def mask_to_skeleton(mask, min_object_size=64, closing_kernel=3):
    binary = clean_mask(mask, min_object_size, closing_kernel)
    return skeletonize(binary).astype(np.uint8)
