import numpy as np
from scipy import ndimage


def reduce(img, factor):
    return img.reshape(img.shape[0] // factor, factor, img.shape[1] // factor, factor).mean(axis=(1, 3))


def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False, mode="constant", cval=0)


def flip(img, direction):
    return np.flip(img, axis=direction)


def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast * rotate(flip(img, direction), angle) + brightness
