
import numpy as np
from PIL import ImageOps as ImgOps
from PIL import Image


def preproc_arr(img_arr, size):
    resize_img = np.zeros((img_arr.shape[0],) + size)
    for i in range(img_arr.shape[0]):
        resize_img[i] = preproc(img_arr[i], size)

    return resize_img


def preproc(img, size):
    img = Image.fromarray(img)
    img = img.resize(size, Image.BILINEAR)
    return np.array(ImgOps.grayscale(img))