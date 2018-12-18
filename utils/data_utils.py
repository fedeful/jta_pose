import PIL
import numpy as np


def imread_pil(path):
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def imread_np(path):
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return np.array(img.convert('RGB'))
