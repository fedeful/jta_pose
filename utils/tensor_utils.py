import torch
import numpy as np
import cv2
import math
from torch.autograd import Variable


def weights_init(m, bn=False):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and bn:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def unorm_print(batch):
    batch = torch.div(batch, 2.0)
    batch = torch.add(batch, 0.5)
    return batch


def normalize_batch(batch):

    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 2.0)
    batch = torch.add(batch, 0.5)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch


def tensor2image(tensor):
    image = 127.5*(tensor.cpu().float().numpy() + 1)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def from_01_to_11(tensor):
    tensor = (tensor - 0.5) * 2
    return tensor


def from_11_to_01(tensor):
    tensor = (tensor / 2) + 0.5
    return tensor


def tensor2numpy(tensor):
    new_tensor = 127.5*(tensor.cpu().float().numpy() + 1)
    if new_tensor.shape[0] == 1:
        new_tensor = np.tile(new_tensor, (3, 1, 1))
    return new_tensor.astype(np.uint8)


def numpy_to_pytorch(tensor, cuda=False, rngzo=True):
    nt = Variable(torch.from_numpy(tensor.transpose(2, 0, 1)).float())
    if cuda:
        nt = nt.cuda()
    if rngzo:
        nt /= 255
    return nt.unsqueeze(0)


def combine_images(imgs, path):
    if type(imgs) is Variable:
        imgs = imgs.data.cpu().numpy()
    else:
        imgs = imgs.numpy()
    num = imgs.shape[0]
    channels = imgs.shape[1]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = imgs.shape[2:]
    image = np.zeros((channels, height * shape[0], width * shape[1]), dtype=imgs.dtype)
    for index, img in enumerate(imgs):
        i = int(index / width)
        j = index % width
        for c in range(channels):
            image[c, i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[c, :, :]

    if channels == 1:
        image = (image * 117.5) + 117.5
        image = np.squeeze(image)
    elif channels == 3:
        #image = (image * 127.5) + 127.5
        image = (image * 255)
        image = image.transpose((1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = image.astype(np.uint8)
    cv2.imwrite(path, image)



