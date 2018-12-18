import torch
import numpy as np
from torch.autograd import Variable
from utils import tensor_utils as tu


class GanNewImage(object):
    def __init__(self, generator, gpu):
        self.gan = generator
        self.gpu = gpu

        if self.gpu:
            self.gan = self.gan.cuda()

    def feed(self, *arg):

        if len(arg) == 2:

            image = tu.from_01_to_11(torch.from_numpy((arg[0].transpose(0, 2, 3, 1))/255))
            pose = torch.from_numpy(arg[1].transpose(0, 2, 3, 1))

            data_in = Variable(torch.cat([image, pose], 1))
            if self.gpu:
                data_in = data_in.cuda()

            generated_image = self.gan(data_in)
            generated_image = tu.from_11_to_01(generated_image.data.cpu())

            return generated_image



