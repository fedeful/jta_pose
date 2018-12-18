import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import blocks


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# Generator with Residual Block
# copy of pose guided generation


class ResUnetDownBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ResUnetDownBlock, self).__init__()

        self.input_nc = in_nc
        self.out_nc = out_nc

        # Down Block
        self.conv_down = nn.Conv2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1)

        # Residual Block
        self.rb = blocks.ResidualBlock(out_nc)

    def forward(self, x):

        x = self.rb(self.conv_down(x))
        return x


class ResUnetUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResUnetUpBlock, self).__init__()

        self.input_nc = in_ch
        self.out_nc = out_ch

        # Up Block
        # self.conv_rch = nn.Conv2d(input_nc, middle_nc, kernel_size=1, stride=1, padding=0)
        self.conv_up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.interconv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        # Residual Block
        self.rb = blocks.ResidualBlock(out_ch)

    def forward(self, x1, x2):

        x1 = self.conv_up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.interconv(x)
        x = self.rb(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, in_ch, out_ch, ngf, img_size=(320,128)):
        super(ResUnet, self).__init__()

        #########################################
        #           INITIAL BLOCK               #
        #########################################

        self.initial_conv_0 = nn.Conv2d(in_ch, ngf, kernel_size=3, padding=1)  # IMG_SZ
        self.initial_conv_1 = blocks.ResidualBlock(ngf)

        #########################################
        #           DOWN BLOCKS                 #
        #########################################

        self.rdb_0 = ResUnetDownBlock(ngf, ngf * 2)  # IMG_SZ / 2
        self.rdb_1 = ResUnetDownBlock(ngf * 2, ngf * 4)  # IMG_SZ / 4
        self.rdb_2 = ResUnetDownBlock(ngf * 4, ngf * 8)  # IMG_SZ / 8
        self.rdb_3 = ResUnetDownBlock(ngf * 8, ngf * 16)  # IMG_SZ / 16
        self.rdb_4 = ResUnetDownBlock(ngf * 16, ngf * 32)  # IMG_SZ / 32

        #########################################
        #           FULLY CONNECTED             #
        #########################################

        self.convlft_0 = nn.Conv2d(ngf * 32, ngf * 16, kernel_size=3, stride=1, padding=1)
        self.convlft_1 = nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=1)

        self.fc_dim = img_size[0]//32 * img_size[1]//32 * ngf * 8
        self.fcl = nn.Linear(self.fc_dim, 1024)
        self.fc2 = nn.Linear(1024, self.fc_dim)

        self.convrgh_0 = nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=1, padding=1)
        self.convrgh_1 = nn.Conv2d(ngf * 16, ngf * 32, kernel_size=3, stride=1, padding=1)
        self.afcres = blocks.ResidualBlock(ngf * 32)

        #########################################
        #           UP BLOCKS                   #
        #########################################

        self.rub_4 = ResUnetUpBlock(ngf * 32, ngf * 16)  # IMG_SZ / 16
        self.rub_3 = ResUnetUpBlock(ngf * 16, ngf * 8)  # IMG_SZ / 8
        self.rub_2 = ResUnetUpBlock(ngf * 8, ngf * 4)  # IMG_SZ / 4
        self.rub_1 = ResUnetUpBlock(ngf * 4, ngf * 2)  # IMG_SZ / 2
        self.rub_0 = ResUnetUpBlock(ngf * 2, ngf)  # IMG_SZ

        #########################################
        #           FINAL BLOCK                 #
        #########################################

        self.final_conv_0 = nn.Conv2d(ngf, out_ch, kernel_size=1, stride=1, padding=0)  # IMG_SZ

    def forward(self, x):

        x1 = self.initial_conv_1(self.initial_conv_0(x))

        x2 = self.rdb_0(x1)
        x3 = self.rdb_1(x2)
        x4 = self.rdb_2(x3)
        x5 = self.rdb_3(x4)
        x6 = self.rdb_4(x5)

        x6 = self.convlft_1(self.convlft_0(x6))

        tmp_shape = x6.shape
        x6 = x6.view(-1, self.fc_dim)
        if x6.shape[0] != tmp_shape[0]:
            print('Invalid batches size {} != {}'.format(tmp_shape[0], x6.shape[0]))
            raise Exception

        x = self.fcl(x6)
        x = self.fc2(x)
        x = x.view(tmp_shape)
        x = self.afcres(self.convrgh_1(self.convrgh_0(x)))

        x = self.rub_4(x, x5)
        x = self.rub_3(x, x4)
        x = self.rub_2(x, x3)
        x = self.rub_1(x, x2)
        x = self.rub_0(x, x1)

        x = self.final_conv_0(x)
        # x = self.final_conv_1(x)

        return x


class SimpleUnet(nn.Module):
    def __init__(self, channel_in, channel_out, channel_f, residual=0):
        super(SimpleUnet, self).__init__()

        self.ci = channel_in
        self.co = channel_out
        self.f = channel_f
        self.r = residual

        #########################################
        #           INITIAL BLOCK               #
        #########################################

        self.initial_convolution = blocks.SimpleDoubleCo(self.ci, self.f)

        #########################################
        #           DOWN BLOCKS                 #
        #########################################

        self.db_0 = blocks.SimpleUnetDownBlock(self.f, self.f * 2)  # IMG_SZ / 2
        self.db_1 = blocks.SimpleUnetDownBlock(self.f * 2, self.f * 4)  # IMG_SZ / 4
        self.db_2 = blocks.SimpleUnetDownBlock(self.f * 4, self.f * 8)  # IMG_SZ / 8
        self.db_3 = blocks.SimpleUnetDownBlock(self.f * 8, self.f * 8)  # IMG_SZ / 16

        #########################################
        #         RESIDUAL BOTTLENECK           #
        #########################################

        if self.r > 0:
            print('Residaul ON')

            residual_list = []
            for i in range(self.r):
                residual_list.append(blocks.ResidualBlock(self.f * 8))
            self.res_seq = nn.Sequential(*residual_list)
        else:
            print('Residual OFF')

        #########################################
        #           UP BLOCKS                   #
        #########################################

        self.ub_3 = blocks.SimpleUnetUpBlock(self.f * 8, self.f * 8, True)  # IMG_SZ / 8
        self.ub_2 = blocks.SimpleUnetUpBlock(self.f * 8, self.f * 4)  # IMG_SZ / 4
        self.ub_1 = blocks.SimpleUnetUpBlock(self.f * 4, self.f * 2)  # IMG_SZ / 2
        self.ub_0 = blocks.SimpleUnetUpBlock(self.f * 2, self.f)  # IMG_SZ

        #########################################
        #           FINAL BLOCK                 #
        #########################################

        self.final_convolution = nn.Conv2d(self.f, self.co, kernel_size=1, stride=1, padding=0)  # IMG_SZ
        self.final_activation = nn.Tanh()

    def forward(self, x):

        x1 = self.initial_convolution(x)

        x2 = self.db_0(x1)
        x3 = self.db_1(x2)
        x4 = self.db_2(x3)
        x = self.db_3(x4)
        if self.r > 0:
            x = self.res_seq(x)
        x = self.ub_3(x, x4)
        x = self.ub_2(x, x3)
        x = self.ub_1(x, x2)
        x = self.ub_0(x, x1)

        x = self.final_convolution(x)
        x = self.final_activation(x)
        return x
