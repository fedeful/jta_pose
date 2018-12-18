import os
import cv2
import torch
import pickle
import numpy as np
import itertools
import torch.nn as nn
import torchvision
from body.pose import Pose, Joint
from torch.utils.data import Dataset
from utils.path_config import PathMng
from torchvision import transforms
from utils import tensor_utils as tu
from utils.data_utils import imread_np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


class MarketDataset(Dataset):

    def __init__(self, train, conf_path, resize=None, dualpose=False, mask=False, affine=False, affine_matrix=False,
                 complete=False):

        self.KIND = 'CPM'
        self.NJOINT = 18

        self.train = train

        if self.train:
            self.image_path = conf_path.market_data_train
            self.image_path_2 = conf_path.market_data_test
            self.info_path_pose = conf_path.market_train_pose
            self.info_path_couple = conf_path.market_train_list
        else:
            self.image_path = conf_path.market_data_test
            self.info_path_pose = conf_path.market_test_pose
            self.info_path_couple = conf_path.market_test_list

        self.mask = mask
        self.train = train
        self.affine = affine
        self.affine_matrix = affine_matrix
        self.dualpose = dualpose
        self.complete = complete

        self.elements_couple = []
        self.elements_pose = {}

        with open(self.info_path_couple, 'rb') as fn:
            self.elements_list = pickle.load(fn)
        with open(self.info_path_pose, 'rb') as fn:
            self.elements_pose = pickle.load(fn)

        self.resize = resize
        self.tt = transforms.Compose([transforms.ToTensor()])
        if self.resize is None:
            self.image_transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.image_transform = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize(self.resize),
                                                       transforms.ToTensor()])

    def __len__(self):
        return len(self.elements_list)

    def __getitem__(self, index):
        return self.generate_data(self.elements_list[index])

    def generate_data(self, info):

        pose_1 = Pose(self.get_joints(self.elements_pose[info[0]]))
        pose_2 = Pose(self.get_joints(self.elements_pose[info[1]]))

        if os.path.exists(os.path.join(self.image_path, info[0])):
            img_1 = self.image_transform(imread_np(os.path.join(self.image_path, info[0])))
        else:
            img_1 = self.image_transform(imread_np(os.path.join(self.image_path_2, info[0])))

        if os.path.exists(os.path.join(self.image_path, info[1])):
            img_2 = self.image_transform(imread_np(os.path.join(self.image_path, info[1])))
        else:
            img_2 = self.image_transform(imread_np(os.path.join(self.image_path_2, info[1])))

        hmap_1 = pose_1.get_heatmap(img_1.shape[1], img_1.shape[2], 0, 0, 0, 0)
        hmap_2 = pose_2.get_heatmap(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
        hmap_1 = self.tt(np.array(hmap_1).transpose(1, 2, 0).astype(np.float32))
        hmap_2 = self.tt(np.array(hmap_2).transpose(1, 2, 0).astype(np.float32))

        if self.dualpose:
            img_in = torch.cat([tu.from_01_to_11(img_1), hmap_1, hmap_2], 0)
        else:
            img_in = torch.cat([tu.from_01_to_11(img_1), hmap_2], 0)

        img_out = tu.from_01_to_11(img_2)

        if self.mask:
            mask_2 = pose_2.get_mask(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0, 12, 4)
            mask_2[mask_2 != 0] = 255
            mask_2 = self.tt(np.expand_dims(mask_2, 2))

            mask_1 = pose_1.get_mask(img_1.shape[1], img_1.shape[2], 0, 0, 0, 0, 12, 4)
            mask_1[mask_1 != 0] = 255
            mask_1 = self.tt(np.expand_dims(mask_1, 2))

        else:
            mask_2 = torch.zeros(img_2.size())
            mask_1 = torch.zeros(img_1.size())

        #added now
        if self.affine:
            p2 = pose_2.get_limb_affine(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
            p1 = pose_1.get_limb_affine(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
            return img_in, img_out, mask_2, [p1, p2]
        if self.affine_matrix:
            lam = pose_1.get_limb_affine_matrix(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
            lam = self.tt(lam)
            return img_in, img_out, mask_2, lam
        if self.complete:
            # p2 = pose_2.get_limb_affine(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
            # p1 = pose_1.get_limb_affine(img_2.shape[1], img_2.shape[2], 0, 0, 0, 0)
            p1 = 0
            p2 = 0
            return img_in[:3], img_out, hmap_1, hmap_2, mask_1, mask_2, [p1, p2]
        else:
            return img_in, img_out, mask_2

    def get_joints(self, coordinates):
        joints = []

        for i in range(self.NJOINT):
            jc = coordinates[i]
            j = Joint(
                x=jc[2], y=jc[1], jtype=jc[0], occluded=0, self_occluded=0, frame=0,
                confidence=jc[3], person_id=None, cam_dist=-1, kind=self.KIND
            )
            joints.append(j)

        return joints


def main():
    pmng = PathMng('prova', 'glob')
    ds = MarketDataset(True, pmng, [128, 64], mask=True,affine=True)
    for i in range(500, 800):
        a = ds.__getitem__(i)
        dima = cv2.cvtColor((tu.from_11_to_01(a[1].data.numpy().transpose(1, 2, 0)) * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)
        mm = a[2].data.numpy().transpose(1, 2, 0)
        mm[mm != 0] = 1
        dimasked = dima * mm.astype(np.uint8)
        # cv2.imshow('masked', cv2.resize(dimasked, (256, 512)))
        cv2.imshow('dest_img'
                   '', cv2.resize(dima, (256, 512)))
        cv2.imshow('initial', cv2.resize(
             cv2.cvtColor((tu.from_11_to_01(a[0].data.numpy().transpose(1, 2, 0)) * 255).astype(np.uint8)[:, :, :3],
                          cv2.COLOR_RGB2BGR), (256, 511)))
        # cv2.imshow('joints', (torch.sum(a[0][3:, :, :], 0) * 255).numpy().astype(np.uint8))
        # cv2.imshow('mask', a[2].data.numpy().squeeze())
        #cv2.imshow('skeleton', a[3])
        p1 = a[3][0]['1112']
        p2 = a[3][1]['1112']

        image = Image.new("RGB", (64, 128))
        draw = ImageDraw.Draw(image)
        points = ((p1[0][0], p1[0][1]), (p1[1][0], p1[1][1]), (p1[3][0], p1[3][1]), (p1[2][0], p1[2][1]))
        draw.polygon((points), fill=200)
        image = np.array(image)
        cv2.imshow('input', cv2.resize(image,(256, 511)))

        image = Image.new("RGB", (64, 128))
        draw = ImageDraw.Draw(image)
        points = ((p2[0][0], p2[0][1]), (p2[1][0], p2[1][1]), (p2[3][0], p2[3][1]), (p2[2][0], p2[2][1]))
        draw.polygon((points), fill=200)
        image = np.array(image)
        cv2.imshow('dest', cv2.resize(image,(256, 511)))

        M2 = cv2.getAffineTransform(np.float32(p1)[:3], np.float32(p2)[:3])
        M2 = M2.astype(np.float32)

        llll = M2.copy()
        M2 = torch.unsqueeze(torch.from_numpy(M2), 0)
        M = cv2.getAffineTransform(np.float32(p2)[:3], np.float32(p1)[:3])
        A = M2.clone()
        A[0, 0, 2] = M2[0, 0, 2] / a[0].shape[1]
        A[0, 0, 2] = M2[0, 1, 2] / a[0].shape[2]
        A[0,:,:2] = A[0,:,:2]/ np.pi

        al = [i for i in np.arange(0, a[0].shape[2])]
        bl = [i for i in np.arange(0, a[0].shape[1])]
        all_pixel = list(itertools.product(al, bl))
        pp = np.append(all_pixel, np.expand_dims(np.ones(np.array(all_pixel).shape[0]), 1), axis=1)
        all_pixel_dest = np.dot(M, np.transpose(pp, [1, 0]))

        img3 = torch.zeros(a[0].shape)

        all_pixel_dest = np.transpose(all_pixel_dest.astype(int)[:, :], [1, 0])
        # marker = np.zeros(image.shape[:2], np.uint8)
        for u, l in zip(np.array(all_pixel), all_pixel_dest):
            if 0 <= int(l[1]) < 128 and 0 <= int(l[0]) < 64:
                img3[:, u[1], u[0]] = a[0][:, l[1], l[0]]

        #all_pixel_dest = np.transpose(all_pixel_dest, [1, 0])
        # all_pixel_dest2 = np.reshape(all_pixel_dest, [128, 64, -1])
        # tmp = all_pixel_dest2[..., 0]
        # all_pixel_dest2[..., 0] = all_pixel_dest2[..., 1]
        # all_pixel_dest2[..., 1] = tmp
        # all_pixel_dest2[..., 0] /= 64
        # all_pixel_dest2[..., 1] /= 128
        # all_pixel_dest2 -=0.5
        # all_pixel_dest2 *= 2
        #
        # apd = torch.from_numpy(all_pixel_dest2).unsqueeze(0).float()
        #
        # grid = torch.nn.functional.affine_grid(torch.randn(1, 2, 3), a[0][:3, ...].unsqueeze(0).shape)
        # x_trans = torch.nn.functional.grid_sample(a[0].unsqueeze(0), apd)
        # x_trans = x_trans.squeeze()
        cv2.imshow('trans', cv2.resize(
            cv2.cvtColor((tu.from_11_to_01(img3.data.numpy().transpose(1, 2, 0)) * 255).astype(np.uint8)[:, :, :3],
                         cv2.COLOR_RGB2BGR), (256, 511)))
        #
        # M_n = np.zeros(M.shape)
        # M_n[0, :] = M[1, :]
        # M_n[1, :] = M[0, :]
        #

        cv2.waitKey(0)

    torchvision.utils.save_image(torch.sum((a[0].data.cpu()[3:, ...]), 0), 'prova.png')
    print('sono qui ')

def main2():
    pmng = PathMng('prova', 'glob')
    ds = MarketDataset(True, pmng, [128, 64], mask=True,affine_matrix=True)
    for i in range(500, 800):
        a = ds.__getitem__(i)
        dima = cv2.cvtColor((tu.from_11_to_01(a[1].data.numpy().transpose(1, 2, 0)) * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)
        mm = a[2].data.numpy().transpose(1, 2, 0)
        mm[mm != 0] = 1
        dimasked = dima * mm.astype(np.uint8)
        cv2.imshow('masked', cv2.resize(dimasked, (256, 512)))
        cv2.imshow('dest_img'
                   '', cv2.resize(dima, (256, 512)))
        cv2.imshow('initial', cv2.resize(
             cv2.cvtColor((tu.from_11_to_01(a[0].data.numpy().transpose(1, 2, 0)) * 255).astype(np.uint8)[:, :, :3],
                          cv2.COLOR_RGB2BGR), (256, 511)))
        # cv2.imshow('joints', (torch.sum(a[0][3:, :, :], 0) * 255).numpy().astype(np.uint8))
        # cv2.imshow('mask', a[2].data.numpy().squeeze())
        cv2.imshow('skeleton', a[3].data.numpy().transpose(1, 2, 0)[..., 1].astype(np.uint8) *255)

        cv2.waitKey(0)

    torchvision.utils.save_image(torch.sum((a[0].data.cpu()[3:, ...]), 0), 'prova.png')
    print('sono qui ')

if __name__ == '__main__':
    main2()
