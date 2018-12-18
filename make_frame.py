import os
import numpy as np
import pickle
import cv2
from jta.video_generator import JtaVideoContainer
from utils.data_utils import imread_np
from body.pose import Pose, Joint
from torchvision import transforms
import torch
from utils import tensor_utils as tu
from jta import joints_utils as ju
from torch.autograd import Variable


class DataManager(object):
    def __init__(self, data_manager, people=None):
        self.KIND = 'CPM'
        self.NJOINT = 18

        self.pm = data_manager
        self.train_data_path = self.pm.market_data_train
        self.test_data_path = self.pm.market_data_test

        self.all_pose = {}
        with open(self.pm.market_train_pose, 'rb') as fn:
            self.all_pose.update(pickle.load(fn))
        with open(self.pm.market_test_pose, 'rb') as fn:
            self.all_pose.update(pickle.load(fn))

        self.all_data = self.generate_data_elem()
        self.data_perm = np.random.permutation(len(self.all_data))

        self.image_transform = transforms.Compose([transforms.ToTensor()])

        if people is None:
            self.association = {}
            self.current_len = 0
        elif len(people) == 0:
            self.association = {}
            self.current_len = 0
        else:
            self.current_len = len(people)
            self.association = {a[0]: [b, np.random.permutation(len(self.all_data[b]))[0]] for a, b in zip(people, self.data_perm)}

    def generate_data_elem(self):
        all_elem = os.listdir(self.train_data_path) + os.listdir(self.test_data_path)
        subcouple = {e: [] for e in set([el.split('_')[0] for el in all_elem])}
        for el in all_elem:
            subcouple[el.split('_')[0]].append(el)
        del subcouple['0000']
        del subcouple['-1']
        flist = [a[1] for a in subcouple.items()]
        return flist

    def update_people_association(self, new_people):
        for person in new_people:
            if person[0] not in self.association.keys():
                tmp_value = self.data_perm[self.current_len]
                self.association[person[0]] = [tmp_value, np.random.permutation(len(self.all_data[tmp_value]))[0]]
                self.current_len += 1

    def __len__(self):
        return len(self.association)

    def __getitem__(self, item):
        filename = self.all_data[self.association[item][0]][self.association[item][1]]

        pose = Pose(self.get_joints(self.all_pose[filename]))

        if os.path.exists(os.path.join(self.train_data_path, filename)):
            img = self.image_transform(imread_np(os.path.join(self.train_data_path, filename)))
        else:
            img = self.image_transform(imread_np(os.path.join(self.test_data_path, filename)))

        img = tu.from_01_to_11(img)
        hmap = pose.get_heatmap(img.shape[1], img.shape[2], 0, 0, 0, 0)
        hmap = self.image_transform(np.array(hmap).transpose(1, 2, 0).astype(np.float32))
        mask = pose.get_mask(img.shape[1], img.shape[2], 0, 0, 0, 0, 12, 4)
        mask[mask != 0] = 255
        mask = self.image_transform(np.expand_dims(mask, 2))
        limbs = self.image_transform(pose.get_limb_affine_matrix(img.shape[1], img.shape[2], 0, 0, 0, 0))

        return img, hmap, mask, limbs

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


def dest_pose(coordinates, KIND, image_transform, shape=(128, 64)):
    joints = []

    for key in coordinates.keys():
        jc = coordinates[key]
        j = Joint(
            x=jc[0], y=jc[1], jtype=ju.CPM_DICTIONARY[key], occluded=jc[2], self_occluded=jc[3], frame=0,
            confidence=1, person_id=None, cam_dist=-1, kind=KIND
        )
        joints.append(j)

    pose = Pose(joints)

    hmap = pose.get_heatmap2(shape[0], shape[1], 0, 0, 0, 0)
    hmap = image_transform(np.array(hmap).transpose(1, 2, 0).astype(np.float32))
    mask = pose.get_mask(shape[0], shape[1], 0, 0, 0, 0, 12, 4)
    mask[mask != 0] = 255
    mask = image_transform(np.expand_dims(mask, 2))

    return hmap, mask


def generate_sequence(annotation, output_folder, background, fm, generator, maskerator, debu=False):

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder, 0o777)
    if not os.path.isdir('/tmp/jta'):
        os.mkdir('/tmp/jta', 0o777)

    jvc = JtaVideoContainer(annotation, output_folder, background)
    all_frames = jvc.get_frames()
    dm = DataManager(fm)

    imtra = transforms.Compose([transforms.ToTensor()])
    count = 0
    for frame in all_frames:
        count += 1

        people_data = jvc.frame_pose(frame)
        dm.update_people_association(people_data)
        if not os.path.isdir('/tmp/jta/{}'.format(frame)):
            os.mkdir('/tmp/jta/{}'.format(frame), 0o777)

        for person in people_data:
            sample_1 = dm[person[0]]
            sample_2 = dest_pose(person[1], 'CPM', imtra)

            with torch.no_grad():
                data_in = Variable(sample_1[0]).unsqueeze(0).cuda()
                data_in_2 = Variable(sample_2[0]).unsqueeze(0).cuda()
                data_out = generator(torch.cat([data_in, data_in_2], 1))
                data_out_2 = maskerator(data_in_2)

            if debu:
                img_1 = cv2.cvtColor((tu.from_11_to_01(sample_1[0].numpy().transpose(1, 2, 0)) * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR)
                cv2.imshow('persona_1', cv2.resize(img_1, (64, 128)))
                cv2.imshow('mask_2', sample_2[1].numpy().squeeze() * 255)
                cv2.imshow('joints_2', (torch.sum(sample_2[0], 0) * 255).numpy().astype(np.uint8))
                iimg = (cv2.cvtColor(
                    (tu.from_11_to_01(data_out.data.cpu().squeeze().numpy().transpose(1, 2, 0)[..., :3]) * 255).astype(
                        np.uint8),
                    cv2.COLOR_RGB2BGR))
                cv2.imshow('persona_2', iimg)
                iimg2 = ((tu.from_11_to_01(data_out_2.data.cpu().squeeze().numpy()) * 255).astype(
                         np.uint8))
                cv2.imshow('maskout', iimg2)

                cv2.waitKey(0)

            cv2.waitKey(0)
            # invoke network
            with open('/tmp/jta/{}/{}.pk'.format(frame, person[0]), 'wb') as fn:
                tmp_obj = list()
                tmp_obj.append(cv2.cvtColor((tu.from_11_to_01(data_out.data.cpu().squeeze().numpy().transpose(1, 2, 0)) * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
                # mmm = (tu.from_11_to_01(data_out_2.data.cpu().squeeze().numpy()))
                # mmm[mmm >= 0.25] = 1
                # mmm[mmm < 0.25] = 0
                # tmp_obj.append(mmm)
                tmp_obj.append(sample_2[1].numpy().squeeze())
                pickle.dump(tmp_obj, fn)

        jvc.save_frame(frame, people_data)


def make_video(in_path, out_path, type_res):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(out_path), fourcc, 30.0, type_res)
    files_name = [int(file_name.split('.')[0]) for file_name in os.listdir(in_path)]
    files_name.sort()
    for single_name in files_name:
        image = cv2.imread(os.path.join(in_path, '{}.jpg'.format(single_name)))
        out.write(cv2.resize(image, type_res))

    out.release()
    cv2.destroyAllWindows()






