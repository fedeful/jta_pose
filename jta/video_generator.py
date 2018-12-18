import imageio
import os
import pickle
import json
import cv2
from jta import joints_utils
import numpy as np
from typing import *
from threading import Thread, Semaphore

MAX_X = 1920
MAX_Y = 1080
PAD_X = 0.2
PAD_Y = 0.15

common_sem = Semaphore(4)


def folder_setup(output_directory: str):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    occ = os.path.join(output_directory, 'occ')
    no_occ = os.path.join(output_directory, 'no_occ')

    if not os.path.exists(occ):
        os.makedirs(occ)
    if not os.path.exists(no_occ):
        os.makedirs(no_occ)


def second_type_folder_setup(output_directory: str):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def frame_avg_distance_cam_spine_0(frame_data):
    avg_cam_distance_frame = 0.
    ped_numbers = 0.

    for p_id in frame_data:
        ped_numbers += 1
        pose = frame_data[p_id]

        spine_0 = joints_utils.get_joint(pose=pose, jid=11)
        cam_distance = np.sqrt(spine_0['x3d'] ** 2 + spine_0['y3d'] ** 2 + spine_0['z3d'] ** 2)
        avg_cam_distance_frame += cam_distance

    avg_cam_distance_frame = avg_cam_distance_frame / ped_numbers

    return avg_cam_distance_frame


def check_saving_box(pose: List[Dict], percentage: float = 0.4, min_percentage: float = 0.2) -> int:
    count = 0
    for joint in pose:
        if joint['occ']:
            count += 1

    count = count / joints_utils.JOINTS

    if count > percentage:
        return 0
    elif percentage >= count >= min_percentage:
        return 1
    elif min_percentage >= count > 0:
        return 0
    else:
        return 2


def box_color(pose: List[Dict], percentage: float = 0.4, min_percentage: float = 0.2) -> Tuple[int, int, int]:
    count = 0
    for joint in pose:
        if joint['occ']:
            count += 1

    count = count / joints_utils.JOINTS

    if count > percentage:
        return 255, 0, 0
    elif percentage >= count >= min_percentage:
        return 0, 255, 0
    elif min_percentage > count > 0:
        return 255, 0, 0
    else:
        return 0, 0, 255


def pedestran_box(pose: List[Dict], cam_distance: float, avg_cam_distance: float):
    max_x = 0
    min_x = 2000
    max_y = 0
    min_y = 2000

    for joint in pose:

        tmp_joint = joints_utils.joint_pos(joint)

        if tmp_joint[0] < min_x:
            min_x = tmp_joint[0]
        if tmp_joint[0] > max_x:
            max_x = tmp_joint[0]

        if tmp_joint[1] < min_y:
            min_y = tmp_joint[1]
        if tmp_joint[1] > max_y:
            max_y = tmp_joint[1]

    # border = int(round(np.power(10, 1 - (cam_distance / 20.0))))
    # print(border, cam_distance)
    cam_distance = cam_distance if cam_distance != 0 else 1
    # border = int((avg_cam_distance / cam_distance) * 10)
    border_w = int(270 / cam_distance)
    border_h = int(250 / cam_distance)
    # print(border, cam_distance)
    # border = border*8 if border >= 0.8 else int(np.log(cam_distance)*5)

    # print(border, cam_distance)

    return (min_x - border_w, min_y - border_h), (max_x + border_w, max_y + border_h)


def box_consistance(img_shape, borders) -> bool:
    if borders[0][0] < 0 or borders[1][0] > img_shape[0]:
        return False
    if borders[0][1] < 0 or borders[1][1] > img_shape[1]:
        return False
    return True


def crop_and_save_image(image, name, co_1, co_2):
    tmp_img = image[co_1[1]:co_2[1], co_1[0]:co_2[0]]
    imageio.imwrite(name, tmp_img)


def crop_and_save_joint(im_sz, joints, name, name3, pedestrans_joint_list, co_1, co_2):
    joints_img = np.zeros(im_sz, dtype=np.uint8)
    for joint in joints:
        if not joint['occ']:
            cv2.circle(
                joints_img, thickness=-1,
                center=joints_utils.joint_pos(joint),
                radius=joints_utils.joint_radius(joint),
                color=joints_utils.joint_color(joint),
            )

    ped_joints_list = []

    for joint in joints:
        dt = dict()
        dt['jid'] = joint['jid']
        dt['occ'] = joint['occ']
        dt['soc'] = joint['soc']
        dt['x2d'] = joint['x2d'] - co_1[0]
        dt['y2d'] = joint['y2d'] - co_1[1]
        dt['x3d'] = joint['x3d']
        dt['y3d'] = joint['y3d']
        dt['z3d'] = joint['z3d']
        ped_joints_list.append(dt)

    pedestrans_joint_list.append({'name': name3, 'data': ped_joints_list})

    tmp_img = joints_img[co_1[1]:co_2[1], co_1[0]:co_2[0]]
    imageio.imwrite(name, tmp_img)


def append_joints_info(joints, file_name, peds_joints_list, co_1, co_2):
    ped_joints_list = []

    for joint in joints:
        dt = dict()
        dt['jid'] = joint['jid']
        dt['occ'] = joint['occ']
        dt['soc'] = joint['soc']
        dt['x2d'] = joint['x2d'] - co_1[0]
        dt['y2d'] = joint['y2d'] - co_1[1]
        dt['x3d'] = joint['x3d']
        dt['y3d'] = joint['y3d']
        dt['z3d'] = joint['z3d']
        ped_joints_list.append(dt)

    peds_joints_list.append({'name': file_name, 'data': ped_joints_list})


def save_joints_info(file_path, joints_list):
    with open(file_path, 'w') as out_file:
        json.dump(joints_list, out_file)


def check_setup_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)


def append_joints_info_second_type(joints, pid, file_name, peds_joints_dict, co_1, co_2):
    ped_joints_list = []

    for joint in joints:
        dt = dict()
        dt['jid'] = joint['jid']
        dt['occ'] = joint['occ']
        dt['soc'] = joint['soc']
        dt['x2d'] = joint['x2d'] - co_1[0]
        dt['y2d'] = joint['y2d'] - co_1[1]
        dt['x3d'] = joint['x3d']
        dt['y3d'] = joint['y3d']
        dt['z3d'] = joint['z3d']
        ped_joints_list.append(dt)

    if pid in peds_joints_dict:
        peds_joints_dict[pid].append({'name': file_name, 'data': ped_joints_list})
    else:
        peds_joints_dict[pid] = [{'name': file_name, 'data': ped_joints_list}]


class JtaVideoContainer(object):
    def __init__(self, path_annotation, path_output, path_background, use_ram=False):
        self.pa = path_annotation
        self.po = path_output
        self.pb = path_background
        self.ur = use_ram

        self.background = cv2.imread(self.pb)

        self.MAX_X = 1920
        self.MAX_Y = 1080
        self.PAD_X = 0.2
        self.PAD_Y = 0.15

        if self.ur:
            with open(self.pa, 'r') as fo:
                self.frames_annotation = json.load(fo)

        if not os.path.isdir(self.po):
            os.mkdir(self.po, 0o777)

    def people_id(self):
        if self.ur:
            json_seq_info = self.frames_annotation
        else:
            with open(self.pa, 'r') as fo:
                json_seq_info = json.load(fo)

        tmp_list = []
        for key in json_seq_info.keys():
            for person_id in json_seq_info[key].keys():
                if person_id not in tmp_list:
                    tmp_list.append(person_id)

    def frame_pose(self, frame_id, rescaled=(128, 64)):
        if self.ur:
            json_seq_info = self.frames_annotation
        else:
            with open(self.pa, 'r') as fo:
                json_seq_info = json.load(fo)
        jpeople = []
        for person_id in json_seq_info[frame_id]:
            # if person_id == 'person_28':
            #     print('stop')
            # print(frame_id, person_id)
            person_info = json_seq_info[frame_id][person_id]

            if joints_utils.all_occ(person_info):
                continue

            x_min, x_max, y_min, y_max = joints_utils.find_mins_maxs(person_info)

            if x_max >= self.MAX_X or y_max >= self.MAX_Y:
                continue

            width = x_max - x_min
            height = y_max - y_min

            x_min = int(x_min - self.PAD_X * width) if int(x_min - self.PAD_X * width) > 0 else 0
            x_max = int(x_max + self.PAD_X * width) if int(x_max + self.PAD_X * width) < self.MAX_X else self.MAX_X

            y_min = int(y_min - self.PAD_Y * height) if int(y_min - self.PAD_Y * height) > 0 else 0
            y_max = int(y_max + self.PAD_Y * height) if int(y_max + self.PAD_Y * height) < self.MAX_Y else self.MAX_Y

            delta_x = x_max - x_min
            delta_y = y_max - y_min

            joints = joints_utils.from_jta_to_cpm(person_info)
            for k in joints.keys():
                joints[k][0] -= x_min
                joints[k][1] -= y_min

            dd2 = {}
            for k in joints.keys():
                tmp_val = [0] * 4
                tmp_val[0] = (rescaled[1] / delta_x) * joints[k][0]
                tmp_val[1] = (rescaled[0] / delta_y) * joints[k][1]
                tmp_val[2] = joints[k][2]
                tmp_val[3] = joints[k][3]
                dd2[k] = tmp_val
            # image = np.zeros((128, 64), dtype=np.uint8)
            # for k in dd2.keys():
            #     if dd2[k][2] == 0:
            #         cv2.circle(image, (int(dd2[k][0]), int(dd2[k][1])), 2, 255, -1)
            #
            # video_reader = imageio.get_reader(self.pv)
            # for frame_number, frame in enumerate(video_reader):
            #     frame_id_tmp = 'frame_{}'.format(frame_number + 1)
            #     if frame_id_tmp != frame_id:
            #         continue
            #     else:
            #         person_crop = frame[y_min:y_max, x_min:x_max, :]
            #         break
            # cv2.imshow('prova', image)
            # cv2.imshow('persona', cv2.cvtColor(cv2.resize(person_crop, (64, 128)), cv2.COLOR_RGB2BGR))
            # cv2.waitKey()

            tmp_tup = (person_id, dd2, x_max, x_min, y_max, y_min)
            jpeople.append(tmp_tup)

        jpeople = sorted(jpeople, key=lambda person: person[5])
        return jpeople

    def save_frame(self, frame_id, jpeopleimg):
        jpeople = sorted(jpeopleimg, key=lambda person: person[5])
        back = self.background.copy()

        for person in jpeople:
            with open('/tmp/jta/{}/{}.pk'.format(frame_id, person[0]), 'rb') as fn:
                person_imgs = pickle.load(fn)

            old_img = back[person[5]:person[4], person[3]:person[2], :]
            new_img = cv2.resize(person_imgs[0], (person[2] - person[3], person[4] - person[5]))
            pos_mask = np.expand_dims(cv2.resize(person_imgs[1], (person[2] - person[3], person[4] - person[5])), 2)
            neg_mask = pos_mask.copy()
            pos_mask[pos_mask != 0] = 1
            neg_mask[pos_mask != 1] = 1
            neg_mask[pos_mask == 1] = 0
            # dilate = np.expand_dims(cv2.dilate(pos_mask, np.ones((3,3))),2)
            #
            # tmp = dilate - pos_mask
            # tmp[tmp < 0] = 0
            #
            # pos_mask = pos_mask - tmp
            # pos_mask[pos_mask < 0] = 0
            #
            # neg_mask = neg_mask - tmp
            # neg_mask[neg_mask < 0] = 0

            # blur = cv2.GaussianBlur(new_img, (5, 5), 0)
            back[person[5]:person[4], person[3]:person[2], :] = neg_mask*old_img + pos_mask*new_img # + tmp*blur,9,75,75

            os.remove('/tmp/jta/{}/{}.pk'.format(frame_id, person[0]))
        cv2.imwrite('{}/{}.jpg'.format(self.po, frame_id.split('_')[1]), back)
        os.rmdir('/tmp/jta/{}'.format(frame_id))

    def get_frames(self):
        if self.ur:
            json_seq_info = self.frames_annotation
        else:
            with open(self.pa, 'r') as fo:
                json_seq_info = json.load(fo)

        return json_seq_info.keys()




