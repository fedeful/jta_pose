import json
from typing import *
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import os

MAX_COLORS = 42
LIMBS = [
    (0, 1),  # head_top -> head_center
    (1, 2),  # head_center -> neck
    (2, 3),  # neck -> right_clavicle
    (3, 4),  # right_clavicle -> right_shoulder
    (4, 5),  # right_shoulder -> right_elbow
    (5, 6),  # right_elbow -> right_wrist
    (2, 7),  # neck -> left_clavicle
    (7, 8),  # left_clavicle -> left_shoulder
    (8, 9),  # left_shoulder -> left_elbow
    (9, 10),  # left_elbow -> left_wrist
    (2, 11),  # neck -> spine0
    (11, 12),  # spine0 -> spine1
    (12, 13),  # spine1 -> spine2
    (13, 14),  # spine2 -> spine3
    (14, 15),  # spine3 -> spine4
    (15, 16),  # spine4 -> right_hip
    (16, 17),  # right_hip -> right_knee
    (17, 18),  # right_knee -> right_ankle
    (15, 19),  # spine4 -> left_hip
    (19, 20),  # left_hip -> left_knee
    (20, 21)  # left_knee -> left_ankle
]
JOINTS = 22
NJ = 14

# NO
NO_JOINTS = [1, 3, 7, 11, 12, 13, 14, 15]


#######____JOINTS___UTILITIES__#############

def reduced_joint_id(joint: Dict) -> int:
    jid = joint_id(joint)
    if jid >= 11:
        return jid - 8
    elif jid >= 7:
        return jid - 3
    elif jid >= 3:
        return jid - 2
    elif jid >= 1:
        return jid - 1
    else:
        return jid


def find_mins_maxs(joints):

    x_min = 2000
    x_max = 0
    y_min = 2000
    y_max = 0

    for joint in joints:

        if joint['x2d'] > x_max:
            x_max = joint['x2d']
        if joint['x2d'] < x_min:
            x_min = joint['x2d']
        if joint['y2d'] > y_max:
            y_max = joint['y2d']
        if joint['y2d'] < y_min:
            y_min = joint['y2d']

    return x_min, x_max, y_min, y_max


def eliminate_joint(joint: Dict) -> bool:
    if joint_id(joint) in NO_JOINTS:
        return True
    return False


def joint_id(joint: Dict) -> int:
    return joint['jid']


def joint_occluded(joint: Dict) -> bool:
    if joint['occ'] == 1:
        return True
    return False


def joint_self_occluded(joint: Dict) -> bool:
    if joint['soc'] == 1:
        return True
    return False


def joint_visible(joint: Dict) -> bool:
    if joint['soc'] == 0 and joint['occ'] == 0:
        return True
    return False


def get_joint(pose: List[Dict], jid: int) -> Optional[Dict]:
    joint = [j for j in pose if j['jid'] == jid]
    if len(joint) == 1:
        return joint[0]
    else:
        return None


def joint_radius(joint: Dict) -> int:
    cam_distance = np.sqrt(joint['x3d'] ** 2 + joint['y3d'] ** 2 + joint['z3d'] ** 2)
    radius = int(round(np.power(10, 1 - (cam_distance / 20.0))))
    return radius if radius >= 1 else 1


def joint_pos(joint: Dict) -> Tuple[int, int]:
    return int(joint['x2d']), int(joint['y2d'])


def joint_pos_list(joint: Dict) -> List[int]:
    return [int(joint['y2d']), int(joint['x2d'])]


def joint_color(joint: Dict) -> Tuple[int, int, int]:

    if joint['occ']:
        return 255, 0, 42
    elif joint['soc']:
        return 255, 128, 42
    else:
        return 0, 255, 42


def get_colors(number_of_colors: int, cmap_name: str = 'rainbow') -> List[List[int]]:
    #colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, number_of_colors))[:, :-1] * 255
    #return colors.astype(int).tolist()
    return 0


def all_occ(pose: List[Dict]) -> bool:
    for j in pose:
        if j['occ'] == 0:
            return False
    return True


