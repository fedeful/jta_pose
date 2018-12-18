# -*- coding: utf-8 -*-
# 🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐

from typing import *

import numpy as np
import torch


class Joint(object):

    CPM_NAMES = [
        'head_top',  # 0
        'neck',  # 1
        'r_shoulder',  # 2
        'r_elbow',  # 3
        'r_wrist',  # 4
        'l_shoulder',  # 5
        'l_elbow',  # 6
        'l_wrist',  # 7
        'r_hip',  # 8
        'r_knee',  # 9
        'r_ankle',  # 10
        'l_hip',  # 11
        'l_knee',  # 12
        'l_ankle',  # 13
        'r_eye',  # 14
        'l_eye',  # 15
        'r_ear',  # 16
        'l_ear',  # 17
    ]

    CPM_NAMES_REDUCED = [
        'head_top',  # 0
        'neck',  # 1
        'r_shoulder',  # 2
        'r_elbow',  # 3
        'r_wrist',  # 4
        'l_shoulder',  # 5
        'l_elbow',  # 6
        'l_wrist',  # 7
        'r_hip',  # 8
        'r_knee',  # 9
        'r_ankle',  # 10
        'l_hip',  # 11
        'l_knee',  # 12
        'l_ankle',  # 13
    ]

    CPM_DICTIONARY = {
        'head_top': 0,  # 0
        'neck': 1,  # 1
        'r_shoulder': 2,  # 2
        'r_elbow': 3,  # 3
        'r_wrist': 4,  # 4
        'l_shoulder': 5,  # 5
        'l_elbow': 6,  # 6
        'l_wrist': 7,  # 7
        'r_hip': 8,  # 8
        'r_knee': 9,  # 9
        'r_ankle': 10,  # 10
        'l_hip': 11,  # 11
        'l_knee': 12,  # 12
        'l_ankle': 13,  # 13
        'r_eye': 14,  # 14
        'l_eye': 15,  # 15
        'r_ear': 16,  # 16
        'l_ear': 17,  # 17
    }

    CPM_LIMBS = [
        (0, 1),  # head_top -> neck
        (1, 2),  # neck -> r_shoulder
        (2, 3),  # r_shoulder -> r_elbow
        (3, 4),  # r_elbow -> r_wrist
        (1, 5),  # neck -> l_shoulder
        (5, 6),  # l_shoulder -> l_elbow
        (6, 7),  # l_elbow -> l_wrist
        (1, 8),  # neck -> r_hip
        (8, 9),  # r_hip -> r_knee
        (9, 10),  # r_knee -> r_ankle
        (1, 11),  # neck -> l_hip
        (11, 12),  # l_hip -> l_knee
        (12, 13),  # l_knee -> l_ankle
        (0, 14),  # head_top -> r_eye
        (0, 15),  # head_top -> l_eye
        (14, 16),  # r_eye -> r_ear
        (15, 17),  # l_eye -> l_ear
    ]

    CPM_LIMBS_AFFINE = [
        (0, 1),  # head_top -> neck
        (2, 3),  # r_shoulder -> r_elbow
        (3, 4),  # r_elbow -> r_wrist
        (5, 6),  # l_shoulder -> l_elbow
        (6, 7),  # l_elbow -> l_wrist
        (1, 8),  # neck -> r_hip
        (8, 9),  # r_hip -> r_knee
        (9, 10),  # r_knee -> r_ankle
        (1, 11),  # neck -> l_hip
        (11, 12),  # l_hip -> l_knee
        (12, 13),  # l_knee -> l_ankle
    ]

    CPM_LIMBS_AFFINE_DICTIONARY = {
        '0': (0, 1),  # head_top -> neck
        '1': (2, 3),  # r_shoulder -> r_elbow
        '2': (3, 4),  # r_elbow -> r_wrist
        '3': (5, 6),  # l_shoulder -> l_elbow
        '4': (6, 7),  # l_elbow -> l_wrist
        '5': (1, 8),  # neck -> r_hip
        '6': (8, 9),  # r_hip -> r_knee
        '7': (9, 10),  # r_knee -> r_ankle
        '8': (1, 11),  # neck -> l_hip
        '9': (11, 12),  # l_hip -> l_knee
        '10': (12, 13),  # l_knee -> l_ankle
    }

    CPM_DRAW_LIMBS = [
        (0, 1),  # head_top -> neck
        (1, 2),  # neck -> r_shoulder
        (2, 3),  # r_shoulder -> r_elbow
        (3, 4),  # r_elbow -> r_wrist
        (1, 5),  # neck -> l_shoulder
        (5, 6),  # l_shoulder -> l_elbow
        (6, 7),  # l_elbow -> l_wrist
        (1, 8),  # neck -> r_hip
        (8, 9),  # r_hip -> r_knee
        (9, 10),  # r_knee -> r_ankle
        (1, 11),  # neck -> l_hip
        (11, 12),  # l_hip -> l_knee
        (12, 13),  # l_knee -> l_ankle
        (0, 14),  # head_top -> r_eye
        (0, 15),  # head_top -> l_eye
        (14, 16),  # r_eye -> r_ear
        (15, 17),  # l_eye -> l_ear
        (8, 11),  # r_hip -> l_hip
        (5, 11),  # l_shoulder -> l_hip
        (2, 8),  # r_shoulder -> r_hip
        (1, 16),  # neck -> r_hear
        (1, 17),  # neck -> l_hear
    ]

    CPM_LIMB_NAMES = [
        'head_top->neck',
        'neck->r_shoulder',
        'r_shoulder->r_elbow',
        'r_elbow->r_wrist',
        'neck->l_shoulder',
        'l_shoulder->l_elbow',
        'l_elbow->l_wrist',
        'neck->body_center',
        'body_center->r_hip',
        'r_hip->r_knee',
        'r_knee->r_ankle',
        'body_center->l_hi',
        'l_hip->l_knee',
        'l_knee->l_ankle',
    ]

    MIN_JOINT_CONFIDENCE = 0.00

    def __init__(self, x: int, y: int, jtype: int, occluded: int, self_occluded: int,
                 frame: int = None, confidence: float = 0.1, person_id: int = None,
                 cam_dist: float = None, taf_value: Tuple[int, int] = None, kind='CPM'):
        """
        :param x: x coordinate of the joint [pixel]
        :param y: y coordinate of the joint [pixel]
        :param jtype: integer type of the joint [example: 0=head_top, 1=head_center, ...]
        :param occluded: is this joint occluded by something that is not the person that owns it?
        :param self_occluded: is this joint occluded by the person that owns it?
        :param frame: number of the video-frame that contains the joint
        :param confidence: confidence of the joint [min=0, max=1]
        :param person_id: unique identifier of the person that owns the joint
        :param cam_dist: distance between camera and joint [m]
        """
        self.x = x
        self.y = y
        self.on_screen = not (x < 0 or x > 1920 or y < 0 or y > 1080)
        self.type = jtype
        self.occluded = occluded
        self.self_occluded = self_occluded
        self.person_id = person_id
        self.frame = frame
        self.cam_dist = cam_dist
        self.confidence = confidence
        self.id_candidates = []
        self.taf_value = taf_value
        self.kind = kind

        if self.kind == 'CPM':
            self.name = Joint.CPM_NAMES[self.type]
        elif self.kind == 'PTK':
            self.name = Joint.CPM_NAMES[self.type]
        elif self.kind == 'JTA':
            self.name = Joint.CPM_NAMES[self.type]
        else:
            self.name = Joint.CPM_NAMES[self.type]


    @property
    def position(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def rad_direction(self) -> Optional[float]:
        """
        :return: the direction, compared to the vector (0,1), pointed by the TAF of that joint
        """
        if self.taf_value is None:
            return None
        else:
            norm_taf_val = self.taf_value / np.linalg.norm(self.taf_value)
            angle = np.arccos(np.clip(np.dot(norm_taf_val, (1, 0)), -1.0, 1.0))
            return angle

    @property
    def cuda_position(self) -> torch.IntTensor:
        p = np.array([self.x, self.y])
        return torch.from_numpy(p).int()

    @position.setter
    def position(self, xy: Tuple[int, int]):
        xy = tuple(xy)
        self.x = int(round(xy[0]))
        self.y = int(round(xy[1]))

    @property
    def numpy(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.type,
            self.occluded,
            self.self_occluded,
            self.person_id,
            self.frame,
            self.cam_dist,
            self.confidence,
        ])

    @property
    def is_occluded(self) -> bool:

        return bool(self.occluded) or bool(self.self_occluded)

    @property
    def is_only_occ(self):
        return bool(self.occluded)

    @property
    def is_self_occluded(self) -> bool:
        return bool(self.self_occluded)

    @property
    def joints_names(self):
        if self.kind == 'CPM':
            return self.CPM_NAMES
        elif self.kind == 'PTK':
            return self.CPM_NAMES
        elif self.kind == 'JTA':
            return self.CPM_NAMES
        else:
            return self.CPM_NAMES

    @property
    def joints_number(self):
        if self.kind == 'CPM':
            return len(self.CPM_NAMES)
        elif self.kind == 'PTK':
            return len(self.CPM_NAMES)
        elif self.kind == 'JTA':
            return len(self.CPM_NAMES)
        else:
            return len(self.CPM_NAMES)

    @property
    def limbs(self):
        if self.kind == 'CPM':
            return self.CPM_LIMBS
        elif self.kind == 'PTK':
            return self.CPM_LIMBS
        elif self.kind == 'JTA':
            return self.CPM_LIMBS
        else:
            return self.CPM_LIMBS

    @property
    def limbs_affine(self):
        if self.kind == 'CPM':
            return self.CPM_LIMBS_AFFINE
        elif self.kind == 'PTK':
            return self.CPM_LIMBS
        elif self.kind == 'JTA':
            return self.CPM_LIMBS
        else:
            return self.CPM_LIMBS

    @property
    def limbs_affine_dictionary(self):
        if self.kind == 'CPM':
            return self.CPM_LIMBS_AFFINE_DICTIONARY
        elif self.kind == 'PTK':
            return self.CPM_LIMBS
        elif self.kind == 'JTA':
            return self.CPM_LIMBS
        else:
            return self.CPM_LIMBS

    @property
    def limbs_to_draw(self):
        if self.kind == 'CPM':
            return self.CPM_DRAW_LIMBS
        elif self.kind == 'PTK':
            return self.CPM_DRAW_LIMBS
        elif self.kind == 'JTA':
            return self.CPM_DRAW_LIMBS
        else:
            return self.CPM_DRAW_LIMBS

    @property
    def limbs_number(self):
        if self.kind == 'CPM':
            return len(self.CPM_LIMBS)
        elif self.kind == 'PTK':
            return len(self.CPM_LIMBS)
        elif self.kind == 'JTA':
            return len(self.CPM_LIMBS)
        else:
            return len(self.CPM_LIMBS)

    @property
    def joints_number_reduced(self):
        if self.kind == 'CPM':
            return len(self.CPM_NAMES_REDUCED)
        elif self.kind == 'PTK':
            return len(self.CPM_NAMES)
        elif self.kind == 'JTA':
            return len(self.CPM_NAMES)
        else:
            return len(self.CPM_NAMES)

    @property
    def get_confidence(self):
        return self.confidence

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return '{' + 'name: \'{}\', pos: {}, occluded: {}, self_occluded {}, person_id: {}, confidence: {:.3f}'.format(
            self.name, self.position, self.occluded, self.self_occluded, self.person_id, self.confidence) + '}'

    __repr__ = __str__
