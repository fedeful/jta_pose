from typing import *

import numpy as np
import torch


class Joint(object):
    # list of joint names (ordered by joint type)
    NAMES = [
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
        'body_center'  # 14
    ]

    # list of limbs
    # >> (A,B) is the limb that links joint of type A to joint of type B
    LIMBS = [
        (0, 1),  # head_top -> neck
        (1, 2),  # neck -> r_shoulder
        (2, 3),  # r_shoulder -> r_elbow
        (3, 4),  # r_elbow -> r_wrist
        (1, 5),  # neck -> l_shoulder
        (5, 6),  # l_shoulder -> l_elbow
        (6, 7),  # l_elbow -> l_wrist
        (1, 14),  # neck -> body_center
        (14, 8),  # body_center -> r_hip
        (8, 9),  # r_hip -> r_knee
        (9, 10),  # r_knee -> r_ankle
        (14, 11),  # body_center -> l_hip
        (11, 12),  # l_hip -> l_knee
        (12, 13),  # l_knee -> l_ankle
    ]

    DRAW_LIMBS = [
        (0, 1),  # head_top -> neck
        (1, 2),  # neck -> r_shoulder
        (2, 3),  # r_shoulder -> r_elbow
        (3, 4),  # r_elbow -> r_wrist
        (1, 5),  # neck -> l_shoulder
        (5, 6),  # l_shoulder -> l_elbow
        (6, 7),  # l_elbow -> l_wrist
        (1, 14),  # neck -> body_center
        (14, 8),  # body_center -> r_hip
        (8, 9),  # r_hip -> r_knee
        (9, 10),  # r_knee -> r_ankle
        (14, 11),  # body_center -> l_hip
        (11, 12),  # l_hip -> l_knee
        (12, 13),  # l_knee -> l_ankle
    ]

    # list of limb names (ordered as list 'LIMBS')
    LIMB_NAMES = [
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

    # total number of joints
    N_JOINTS = len(NAMES)

    # total number of limbs
    N_LIMBS = len(LIMBS)

    def __init__(self, x: int, y: int, jtype: int, occluded: int, self_occluded: int,
                 frame: int = None, confidence: float = 1., person_id: int = None,
                 cam_dist: float = None, taf_value: Tuple[int, int] = None):
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
        self.name = Joint.NAMES[self.type]
        self.cam_dist = cam_dist
        self.confidence = confidence
        self.id_candidates = []  # type: List[Tuple[int, float]]
        self.taf_value = taf_value

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
    def os_occluded(self) -> bool:
        """
        '(O)ccluded or (S)elf(_OCCLUDED) = OS_OCCLUDED
        """
        return bool(self.occluded) or bool(self.self_occluded)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return '{' + 'name: \'{}\', pos: {}, occluded: {}, self_occluded {}, person_id: {}, confidence: {:.3f}'.format(
            self.name, self.position, self.occluded, self.self_occluded, self.person_id, self.confidence) + '}'

    __repr__ = __str__
