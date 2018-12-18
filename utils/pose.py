import math
from copy import deepcopy
from typing import *

import cv2
import numpy as np
from numpy import ndarray as NPArray

from shared.bbox import BBox
from shared.joint import Joint

Point2D = Tuple[int, int]


def gkern(h: int, w: int, center: Point2D, s: float = 4) -> np.ndarray:
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]

    x0 = center[0]
    y0 = center[1]

    return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)


class Pose(list):

    def __init__(self, joints: List[Joint] = None, id: int = None):
        super(Pose, self).__init__(joints if joints != None else [])
        if not joints is None and len(joints) > 0 and id is None:
            self.id = joints[0].person_id
        elif not id is None:
            self.id = id
        else:
            raise ValueError('ERROR: the pose needs an ID.')

    @property
    def id(self) -> int:
        return self._id

    @property
    def not_none(self) -> List[Joint]:
        """
        :return: a list of the not-None joints of the pose
        """
        return [j for j in self if j is not None]

    @property
    def cam_dist(self) -> float:
        if len(self) == 0:
            return -1
        return float(np.mean([j.cam_dist for j in self]))

    @property
    def size(self) -> int:
        return len(self)

    @id.setter
    def id(self, new_id: int):
        self._id = new_id
        for j in range(len(self)):
            self[j].person_id = self._id

    @property
    def center(self) -> Tuple[float, float]:
        points = np.array([j.position for j in self])
        mid_x = np.mean(points[0, :])  # type: float
        mix_y = np.mean(points[1, :])  # type: float
        return mid_x, mix_y

    def remove_duplicates(self):
        """
        Remove duplicate joints from the pose.
        Example: if the pose contains 3 left knees, I want to keep the
        knee with the best confidence and I want to remove the others.
        """
        for joint_type in range(Joint.N_JOINTS):
            joints_of_that_type = [j for j in self.not_none if j.type == joint_type]
            if len(joints_of_that_type) > 1:
                best_joint = max(joints_of_that_type, key=lambda j: j.confidence)
                for j in joints_of_that_type:
                    if id(j) != id(best_joint):
                        del self[self.index(j)]

    def copy(self) -> 'Pose':
        return deepcopy(self)

    def sort_by_type(self):
        """
        ATTENTION: inplace
        """
        if len(self) > 1:
            self.sort(key=lambda joint: joint.type)

    def sort_by_confidence(self):
        """
        ATTENTION: inplace
        """
        self.sort(key=lambda joint: joint.confidence)

    def get_joint_of_type(self, jtype: int) -> Optional[Joint]:
        """
        Returns the joint with the desired type
        (or 'None' if the pose does not contain the joint with the desired type)
        """
        for j in self:
            if j.type == jtype:
                return j
        return None

    def __str__(self) -> str:
        n_joints = len(self)
        description = 'POSE#{} ({} {}): {{\n'.format(self.id, n_joints, 'joint' if n_joints == 1 else 'joints')
        for i, j in enumerate(self):
            description += '   {}) {}\n'.format(i, j)
        description += '}'
        return description

    def __repr__(self) -> str:
        n_joints = len(self)
        return 'POSE#{} ({} {}): {{\n'.format(self.id, n_joints, 'joint' if n_joints == 1 else 'joints')

    def __iter__(self) -> Iterator[Joint]:
        return super(Pose, self).__iter__()

    def __getitem__(self, index: int) -> Joint:
        return super(Pose, self).__getitem__(index)

    def get_heatmaps(self, h: int, w: int) -> List[NPArray]:
        """
        :param h: height of HMaps [pixel]
        :param w: width of HMaps [pizel]
        :returns: list containing 1 HMap for each type of joint
        """
        heatmaps = []
        for i in range(Joint.N_JOINTS):
            joint = self[i]
            if joint is None:
                hmap = np.zeros((h, w)).astype(np.float32)
                heatmaps.append(hmap)
            else:
                cx = int(round(joint.x / 8.0))
                cy = int(round(joint.y / 8.0))
                k = -1 if joint.occluded or joint.self_occluded else 1
                heatmaps.append(k * gkern(h=h, w=w, s=12.5 / joint.cam_dist, center=(cx, cy)))
        # heatmaps.append(k * gkern(h=h, w=w, s=1, center=(cx, cy)))
        return heatmaps

    def get_hmaps(self, h: int, w: int, sigma: float = None) -> (np.ndarray, np.ndarray):
        """
        returns 2 lists of HMaps (1 for visible joints and 1 for occluded joints)
        """
        hmaps_no_occl = []
        hmaps_only_occl = []
        for i in range(Joint.N_JOINTS):
            joint = self.get_joint_of_type(jtype=i)

            empty_hmap = np.zeros((h, w)).astype(np.float32)
            if joint is None:
                hmaps_no_occl.append(empty_hmap)
                hmaps_only_occl.append(empty_hmap)
            else:
                cx = joint.x
                cx = int(round(cx / 8.0))
                cy = joint.y
                cy = int(round(cy / 8.0))
                if sigma is None:
                    sigma = (100 / joint.cam_dist)
                hmap = gkern(h=h, w=w, s=sigma, center=(cx, cy))

                if joint.os_occluded:
                    hmaps_only_occl.append(hmap)
                    hmaps_no_occl.append(empty_hmap)
                else:
                    hmaps_no_occl.append(hmap)
                    hmaps_only_occl.append(empty_hmap)

        return hmaps_no_occl, hmaps_only_occl

    def get_pafs(self, h: int, w: int, line_width: int = 4) -> List[NPArray]:
        """
        returns a list containing 2 PAFs for each type of limb
        """
        pafs = []
        for (a, b) in Joint.LIMBS:
            paf = np.zeros((h, w)).astype(np.float32)
            joint1 = self.get_joint_of_type(a)
            joint2 = self.get_joint_of_type(b)
            if joint1 is not None and joint2 is not None:
                paf_x = paf.copy()
                paf_y = paf.copy()
                vec = np.subtract(joint2.position, joint1.position)
                norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                if norm != 0:
                    vec = np.divide(vec, norm)

                    # c1 = (c1x, c1y) = coordinates of the 1st joint
                    c1x = joint1.x
                    c1x = int(round(c1x / 8.0))
                    c1y = joint1.y
                    c1y = int(round(c1y / 8.0))
                    c1 = (c1x, c1y)

                    # c2 = (c2x, c2y) = coordinates of the 2nd joint
                    c2x = joint2.x
                    c2x = int(round(c2x / 8.0))
                    c2y = joint2.y
                    c2y = int(round(c2y / 8.0))
                    c2 = (c2x, c2y)

                    # paf_x and paf_y are lines from c1 to c2
                    cv2.line(paf_x, c1, c2, vec[0], line_width)
                    cv2.line(paf_y, c1, c2, vec[1], line_width)

                pafs.append(paf_x)
                pafs.append(paf_y)
            else:
                pafs.append(paf)
                pafs.append(paf)
        return pafs

    def get_pufs(self, h: int, w: int, line_width: int = 4) -> List[NPArray]:
        """
        returns a list containing 2 PAFs for each type of limb
        """
        pufs = []
        for (a, b) in Joint.LIMBS:
            paf = np.zeros((h, w)).astype(np.float32)
            ref_x = np.ones_like(paf)
            ref_y = np.zeros_like(paf)
            joint1 = self[a]
            joint2 = self[b]
            if joint1 is not None and joint2 is not None:
                paf_x = paf.copy()
                paf_y = paf.copy()
                vec = np.subtract(joint1.position, joint2.position)
                norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                if norm != 0:
                    vec = np.divide(vec, norm)

                    # c1 = (c1x, c1y) = coordinates of the 1st joint
                    c1x = joint1.x
                    c1x = int(round(c1x / 8.0))
                    c1y = joint1.y
                    c1y = int(round(c1y / 8.0))
                    c1 = (c1x, c1y)

                    # c2 = (c2x, c2y) = coordinates of the 2nd joint
                    c2x = joint2.x
                    c2x = int(round(c2x / 8.0))
                    c2y = joint2.y
                    c2y = int(round(c2y / 8.0))
                    c2 = (c2x, c2y)

                    # paf_x and paf_y are lines from c1 to c2
                    cv2.line(paf_x, c1, c2, vec[0], line_width)
                    cv2.line(paf_y, c1, c2, vec[1], line_width)

                puf = paf_x * ref_x + paf_y * ref_y
                pufs.append(puf)
            else:
                pufs.append(paf)
        return pufs

    def get_tafs(self, h: int, w: int, prev_pose: 'Pose', line_width: int = 6) -> List[NPArray]:
        """
        returns a list containing 2 TAFs for each type of joint
        """
        tafs = []
        for joint1 in self:
            taf = np.zeros((h, w)).astype(np.float32)
            if prev_pose is None:
                joint2 = None
            else:
                joint2 = prev_pose[joint1.type]
            if joint2 is not None and prev_pose is not None:
                taf_x = taf.copy()
                taf_y = taf.copy()
                vec = np.subtract(joint2.position, joint1.position)
                norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                if norm != 0:
                    vec = np.divide(vec, norm)

                    # c1 = (c1x, c1y) = coordinates of the 1st joint
                    c1x = joint1.x
                    c1x = int(round(c1x / 8.0))
                    c1y = joint1.y
                    c1y = int(round(c1y / 8.0))
                    c1 = (c1x, c1y)

                    # c2 = (c2x, c2y) = coordinates of the 2nd joint
                    c2x = joint2.x
                    c2x = int(round(c2x / 8.0))
                    c2y = joint2.y
                    c2y = int(round(c2y / 8.0))
                    c2 = (c2x, c2y)

                    # paf are lines from c1 to c2
                    line_width = 6
                    cv2.line(taf_x, c1, c2, vec[0], line_width)
                    cv2.line(taf_y, c1, c2, vec[1], line_width)

                tafs.append(taf_x)
                tafs.append(taf_y)
            else:
                tafs.append(taf)
                tafs.append(taf)
        return tafs

    def is_all_occluded(self) -> bool:
        for j in self:
            if not j.occluded:
                return False
        return True

    def is_good(self, maximum_occlusion_degree: float = 0.8) -> bool:
        """
        :param maximum_occlusion_degree: values in [0, 1]
        """

        # if all the joints of the pose are offscreen
        # the pose is considered NOT good
        n_offscreen_joints = 0
        for joint in self:
            if not joint.on_screen:
                n_offscreen_joints += 1
        if n_offscreen_joints == len(self):
            return False

        # if 'head_top' is not occluded, the pose is
        # automatically considered GOOD
        if not self[0].occluded:
            return True

        # if the occlusion degree is > then the specified threshold
        # the pose is considered NOT good
        occlusion_degree = 0
        for joint in self:
            if joint.occluded:
                occlusion_degree += 1
        occlusion_degree = occlusion_degree / len(self)
        return not (occlusion_degree > maximum_occlusion_degree)

    def has_height_in_range(self, min_h: int = 100, max_h: int = 400) -> bool:
        """
        :return: 'True' if the pose has a height in range [min_h, max_h], 'False' otherwise
        """
        height = max(self[10].y, self[13].y) - self[0].y
        if (height >= min_h and height <= max_h):
            return True
        else:
            return False

    def get_bbox(self, w_inc_perc: float = 0, h_inc_perc: float = 0) -> BBox:
        """
        :param w_inc_perc: percentage increase in width
        :param h_inc_perc: percentage increase in height
        :return: bounding box
        """
        x_min = np.min([j.x for j in self])
        x_max = np.max([j.x for j in self])
        y_min = np.min([j.y for j in self])
        y_max = np.max([j.y for j in self])
        h = x_max - x_min
        w = y_max - y_min
        dw = int(round(w * w_inc_perc))
        dh = int(round(h * h_inc_perc))

        return BBox(
            x_min=x_min - dw,
            x_max=x_max + dw,
            y_min=y_min - dh,
            y_max=y_max + dh,
            person_id=self.id
        )

    @staticmethod
    def euc_distance(pose1: 'Pose', pose2: 'Pose') -> float:
        """
        :return: euclidian distance between 2 poses
        """
        dist_sum = 0.0
        dist_count = 0
        for joint_type in range(Joint.N_JOINTS):
            j1 = pose1.get_joint_of_type(joint_type)
            j2 = pose2.get_joint_of_type(joint_type)
            if (j1 is not None) and (j2 is not None):
                dist_sum += np.linalg.norm(np.array(j1.position) - np.array(j2.position))
                dist_count += 1
        return (dist_sum / dist_count) if dist_count != 0 else float("inf")

    @staticmethod
    def point_distance(pose1: 'Pose', pose2: 'Pose') -> float:
        """
        :return: 'point' euclidian distance between 2 poses
        """
        bbox1 = pose1.get_bbox()
        bbox2 = pose2.get_bbox()
        p1 = (int(round((bbox1.x_min + bbox1.x_max) / 2)), bbox1.y_max)
        p2 = (int(round((bbox2.x_min + bbox2.x_max) / 2)), bbox2.y_max)
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def euc_track(cur_poses: List['Pose'], prev_poses: List['Pose']):

        if len(cur_poses) == 0 or len(prev_poses) == 0:
            return

        assignments = []
        for cur_pose in cur_poses:
            prev_pose = min(prev_poses, key=lambda prev_pose: Pose.euc_distance(cur_pose, prev_pose))
            cur_pose.id = prev_pose.id
            edist = Pose.euc_distance(cur_pose, prev_pose)
            assignments.append((prev_pose.id, cur_pose, edist))
        assignments = np.array(assignments)

        max_id = assignments[:, 0].max()
        duplicates = []
        for i in range(max_id + 1):
            duplicate = np.array([a for a in assignments if a[0] == i])
            if len(duplicate) > 1:
                duplicates.append(duplicate)

        for d_set in duplicates:
            correct_a = min(d_set, key=lambda x: x[2])[1]
            for a in d_set:
                if a[1] != correct_a:
                    new_id = 0
                    list_ids = [p.id for p in cur_poses]
                    while True:
                        if not new_id in list_ids:
                            a[1].id = new_id
                            break
                        else:
                            new_id += 1

    @staticmethod
    def pnt_track(cur_poses: List['Pose'], prev_poses: List['Pose']):

        if len(cur_poses) == 0 or len(prev_poses) == 0:
            return

        assignments = []
        for cur_pose in cur_poses:
            prev_pose = min(prev_poses, key=lambda prev_pose: Pose.point_distance(cur_pose, prev_pose))
            cur_pose.id = prev_pose.id
            edist = Pose.euc_distance(cur_pose, prev_pose)
            assignments.append((prev_pose.id, cur_pose, edist))
        assignments = np.array(assignments)

        max_id = assignments[:, 0].max()
        duplicates = []
        for i in range(max_id + 1):
            duplicate = np.array([a for a in assignments if a[0] == i])
            if len(duplicate) > 1:
                duplicates.append(duplicate)

        for d_set in duplicates:
            correct_a = min(d_set, key=lambda x: x[2])[1]
            for a in d_set:
                if a[1] != correct_a:
                    new_id = 0
                    list_ids = [p.id for p in cur_poses]
                    while True:
                        if not new_id in list_ids:
                            a[1].id = new_id
                            break
                        else:
                            new_id += 1