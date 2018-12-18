# -*- coding: utf-8 -*-
# 🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐

import math
import warnings
from copy import deepcopy
from typing import *
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import cv2
import numpy as np
from numpy import ndarray as NPArray

from body.bbox import BBox
from body.joint import Joint

Point2D = Tuple[int, int]

MIN_JOINT_CONFIDENCE = 0.1


def gkern(h: int, w: int, center: Point2D, s: float = 4) -> np.ndarray:
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]

    x0 = center[0]
    y0 = center[1]

    return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)


class Pose(list):

    def __init__(self, joints: List[Joint] = None, idp: int = None):

        super(Pose, self).__init__([] if joints is None else joints)
        if joints is not None and len(joints) > 0 and id is None:
            self._id = joints[0].person_id
        elif id is not None:
            self._id = idp
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
        for joint_type in range(self[0].joints_number):
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

    def get_heatmap(self, h, w, sub_h, sub_w, scale_h, scale_w, sigma=2):
        hmaps = []
        for i in range(self[0].joints_number_reduced):
            joint = self.get_joint_of_type(jtype=i)
            if joint is None:
                shmap = np.zeros((h, w)).astype(np.float32)
                hmaps.append(shmap)
            elif joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                shmap = np.zeros((h, w)).astype(np.float32)
                hmaps.append(shmap)
            elif joint.is_occluded:
                shmap = np.zeros((h, w)).astype(np.float32)
                hmaps.append(shmap)
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                hmaps.append(gkern(h=h, w=w, s=sigma, center=(cx, cy)))
        return hmaps

    def get_heatmap2(self, h, w, sub_h, sub_w, scale_h, scale_w, sigma=2):
        hmaps = []
        for i in range(self[0].joints_number_reduced):
            joint = self.get_joint_of_type(jtype=i)
            if joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                shmap = np.zeros((h, w)).astype(np.float32)
                hmaps.append(shmap)
            elif joint.is_occluded:
                shmap = np.zeros((h, w)).astype(np.float32)
                hmaps.append(shmap)
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                hmaps.append(gkern(h=h, w=w, s=sigma, center=(cx, cy)))

        joint = self.get_joint_of_type(jtype=0)
        if joint.confidence > 0.01:
            cx = joint.x
            cy = joint.y
            cx = int(cx)
            cy = int(cy)
            hmaps.append(gkern(h=h, w=w, s=sigma, center=(cx, cy)))
        else:
            shmap = np.zeros((h, w)).astype(np.float32)
            hmaps.append(shmap)

        return hmaps

    def get_mask(self, h, w, sub_h, sub_w, scale_h, scale_w, line_width, circle_width):
        mask = np.zeros((h, w)).astype(np.uint8)
        joint_data = {}
        for i in range(self[0].joints_number):
            joint = self.get_joint_of_type(jtype=i)
            if joint is None:
                joint_data[i] = {'ok': False}
                continue
            elif joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                joint_data[joint.type] = {'ok': False}
                continue
            elif joint.is_only_occ:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                joint_data[joint.type] = {
                    'ok': False,
                    'cx': cx,
                    'cy': cy
                }
                continue
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                joint_data[joint.type] = {
                    'ok': True,
                    'cx': cx,
                    'cy': cy
                }

        for (a, b) in self[0].limbs_to_draw:
            if joint_data[a]['ok'] and joint_data[b]['ok']:
                cv2.line(mask,
                         (joint_data[a]['cx'], joint_data[a]['cy']),
                         (joint_data[b]['cx'], joint_data[b]['cy']),
                         255,
                         line_width)
                cv2.circle(
                    mask,
                    (joint_data[a]['cx'], joint_data[a]['cy']),
                    circle_width,
                    255,
                    -1
                )
                cv2.circle(
                    mask,
                    (joint_data[b]['cx'], joint_data[b]['cy']),
                    circle_width,
                    255,
                    -1
                )
            elif joint_data[a]['ok'] and 'cx' in joint_data[b]:
                cv2.circle(
                    mask,
                    (joint_data[a]['cx'], joint_data[a]['cy']),
                    circle_width,
                    255,
                    -1
                )

                bx = int((joint_data[a]['cx'] + joint_data[b]['cx'])/2)
                by = int((joint_data[a]['cy'] + joint_data[b]['cy'])/2)

                cv2.line(mask,
                         (joint_data[a]['cx'], joint_data[a]['cy']),
                         (bx, by),
                         255,
                         line_width)

            elif joint_data[b]['ok'] and 'cx' in joint_data[a]:
                cv2.circle(
                    mask,
                    (joint_data[b]['cx'], joint_data[b]['cy']),
                    circle_width,
                    255,
                    -1
                )
                bx = int((joint_data[a]['cx'] + joint_data[b]['cx']) / 2)
                by = int((joint_data[a]['cy'] + joint_data[b]['cy']) / 2)

                cv2.line(mask,
                         (bx, by),
                         (joint_data[b]['cx'], joint_data[b]['cy']),
                         255,
                         line_width)

        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
        mask = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=1)
        return mask

    def get_skeleton(self, h, w, sub_h, sub_w, scale_h, scale_w, line_width, cirlce_width):
        skeleton = np.zeros((h, w)).astype(np.uint8)
        joint_data = {}
        for i in range(self[0].joints_number):
            joint = self.get_joint_of_type(jtype=i)

            if joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                joint_data[joint.type] = {'ok': False}
                continue
            elif joint.is_occluded:
                joint_data[joint.type] = {'ok': False}
                continue
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                joint_data[joint.type] = {
                    'ok': True,
                    'cx': cx,
                    'cy': cy
                }

        for (a, b) in self[0].limbs_to_draw:
            if joint_data[a]['ok'] and joint_data[b]['ok']:
                cv2.line(skeleton,
                         (joint_data[a]['cx'], joint_data[a]['cy']),
                         (joint_data[b]['cx'], joint_data[b]['cy']),
                         255,
                         line_width)
                cv2.circle(
                    skeleton,
                    (joint_data[a]['cx'], joint_data[a]['cy']),
                    cirlce_width,
                    255,
                    -1
                )
                cv2.circle(
                    skeleton,
                    (joint_data[b]['cx'], joint_data[b]['cy']),
                    cirlce_width,
                    255,
                    -1
                )
        return skeleton

    def get_limb_affine_matrix(self, h, w, sub_h, sub_w, scale_h, scale_w):
        affine_matrix = np.zeros((h, w, len(self[0].limbs_affine))).astype(np.uint8)
        joint_data = {}
        for i in range(self[0].joints_number):
            joint = self.get_joint_of_type(jtype=i)

            if joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                joint_data[joint.type] = {'ok': False}
                continue
            elif joint.is_occluded:
                joint_data[joint.type] = {'ok': False}
                continue
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                joint_data[joint.type] = {
                    'ok': True,
                    'cx': cx,
                    'cy': cy
                }

        rectangle_coords = dict()
        for i in np.arange(len(self[0].limbs_affine_dictionary)):
            a, b = self[0].limbs_affine_dictionary['{}'.format(i)]

            if joint_data[a]['ok'] and joint_data[b]['ok']:
                ax_ = joint_data[a]['cx']
                ay_ = joint_data[a]['cy']
                bx_ = joint_data[b]['cx']
                by_ = joint_data[b]['cy']
                dpoints = self.find_polygon_vertex(ax_, ay_, bx_, by_, w, h)
                rectangle_coords['{}_{}'.format(a, b)] = dpoints
                dl = dpoints
                image_rgb = Image.new("RGB", (64, 128))
                draw = ImageDraw.Draw(image_rgb)
                points = ((dl[0][0], dl[0][1]), (dl[1][0], dl[1][1]), (dl[3][0], dl[3][1]), (dl[2][0], dl[2][1]))
                draw.polygon((points), fill=255)
                affine_matrix[..., i] = np.array(image_rgb)[..., 0]
            else:
                affine_matrix[..., i] = np.zeros((128, 64))

        return affine_matrix

    def get_limb_affine(self, h, w, sub_h, sub_w, scale_h, scale_w):
        skeleton = np.zeros((h, w)).astype(np.uint8)
        joint_data = {}
        for i in range(self[0].joints_number):
            joint = self.get_joint_of_type(jtype=i)
            if joint is None:
                joint_data[i] = {'ok': False}
                continue
            elif joint.confidence < joint.MIN_JOINT_CONFIDENCE:
                joint_data[joint.type] = {'ok': False}
                continue
            elif joint.is_occluded:
                joint_data[joint.type] = {'ok': False}
                continue
            else:
                if scale_h != 0 and scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_h != 0:
                    cx = int(round(joint.x - sub_w))
                    cy = int(round((joint.y - sub_h) * scale_h))
                elif scale_w != 0:
                    cx = int(round((joint.x - sub_w) * scale_w))
                    cy = int(round(joint.y - sub_h))
                else:
                    cx = int(round((joint.x - sub_w)))
                    cy = int(round((joint.y - sub_h)))

                joint_data[joint.type] = {
                    'ok': True,
                    'cx': cx,
                    'cy': cy
                }

        rectangle_coords = dict()
        for (a, b) in self[0].limbs_affine:

            if joint_data[a]['ok'] and joint_data[b]['ok']:
                ax_ = joint_data[a]['cx']
                ay_ = joint_data[a]['cy']
                bx_ = joint_data[b]['cx']
                by_ = joint_data[b]['cy']

                dpoints = self.find_polygon_vertex(ax_, ay_, bx_, by_, w, h)
                rectangle_coords['{}{}'.format(a, b)] = dpoints
            else:
                rectangle_coords['{}{}'.format(a, b)] = []


        return rectangle_coords

    def find_polygon_vertex(self, ax, ay, bx, by, max_val_x, max_val_y):
        a_b_dist = int(((ax - bx) ** 2 + (ay - by) ** 2) ** (1 / 2))

        # coeff = np.polyfit([ax, bx], [ay, by], 1)
        # mval = np.round(coeff[0], 3)
        if (ax-bx) != 0:
            mval = np.round((ay-by)/(ax-bx), 3)
            if mval != 0:
                mval = -(1/mval)
            else:
                mval = 100
        else:
            mval = 0

        urp = self.get_couple_xy(int(ax + a_b_dist / 2), mval, ax, ay, max_val_x, max_val_y)
        ulp = self.get_couple_xy(int(ax - a_b_dist / 2), mval, ax, ay, max_val_x, max_val_y)
        brp = self.get_couple_xy(int(bx + a_b_dist / 2), mval, bx, by, max_val_x, max_val_y)
        blp = self.get_couple_xy(int(bx - a_b_dist / 2), mval, bx, by, max_val_x, max_val_y)

        return [urp, ulp, brp, blp]

    def get_couple_xy(self, new_x_coord, m, x1, y1, max_x, max_y):

        new_x_coord = new_x_coord if new_x_coord > 0 else 0
        new_x_coord = new_x_coord if new_x_coord < max_x else max_x
        if m != 0:
            q = int(y1 - m * x1)
            new_y_coord = int(m * new_x_coord + q)
        else:
            new_y_coord = y1

        new_y_coord = new_y_coord if new_y_coord > 0 else 0
        new_y_coord = new_y_coord if new_y_coord < max_y else max_y

        return [new_x_coord, new_y_coord]

    def get_hmaps(self, h: int, w: int, sigma: float = None) -> (np.ndarray, np.ndarray):
        """
        returns 2 lists of HMaps (1 for visible joints and 1 for occluded joints)
        """
        hmaps_no_occl = []
        hmaps_only_occl = []
        for i in range(self[0].joints_number):
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

                if joint.is_occluded:
                    hmaps_only_occl.append(hmap)
                    hmaps_no_occl.append(empty_hmap)
                else:
                    hmaps_no_occl.append(hmap)
                    hmaps_only_occl.append(empty_hmap)

        return hmaps_no_occl, hmaps_only_occl

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
        for joint_type in range(pose1[0].joints_number):
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
