# -*- coding: utf-8 -*-
# 🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐

from typing import *

import cv2
from numpy import ndarray as NPArray


class BBox(List[int]):

    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int, person_id: int = None):
        """
        * (x_min, y_min) = coordinates of the lower left corner
        * (x_max, y_max) = coordinates of the upper right corner
        """
        super(BBox, self).__init__([x_min, y_min, x_max, y_max])
        self.id = person_id

    @classmethod
    def init_from_torch(cls, t):
        """
        Returns a BBox from a list of 4 torch 1x1 tensors.
        """
        return BBox(
            x_min=int(t[0].numpy()[0]),
            y_min=int(t[1].numpy()[0]),
            x_max=int(t[2].numpy()[0]),
            y_max=int(t[3].numpy()[0]),
        )

    @classmethod
    def init_from_list(cls, t):
        """
        Returns a BBox from a list.
        """
        return BBox(
            x_min=int(t[0]),
            y_min=int(t[1]),
            x_max=int(t[2]),
            y_max=int(t[3]),
        )

    @property
    def area(self) -> float:
        """
        :return: area of the BBox
        """
        return float((self[2] - self[0]) * (self[3] - self[1]))

    # GETTER/SETTER for x_min
    @property
    def x_min(self) -> int:
        return self[0]

    @x_min.setter
    def x_min(self, new_value: int):
        self[0] = new_value

    # GETTER/SETTER for y_min
    @property
    def y_min(self) -> int:
        return self[1]

    @y_min.setter
    def y_min(self, new_value: int):
        self[1] = new_value

    # GETTER/SETTER for x_max
    @property
    def x_max(self) -> int:
        return self[2]

    @x_max.setter
    def x_max(self, new_value: int):
        self[2] = new_value

    # GETTER/SETTER for y_max
    @property
    def y_max(self) -> int:
        return self[3]

    @y_max.setter
    def y_max(self, new_value: int):
        self[3] = new_value

    @property
    def w(self) -> int:
        """
        :return: height of the BBox
        """
        return self.x_max - self.x_min

    @property
    def h(self) -> int:
        """
        :return: width of the BBox
        """
        return self.y_max - self.y_min

    def has_height_in_range(self, min_h: int = 100, max_h: int = 400) -> bool:
        """
        :return: 'True' if the bbox has a height in range [min_h, max_h], 'False' otherwise
        """
        height = self.h
        if (height >= min_h and height <= max_h):
            return True
        else:
            return False

    @property
    def is_empty(self) -> bool:
        """
        A BBox is considered empty if its height or width is <= 0
        """
        return self.w <= 0 or self.h <= 0

    def __and__(self, other: 'BBox') -> 'BBox':
        """
        Intersection operator between boxes.
        :example: (B1 & B2) is the intersection between B1 and B2
        :note: returns an empty BBox (BBox(0,0,0,0)) if the intersection is empty
        """
        bbox = BBox(
            max(self[0], other[0]),
            max(self[1], other[1]),
            min(self[2], other[2]),
            min(self[3], other[3]),
        )
        if bbox.is_empty:
            return BBox(0, 0, 0, 0)
        else:
            return bbox

    @staticmethod
    def get_union_area(bbox1: 'BBox', bbox2: 'BBox') -> float:
        """
        Returns the area of ​​the portion of space given by the
        union of the two required BBoxes
        """
        return bbox1.area + bbox2.area - (bbox1 & bbox2).area

    @staticmethod
    def iou(bbox1: 'BBox', bbox2: 'BBox') -> float:
        """
        Returns the intersection over union of the required BBoxes
        """
        return (bbox1 & bbox2).area / BBox.get_union_area(bbox1, bbox2)

    def draw_on_image(self, img: NPArray, color: Tuple[int, int, int] = (170, 225, 50), linewidth: int = 2) -> NPArray:
        """
        Draw the box on the required image

        :param img: image on which to draw the BBox
        :param color: color of the BBox rectangle
        :param linewidth: thickness BBox rectangle boundary
        :return: image with the BBox
        """
        return cv2.rectangle(img, (self.x_min, self.y_min), (self.x_max, self.y_max), color, linewidth)

    def __str__(self):
        return 'BBox.{}: {}'.format(self.id, super(BBox, self).__str__())
