import os
import cv2
import numpy as np
from utils.data_utils import imread_np


def gkern(h: int, w: int, center, s: float = 4):
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]

    x0 = center[0]
    y0 = center[1]

    return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)


class Annotation(object):
    _MAX_X = 1920
    _MAX_Y = 1080
    _PAD_X = 0.2
    _PAD_Y = 0.15
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

    JNAMES = [
        'Right Ankle',  # 0
        'Right Knee',  # 1
        'Right Hip',  # 2
        'Left Hip',  # 3
        'Left Knee',  # 4
        'Left Ankle',  # 5
        'Right Wrist',  # 6
        'Right Elbow',  # 7
        'Right Shoulder',  # 8
        'Left Shoulder',  # 9
        'Left Elbow',  # 10
        'Left Wrist',  # 11
        'Head - bottom',  # 12
        'Nose',  # 13
        'Head - top'  # 14
    ]

    JLIMBS = [
        (0, 1),  # right ankle -> right knee
        (1, 2),  # right knee -> right hip
        (2, 8),  # right hip -> right shoulder
        (8, 7),  # right shoulder -> right elbow
        (7, 6),  # right elbow -> right wrist
        (5, 4),  # left ankle -> left knee
        (4, 3),  # left knee -> left hip
        (3, 9),  # left hip -> left shoulder
        (9, 10),  # left shoulder -> left elbow
        (10, 11),  # left elbow -> left wrist
        (9, 12),  # left shoulder -> bottom head
        (8, 12),  # right shoulder -> bottom head
        (12, 13),  # bottom head -> nose
        (13, 14),  # nose -> top head
    ]
    JBODY = [
        (0, 1),  # right ankle -> right knee
        (1, 2),  # right knee -> right hip
        (2, 8),  # right hip -> right shoulder
        (8, 7),  # right shoulder -> right elbow
        (8, 14),  # right shoulder -> top head
        (9, 14),  # left shoulder -> top head
        (8, 13),  # right shoulder -> nose
        (9, 13),  # left shoulder -> nose
        (7, 6),  # right elbow -> right wrist
        (5, 4),  # left ankle -> left knee
        (4, 3),  # left knee -> left hip
        (3, 9),  # left hip -> left shoulder
        (2, 3),  # right hip -> left hip
        (3, 8),  # left hip -> right shoulder
        (2, 9),  # right hip -> left shoulder
        (9, 10),  # left shoulder -> left elbow
        (10, 11),  # left elbow -> left wrist
        (9, 12),  # left shoulder -> bottom head
        (8, 12),  # right shoulder -> bottom head
        (12, 13),  # bottom head -> nose
        (13, 14),  # nose -> top head
    ]

    # list of limbs
    # >> (A,B) is the limb that links joint of type A to joint of type B
    LIMBS = [
        (14, 13),  # head_top -> nose
        (13, 12),  # nose -> head_bottom
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
    N_JOINTS = len(JNAMES)

    # total number of limbs
    N_LIMBS = len(LIMBS)

    def __init__(self, jnotes):
        # order: x_min | x_max | y_min | y_max
        self.head = [jnotes['x1'][0], jnotes['x2'][0], jnotes['y1'][0], jnotes['y2'][0]]
        self.score = jnotes['score'][0]
        self.scale = jnotes['scale'][0]
        self.source_image = jnotes['image']

        self.joints = {}
        # order: id | x | y | visibility
        if len(jnotes['annopoints']) == 0:
            print('lunghezza zero')
        for joint in jnotes['annopoints'][0]['point']:
            self.joints['{}'.format(joint['id'][0])] = [joint['x'][0], joint['y'][0], joint['is_visible'][0]]

    def get_bbox(self, img_w, img_h, w_inc_perc=0.2, h_inc_perc=0.15):

        x_min = np.min([joint[0] for joint in self.joints.values()])
        x_max = np.max([joint[0] for joint in self.joints.values()])
        y_min = np.min([joint[1] for joint in self.joints.values()])
        y_max = np.max([joint[1] for joint in self.joints.values()])

        h = x_max - x_min
        w = y_max - y_min
        dw = int(round(w * w_inc_perc))
        dh = int(round(h * h_inc_perc))

        x_min = 0 if x_min - dw < 0 else x_min - dw
        x_max = img_w if x_max + dw > img_w else x_max + dw
        y_min = 0 if y_min - dh < 0 else y_min - dh
        y_max = img_h if y_max + dh > img_h else y_max + dh

        return int(round(x_min)), int(round(x_max)), int(round(y_min)), int(round(y_max))

    def get_heatmaps(self, h, w, sub_h, sub_w, scale_h, scale_w):
        heatmaps = []
        for i in range(len(Annotation.JNAMES)):
            if '{}'.format(i) not in self.joints.keys():
                hmap = np.zeros((h, w)).astype(np.float32)
                heatmaps.append(hmap)
            else:
                joint = self.joints['{}'.format(i)]
                if scale_h != 0:
                    cx = int(round((joint[0] - sub_w) * scale_w))
                    cy = int(round((joint[1] - sub_h) * scale_h))
                else:
                    cx = int(round((joint[0] - sub_w)))
                    cy = int(round((joint[1] - sub_h)))

                k = 1 if joint[2] else -1

                # heatmaps.append(k * gkern(h=h, w=w, s=12.5 / joint.cam_dist, center=(cx, cy)))
                heatmaps.append(k * gkern(h=h, w=w, s=12.5, center=(cx, cy)))
        return heatmaps

    def get_mask(self, h, w, sub_h, sub_w, scale_h, scale_w):
        jsar = {}
        mask = np.zeros((h, w)).astype(np.uint8)
        for i in range(len(Annotation.JNAMES)):
            if '{}'.format(i) not in self.joints.keys():
                jsar[i] = {'presence': False}
            else:
                joint = self.joints['{}'.format(i)]
                if scale_h != 0:
                    cx = int(round((joint[0] - sub_w) * scale_w))
                    cy = int(round((joint[1] - sub_h) * scale_h))
                else:
                    cx = int(round((joint[0] - sub_w)))
                    cy = int(round((joint[1] - sub_h)))
                jsar[i] = {
                    'presence': True,
                    'cx': cx,
                    'cy': cy
                }
        for i in Annotation.JBODY:
            if jsar[i[0]]['presence'] and jsar[i[1]]['presence']:
                cv2.line(mask,
                         (jsar[i[0]]['cx'], jsar[i[0]]['cy']),
                         (jsar[i[1]]['cx'], jsar[i[1]]['cy']),
                         255,
                         3)
                cv2.circle(
                    mask,
                    (jsar[i[0]]['cx'], jsar[i[0]]['cy']),
                    8,
                    255,
                    -1
                )
                cv2.circle(
                    mask,
                    (jsar[i[1]]['cx'], jsar[i[1]]['cy']),
                    8,
                    255,
                    -1
                )
                mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
                mask = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=1)
        return mask

    def get_image(self, frame_path):
        return imread_np(os.path.join(frame_path, self.source_image))
