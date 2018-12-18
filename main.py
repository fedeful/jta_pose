import argparse
import torch
import os
import shutil
from utils import path_config
from networks.generators import SimpleUnet
from make_frame import generate_sequence, make_video


def make_sequence(parsed):

    annotations = os.path.join(parsed.path, 'annotations', 'seq_{}.json'.format(parsed.name))
    backgrounds = os.path.join(parsed.path, 'backgrounds', '{}.jpeg'.format(parsed.name))
    if not os.path.isdir(os.path.join(parsed.path, 'frames')):
        os.mkdir(os.path.join(parsed.path, 'frames'), 0o777)
    output_path = os.path.join(parsed.path, 'frames', '{}'.format(parsed.name))

    gen = SimpleUnet(18, 3, 64, 0).cuda().eval()
    mask = SimpleUnet(15, 1, 64, 0).cuda().eval()

    checkpoint = torch.load('./person.checkpoint')
    gen.load_state_dict(checkpoint['model_generator'])

    checkpoint = torch.load('./mask.checkpoint')
    mask.load_state_dict(checkpoint['model_generator'])

    fm = path_config.PathMng(None, None)
    generate_sequence(annotations, output_path, backgrounds, fm, gen, mask)


def make_videos(parsed):
    input_frame_path = os.path.join(parsed.path, 'frames', '{}'.format(parsed.name))
    output_path = os.path.join(parsed.path, 'videos_generated')

    if not os.path.exists(output_path):
        os.makedirs(output_path, 0o777)

    output_file = os.path.join(output_path, '{}.avi'.format(parsed.name))

    data_type = (1280, 720)
    if parsed.video_type > 1:
        data_type = (1920, 1080)

    make_video(input_frame_path, output_file, data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/federico/Volume1/remote/datasets/jta_dataset/')
    parser.add_argument('--name', type=str)
    parser.add_argument('--video_type', type=int, default=1)
    parsed = parser.parse_args()

    if os.path.isdir('/tmp/jta'):
        shutil.rmtree('/tmp/jta')
    print('Begin Frames Generation')
    make_sequence(parsed)
    print('End Frames Generation')
    print('Begin Video Generation')
    make_videos(parsed)
    print('End video Generation')

    if os.path.isdir('/tmp/jta'):
        shutil.rmtree('/tmp/jta')









