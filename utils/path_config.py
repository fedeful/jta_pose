import os
import socket

from utils.folder_utils import check_generate_dir

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)


class PathMng(object):
    HOSTNAME = socket.gethostname()
    h_name = 'CURRENT HOSTNAME: {} '.format(HOSTNAME)
    print(h_name)

    if HOSTNAME == 'vegeta':
        # DS_PATH = '/majinbu/public/capra'
        # PJ_PATH = '/home/federico/PycharmProjects/Ped_Gen'
        # LG_PATH = '/majinbu/public/federico/pedgenlog'
        DS_PATH = '/homes/ffulgeri'
        PJ_PATH = '/homes/ffulgeri/PycharmProjects/Ped_Gen'
        LG_PATH = '/homes/ffulgeri/results'
    elif HOSTNAME == 'DESKTOP-LN3FIJI':
        DS_PATH = 'C:/Users/ffulg/Desktop'
        PJ_PATH = 'C:/Users/ffulg/PycharmProjects/Ped_Gen'
        LG_PATH = 'C:/Users/ffulg/Desktop/pedgenlog'
    elif HOSTNAME == 'leonida':
        DS_PATH = '/media/federico/Volume1/remote'
        PJ_PATH = '/home/federico/PycharmProjects/Ped_Gen'
        LG_PATH = '/home/federico/Desktop/pedgenlog'
    elif HOSTNAME.startswith('aimagelab-srv'):
        DS_PATH = '/homes/ffulgeri'
        PJ_PATH = '/homes/ffulgeri/PycharmProjects/Ped_Gen'
        LG_PATH = '/homes/ffulgeri/results'
        #LG_PATH = '/homes/ffulgeri/debres'
    elif HOSTNAME.startswith('node'):
        DS_PATH = '/gpfs/work/IscrC_DeepCA_1/ffulgeri'
        PJ_PATH = '/galileo/home/userexternal/ffulgeri/Ped_Gen'
        LG_PATH = '/gpfs/work/IscrC_DeepCA_1/ffulgeri/results'
    else:
        DS_PATH = '/homes/ffulgeri'
        PJ_PATH = '/homes/ffulgeri/PycharmProjects/Ped_Gen'
        LG_PATH = '/homes/ffulgeri/results'

    def __init__(self, experiment_name, global_name):

        self.pose_track_path = os.path.join(PathMng.DS_PATH, 'Datasets/PoseTrack/posetrack_data')
        self.pose_track_train = os.path.join(PathMng.PJ_PATH, 'dataset/pt_train')
        self.pose_track_test = os.path.join(PathMng.PJ_PATH, 'dataset/pt_validation')

        # self.market_train_list = os.path.join(PathMng.PJ_PATH,
        #                                       'datasets/market1501/configuration/train_couple_list_cl.pickle')
        # self.market_train_pose = os.path.join(PathMng.PJ_PATH,
        #                                       'datasets/market1501/configuration/train_coordinates_list_cl.pickle')
        # self.market_test_list = os.path.join(PathMng.PJ_PATH,
        #                                      'datasets/market1501/configuration/test_couple_list_cl.pickle')
        # self.market_test_pose = os.path.join(PathMng.PJ_PATH,
        #                                      'datasets/market1501/configuration/test_coordinates_list_cl.pickle')

        self.market_train_list = os.path.join(PathMng.PJ_PATH,
                                              'datasets/market1501/configuration/train_couple_list_final.pickle')
        self.market_train_pose = os.path.join(PathMng.PJ_PATH,
                                              'datasets/market1501/configuration/train_coordinates_list_final.pickle')
        self.market_test_list = os.path.join(PathMng.PJ_PATH,
                                             'datasets/market1501/configuration/test_couple_list_final.pickle')
        self.market_test_pose = os.path.join(PathMng.PJ_PATH,
                                             'datasets/market1501/configuration/test_coordinates_list_final.pickle')
        self.market_data_train = os.path.join(PathMng.DS_PATH, 'Datasets/Market-1501-v15.09.15/bounding_box_train')
        self.market_data_test = os.path.join(PathMng.DS_PATH, 'Datasets/Market-1501-v15.09.15/bounding_box_test')

        self.experiment_results = os.path.join(PathMng.LG_PATH, '{}'.format(experiment_name))
        self.global_results = os.path.join(PathMng.LG_PATH, '{}'.format(global_name))

    def folders_initialization(self):
        check_generate_dir(self.experiment_results)
        check_generate_dir(self.global_results)
