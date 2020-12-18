from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import glob
import os.path as osp
import random
import numpy as np
import pdb


class SYSU(object):
    data_root = './data/data/SYSU-MM01/'
    # data_root = './data/SYSU-MM01/'
    train_info_path = osp.join(data_root, 'exp/train_id.txt')
    val_info_path = osp.join(data_root, 'exp/val_id.txt')
    test_info_path = osp.join(data_root, 'exp/test_id.txt')

    def __init__(self, mode='all', shot='single', relabel=True):
        self._check_before_run()
        train_rgb_imgs, num_train_rgb_pid, num_train_rgb_imgs, \
        train_ir_imgs, num_train_ir_pid, num_train_ir_imgs, num_train_pid = self._process_train_data(relabel=relabel)

        gal_imgs, num_gal_pids, num_gal_imgs = self._process_gallery_data(mode=mode, shot=shot)
        query_imgs, num_query_pids, num_query_imgs = self._process_query_data(shot='all')
        # pdb.set_trace()

        print("==> SYSU MM01 loaded")
        print("Dataset statistics:")
        print("--------------------------------------------------")
        print("    subset      |      # ids    |    # imgs     ")
        print("-" * 50)
        print("    train       |      {:04d}     |     {:06d}    "
              .format(num_train_pid, num_train_ir_imgs + num_train_rgb_imgs))
        print('-'*50)
        print("  rgb   |   ir  | {:04d} | {:04d}   | {:06d} | {:06d} "
              .format(num_train_rgb_pid, num_train_ir_pid, num_train_rgb_imgs, num_train_ir_imgs))
        print('-' * 50)
        print("    gallery     |       {:04d}    |     {:06d}    "
              .format(num_gal_pids, num_gal_imgs))
        print('-' * 50)
        print("    query       |       {:04d}    |     {:06d}    "
              .format(num_query_pids, num_query_imgs))
        print("--------------------------------------------------")
        # pdb.set_trace()

        self.train_rgb_imgs = train_rgb_imgs
        self.train_ir_imgs = train_ir_imgs
        self.gallery_imgs = gal_imgs
        self.query_imgs = query_imgs
        self.num_train_rgb_pids = num_train_rgb_pid
        self.num_train_ir_pids = num_train_ir_pid
        self.num_gallery_pids = num_gal_pids
        self.num_query_pids = num_query_pids

    def _check_before_run(self):
        if not osp.exists(self.data_root):
            raise RuntimeError("'{}' is not available".format(self.data_root))
        if not osp.exists(self.train_info_path):
            raise RuntimeError("'{}' is not available".format(self.train_info_path))
        if not osp.exists(self.val_info_path):
            raise RuntimeError("'{}' is not available".format(self.val_info_path))
        if not osp.exists(self.test_info_path):
            raise RuntimeError("'{}' is not available".format(self.test_info_path))

    def _load_data_info(self, info_file):
        with open(info_file, 'r') as f:
            ids = f.read().splitlines()
            ids = ["%04d" % int(i) for i in ids[0].split(',')]
        return ids

    def _process_train_data(self, relabel=True):
        rgb_cams = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cams = ['cam3', 'cam6']
        rgb_imgs = []
        ir_imgs = []
        num_rgb_pid = 0
        num_ir_pid = 0
        num_rgb_imgs = 0
        num_ir_imgs = 0
        rgb_flag = False
        ir_flag = False

        id_train = self._load_data_info(self.train_info_path)
        id_val = self._load_data_info(self.val_info_path)
        id_all = list(set(id_train).union(set(id_val)))
        # id_train = np.unique(id_train)
        pid2label = {pid: label for label, pid in enumerate(sorted(id_all))}
        for id in sorted(id_all):
            for cam in rgb_cams:
                img_dir = osp.join(self.data_root, cam, id)
                if osp.isdir(img_dir):
                    rgb_flag = True
                    cur_files = sorted(glob.glob(img_dir+'/*.jpg'))
                    num_rgb_imgs += len(cur_files)
                    for img in cur_files:
                        if relabel:
                            pid = pid2label[id]
                        else:
                            pid = id
                        rgb_imgs.append((img, pid, int(cam[-1])))
            if rgb_flag:
                num_rgb_pid += 1
                rgb_flag = False

            for cam in ir_cams:
                img_dir = osp.join(self.data_root, cam, id)
                if osp.isdir(img_dir):
                    ir_flag = True
                    cur_files = sorted(glob.glob(img_dir+'/*.jpg'))
                    num_ir_imgs += len(cur_files)
                    for img in cur_files:
                        if relabel:
                            pid = pid2label[id]
                        else:
                            pid = id
                        ir_imgs.append((img, pid, int(cam[-1])))
            if ir_flag:
                num_ir_pid += 1
                ir_flag = False
        num_pid = len(id_all)
        return rgb_imgs, num_rgb_pid, num_rgb_imgs, ir_imgs, num_ir_pid, num_ir_imgs, num_pid

    def _process_gallery_data(self, mode='all', shot='all'):
        if mode == 'all':
            rgb_cams = ['cam1', 'cam2', 'cam4', 'cam5']
        elif mode == 'indoor':
            rgb_cams = ['cam1', 'cam2']
        else:
            raise KeyError("mode '{}' does not exist. Pls choose from 'all' and 'indoor'...".format(mode))
        print("executing {}-search {}-shot mode...".format(mode, shot))

        num_gal_imgs = 0
        num_gal_pids = 0
        gal_imgs = []

        id_gallery = self._load_data_info(self.test_info_path)
        for id in sorted(id_gallery):
            num_gal_pids += 1
            for cam in rgb_cams:
                img_dir = osp.join(self.data_root, cam, id)
                if osp.isdir(img_dir):
                    cur_files = sorted(glob.glob(img_dir + '/*.jpg'))
                    if shot == 'single':
                        num_gal_imgs += 1
                        gal_imgs.append((random.choice(cur_files), id, int(cam[-1])))
                    elif shot == 'multi':
                        num_gal_imgs += 10
                        gal_imgs.extend(list(zip(random.sample(cur_files, 10), [id] * 10, [int(cam[-1])] * 10)))
                    elif shot == 'all':
                        num_gal_imgs += len(cur_files)
                        gal_imgs.extend(list(zip(cur_files, [id] * len(cur_files), [int(cam[-1])] * len(cur_files))))
                    else:
                        raise KeyError("{}-shot mode does not support. Pls. choose from 'single', 'multi' and 'all'...")

        return gal_imgs, num_gal_pids, num_gal_imgs

    def _process_query_data(self, shot='all'):
        ir_cams = ['cam3', 'cam6']
        id_query = self._load_data_info(self.test_info_path)
        query_imgs = []
        num_query_imgs = 0
        num_query_pids = 0

        for id in sorted(id_query):
            num_query_pids += 1
            for cam in ir_cams:
                img_dir = osp.join(self.data_root, cam, id)
                if osp.isdir(img_dir):
                    cur_files = sorted(glob.glob(img_dir + '/*.jpg'))
                    if shot == 'single':
                        num_query_imgs += 1
                        query_imgs.append((random.choice(cur_files), id, int(cam[-1])))
                    elif shot == 'multi':
                        num_query_imgs += 10
                        query_imgs.extend(list(zip(random.sample(cur_files, 10), [id] * 10, [int(cam[-1])] * 10)))
                    elif shot == 'all':
                        num_query_imgs += len(cur_files)
                        query_imgs.extend(list(zip(cur_files, [id] * len(cur_files), [int(cam[-1])] * len(cur_files))))
                    else:
                        raise KeyError("{}-shot mode does not support. Pls choose from 'single', 'multi' and 'all'...")
        return query_imgs, num_query_pids, num_query_imgs


class REGDB(object):
    data_root = './data/dataset/RegDB/'

    def __init__(self, trial=1):
        self.train_rgb_info_path = osp.join(self.data_root, 'idx/train_visible_{}.txt'.format(trial))
        self.train_ir_info_path = osp.join(self.data_root, 'idx/train_thermal_{}.txt'.format(trial))
        self.test_rgb_info_path = osp.join(self.data_root, 'idx/test_visible_{}.txt'.format(trial))
        self.test_ir_info_path = osp.join(self.data_root, 'idx/test_thermal_{}.txt'.format(trial))
        self._check_before_run()
        train_rgb_imgs, num_train_rgb_pids, num_train_rgb_imgs = self._process_data(self.train_rgb_info_path)
        train_ir_imgs, num_train_ir_pids, num_train_ir_imgs = self._process_data(self.train_ir_info_path)
        gal_imgs, num_gal_pids, num_gal_imgs = self._process_data(self.test_rgb_info_path)
        query_imgs, num_query_pids, num_query_imgs = self._process_data(self.test_ir_info_path)

        print("==> RegDB Dataset loaded")
        print("Dataset statistics:")
        print("--------------------------------------------------")
        print("    subset      |      # ids    |    # imgs     ")
        print("-" * 50)
        print("    train       |      {:04d}     |     {:06d}    "
              .format(num_train_rgb_pids, num_train_ir_imgs + num_train_rgb_imgs))
        print('-' * 50)
        print("  rgb   |   ir  | {:04d} | {:04d}   | {:06d} | {:06d} "
              .format(num_train_rgb_pids, num_train_ir_pids, num_train_rgb_imgs, num_train_ir_imgs))
        print('-' * 50)
        print("    gallery     |       {:04d}    |     {:06d}    "
              .format(num_gal_pids, num_gal_imgs))
        print('-' * 50)
        print("    query       |       {:04d}    |     {:06d}    "
              .format(num_query_pids, num_query_imgs))
        print("--------------------------------------------------")

        self.train_rgb_imgs = train_rgb_imgs
        self.train_ir_imgs = train_ir_imgs
        self.gallery_imgs = gal_imgs
        self.query_imgs = query_imgs
        self.num_train_rgb_pids = num_train_rgb_pids
        self.num_train_ir_pids = num_train_ir_pids
        self.num_gallery_pids = num_gal_pids
        self.num_query_pids = num_query_pids

    def _check_before_run(self):
        if not osp.exists(self.data_root):
            raise RuntimeError("'{}' is not available".format(self.data_root))
        if not osp.exists(self.train_rgb_info_path):
            raise RuntimeError("'{}' is not available".format(self.train_rgb_info_path))
        if not osp.exists(self.train_ir_info_path):
            raise RuntimeError("'{}' is not available".format(self.train_ir_info_path))
        if not osp.exists(self.test_rgb_info_path):
            raise RuntimeError("'{}' is not available".format(self.test_rgb_info_path))
        if not osp.exists(self.test_ir_info_path):
            raise RuntimeError("'{}' is not available".format(self.test_ir_info_path))

    def _process_data(self, info_file):
        cam = info_file.split('_')[1]
        imgs = []
        pids = []
        with open(info_file, 'rt') as f:
            data_file_list = f.read().splitlines()
            if cam == 'visible':
                camid = 1
            elif cam == 'thermal':
                camid = 2
            else:
                raise RuntimeError("camera type '{}' does not support...".format(cam))
            for file in data_file_list:
                imgs.append((file.split(' ')[0], int(file.split(' ')[1]), camid))
                pids.append(int(file.split(' ')[1]))
            num_imgs = len(data_file_list)
            num_pids = len(np.unique(pids))
            return imgs, num_pids, num_imgs


__factory = {
    'sysu': SYSU,
    'regdb': REGDB
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknow dataset: {}".format(name))
    return __factory[name](*args, **kwargs)


# if __name__ == "__main__":
#     dataset = init_dataset(name='sysu', shot='all')