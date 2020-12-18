from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import pdb


class RandomIdentitySampler(Sampler):
    """Random sample N identities each with K instances."""
    def __init__(self, rgb_source, ir_source, batch_pid, batch_instance=1):
        self.batch_size = batch_pid
        self.batch_instance = batch_instance

        self.rgb_index_dic = defaultdict(list)
        self.ir_index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(rgb_source):
            self.rgb_index_dic[pid].append(index)

        for index, (_, pid, _) in enumerate(ir_source):
            self.ir_index_dic[pid].append(index)

        self.pids = list(self.rgb_index_dic.keys())
        self.num_pids = len(self.pids)
        # self.num_spls = np.maximum(len(rgb_source), len(ir_source))

    def __iter__(self):
        indices = torch.randperm(self.num_pids)
        ret = []
        for i in indices:
            pid = self.pids[i]
            temp_rgb = self.rgb_index_dic[pid]
            temp_ir = self.ir_index_dic[pid]
            rgb_replace = False if len(temp_rgb) >= self.batch_instance else True
            ir_replace = False if len(temp_ir) >= self.batch_instance else True
            temp_rgb = np.random.choice(temp_rgb, size=self.batch_instance, replace=rgb_replace)
            temp_ir = np.random.choice(temp_ir, size=self.batch_instance, replace=ir_replace)
            temp = zip(temp_rgb, temp_ir)
            ret.extend(list(temp))
        return iter(ret)

    def __len__(self):
        return self.num_pids * self.batch_instance


class IdentitySampler(Sampler):
    """
    Sample person identities evenly in each batch
    """
    def __init__(self, rgb_source, ir_source, batch_size, batch_instance=1):
        self.batch_size = batch_size
        self.batch_instance = batch_instance
        self.rgb_index_dic = defaultdict(list)
        self.ir_index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(rgb_source):
            self.rgb_index_dic[pid].append(index)

        for index, (_, pid, _) in enumerate(ir_source):
            self.ir_index_dic[pid].append(index)

        self.pids = list(self.rgb_index_dic.keys())
        self.num_spls = np.maximum(len(rgb_source), len(ir_source))
        self.num_batch = self.num_spls//self.batch_size + 1
        self.batch_num_ids = self.batch_size//self.batch_instance

    def __iter__(self):
        ret = []
        for j in range(self.num_batch):
            batch_ids = np.random.choice(self.pids, self.batch_num_ids, replace=False)
            for i in range(self.batch_num_ids):
                temp_rgb = self.rgb_index_dic[batch_ids[i]]
                temp_ir = self.ir_index_dic[batch_ids[i]]

                rgb_replace = False if len(temp_rgb) >= self.batch_instance else True
                ir_replace = False if len(temp_ir) >= self.batch_instance else True
                temp_rgb = np.random.choice(temp_rgb, size=self.batch_instance, replace=rgb_replace)
                temp_ir = np.random.choice(temp_ir, size=self.batch_instance, replace=ir_replace)
                temp = zip(temp_rgb, temp_ir)
                ret.extend(list(temp))
        return iter(ret)

    def __len__(self):
        # return self.num_batch * self.batch_instance
        return self.num_spls



# class OneIdentitySampler(Sampler):
#     """
#     Sample person identities evenly in each batch
#     """
#     def __init__(self, rgb_source, ir_source, batch_size):
#         self.batch_size = batch_size
#         self.rgb_index_dic = defaultdict(list)
#         self.ir_index_dic = defaultdict(list)
#         for index, (_, pid, _) in enumerate(rgb_source):
#             self.rgb_index_dic[pid].append(index)
#
#         for index, (_, pid, _) in enumerate(ir_source):
#             self.ir_index_dic[pid].append(index)
#
#         self.pids = list(self.rgb_index_dic.keys())
#         self.num_spls = np.maximum(len(rgb_source), len(ir_source))
#         self.num_batch = self.num_spls//self.batch_size + 1
#
#     def __iter__(self):
#         ret = []
#         for j in range(self.num_batch):
#             batch_ids = np.random.choice(self.pids, self.batch_size, replace=False)
#
#             for i in range(self.batch_size):
#                 rgb_sample = np.random.choice(self.rgb_index_dic[batch_ids[i]], 1)
#                 ir_sample = np.random.choice(self.ir_index_dic[batch_ids[i]], 1)
#                 ret.append(list(zip(rgb_sample, ir_sample)))
#         # print(ret)
#         return iter(ret)
#
#     def __len__(self):
#         return self.num_spls
#         # return self.num_batch * self.batch_size
#
#
# if __name__ == "__main__":
#     from data_manager import init_dataset
#     dataset = init_dataset('sysu')
#     sampler = OneIdentitySampler(dataset.train_rgb_imgs, dataset.train_ir_imgs, batch_size=64)
#     count = 0
#     pdb.set_trace()
#     for idx in sampler:
#         print(idx)
#         count += 1
#     print(count)