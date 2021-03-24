from torch.utils.data import Dataset
import requests
from retrying import retry
import numpy as np
from .utils import get_rank, get_world_size, barrier
import math
import json


class BaseDataset(Dataset):
    def __init__(self, meta_file):
        super(BaseDataset, self).__init__()
        self.metas = self.parse(meta_file)

    def parse(self, meta_file):
        metas = []
        with open(meta_file) as f:
            for line in f.readlines():
                metas.append(line.strip())
        return metas

    def __getitem__(self, idx):
        return self.metas[idx]


class ServerDataset(BaseDataset):
    def __init__(self, meta_file, server_ip, server_port, timeout=1000):
        super(ServerDataset, self).__init__(meta_file)
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.meta_num = self.get_meta_num()

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def get_meta_num(self):
        meta_num = requests.get('http://{}:{}/get_len'.format(
            self.server_ip, self.server_port), timeout=self.timeout).json()
        return int(meta_num)

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def get_meta(self, idx):
        meta = requests.get('http://{}:{}/get/{}'.format(
            self.server_ip, self.server_port, idx), timeout=self.timeout).json()
        return meta


class RankDataset(BaseDataset):
    '''
    实际流程
    获取rank和world_size 信息 -> 获取dataset长度 -> 根据dataset长度产生随机indices ->
    给不同的rank 分配indices -> 根据这些indices产生metas
    '''

    def __init__(self, meta_file, is_test=False, reload_cfg=None):
        self.world_size = get_world_size()
        self.rank = get_rank()
        if reload_cfg is None:
            reload_cfg = {}
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 1)
        self.is_test = is_test
        super(RankDataset, self).__init__(meta_file)

    def count_dataset_size(self, file_name):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024 * 8
        with open(file_name) as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer)
                                              for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)

    def get_rank_indices(self, meta_file):
        dataset_size = self.count_dataset_size(meta_file)
        if self.is_test:
            return list(range(0, dataset_size)), dataset_size
        indices = self.get_group_random_indices(dataset_size)
        rank_num_samples = int(
            math.ceil(dataset_size * 1.0 / (self.world_size * self.mini_epoch)))
        total_size = rank_num_samples * self.world_size * self.mini_epoch
        indices += indices[:(total_size - len(indices))]
        offset = rank_num_samples * self.rank
        mini_epoch_offset_begin = self.mini_epoch_idx * rank_num_samples * self.world_size
        mini_epoch_offset_end = (self.mini_epoch_idx + 1) * \
            rank_num_samples * self.world_size
        rank_indices = indices[mini_epoch_offset_begin:
                               mini_epoch_offset_end][offset:offset + rank_num_samples]
        assert len(rank_indices) == rank_num_samples
        return rank_indices, rank_num_samples

    def get_group_random_indices(self, dataset_size):
        '''
        分组产生随机数，避免一次性产生带来的内存溢出问题（5亿会占用很大的内存）
        1. 先切分成group组，每组设置不同的随机种子
        2. 对不同组进行二次随机，进一步近似全局随机
        '''

        indices = []
        temp_indices = []
        mini_dataset_size = int(math.ceil(dataset_size * 1.0 / self.group))
        group = math.ceil(dataset_size * 1.0 / mini_dataset_size)
        last_size = dataset_size - mini_dataset_size * (group - 1)
        for i in range(group):
            if i <= group - 2:
                cur_size = mini_dataset_size
            else:
                cur_size = last_size
            np.random.seed(self.seed + i + 10000)
            _indices = np.random.permutation(cur_size).astype(np.int32)
            _indices += i * mini_dataset_size
            temp_indices.append(_indices.tolist())
        np.random.seed(self.seed + 10000)
        for i in np.random.permutation(group):
            indices.extend(temp_indices[i])
        return indices

    def read_from_buf(self, fileObj, lineSign):
        '''
        避免一次性读入所有文件，每次读入一个buf，减少内存占用
        '''
        buf = ""
        while True:
            lines = buf.split(lineSign)
            for line in lines[0:-1]:
                yield line
            buf_size = 1024 * 1024 * 8
            chunk = fileObj.readline(buf_size)
            if not chunk:
                break
            buf = chunk

    def _read(self, meta_file):
        data_lst = []
        rank_indices, rank_num_samples = self.get_rank_indices(meta_file)
        rank_indices = set(rank_indices)
        idx = 0
        with open(meta_file) as f:
            for line in self.read_from_buf(f, "\n"):
                if idx in rank_indices:
                    filename = line.rstrip()[:-2]
                    label = line.rstrip()[-1]
                    data_lst.append([filename, label, idx])
                idx += 1
        if len(rank_indices) != rank_num_samples:
            data_lst += data_lst[:(rank_num_samples - len(rank_indices))]
        return data_lst

    def parse(self, meta_file):
        '''
        parse meta_file
        return: metas
        '''
        metas = self._read(meta_file)
        return metas


class RankServerDataset(BaseDataset):
    '''
    server 中心化读取流程
    获取rank, world_size 信息 -> 获取reload_cfg(重新生成dataloader需要) ->
    获取dataset 长度 -> 向server端发送生成indices请求 -> 从server 端获取rank的dataset size ->
    分组从server 端获取metas

    '''

    def __init__(self,
                 meta_file,
                 server_ip,
                 server_port,
                 is_test=False,
                 reload_cfg={},
                 timeout=1000):
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.timeout = timeout
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 10)
        self.cache_size = reload_cfg.get('cache_size', 100000)
        self.is_test = is_test
        self.server_ip = server_ip
        self.server_port = server_port
        super(RankServerDataset, self).__init__(meta_file)

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def prepare_rank_indices(self):
        key_info = {
            "mini_epoch": self.mini_epoch,
            "mini_epoch_idx": self.mini_epoch_idx,
            "world_size": self.world_size,
            "seed": self.seed,
            "group": self.group
        }
        status = requests.get('http://{}:{}/set_rank_indices/{}'.format(
            self.server_ip, self.server_port, json.dumps(key_info)), timeout=self.timeout).json()
        assert status == 1

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def get_rank_metas(self, cur_idx):
        key_info = {
            "begin": cur_idx[0],
            "end": cur_idx[1],
            "rank": self.rank,
            "mini_epoch_idx": self.mini_epoch_idx,
            "world_size": self.world_size,
        }
        metas = requests.get('http://{}:{}/get_rank_metas/{}'.format(
            self.server_ip, self.server_port, json.dumps(key_info)), timeout=self.timeout).json()
        return metas

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def get_dataset_size(self):
        size = requests.get(
            'http://{}:{}/get_len'.format(self.server_ip, self.server_port), timeout=self.timeout).json()
        return int(size)

    @retry(stop_max_delay=10, stop_max_attempt_number=10)
    def get_rank_size(self):
        size = requests.get('http://{}:{}/get_rank_size/{}'.format(
            self.server_port, self.server_ip, self.rank), timeout=self.timeout).json()
        return int(size)

    def _read(self, meta_file):
        data_lst = []
        # dataset_size = self.get_dataset_size()
        if self.rank == 0:
            self.prepare_rank_indices()
        barrier()
        rank_size = self.get_rank_size()
        idx = 0
        num = rank_size // self.cache_size + 1
        for i in range(num):
            cur_idx = [i * self.cache_size, (i + 1) * self.cache_size]
            lines = self.get_rank_metas(cur_idx)
            for line in lines:
                filename = line.rstrip()[:-2]
                label = line.rstrip()[-1]
                data_lst.append([filename, label, idx])
                idx += 1
        return data_lst

    def parse(self, meta_file):
        '''
        parse meta_file
        return: metas
        '''
        metas = self._read(meta_file)
        return metas
