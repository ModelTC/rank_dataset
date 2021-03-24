# Copyright 2021 Sensetime Yongqiang Yao, Tianzi Xiao
from __future__ import division

# Standard Library
import argparse
import json  # noqa F402
import os
import random  # noqa F402
import numpy as np

# Import from third library
from flask import Flask  # noqa F402
import math


# from gevent import monkey
# monkey.patch_all()

parser = argparse.ArgumentParser(description='Server')

parser.add_argument(
    '--meta_file',
    dest='meta_file',
    default='',
    help='meta file for training')
parser.add_argument(
    '--port',
    default=28889,
    type=int,
    help='server port')


app = Flask(__name__)

global data_lst
data_lst = []
global indices
indices = []
global rank_num_samples
rank_num_samples = 0
global is_init
is_init = False


@app.route('/get/<key>')
def get(key):
    meta = data_lst[int(key)]
    return json.dumps(meta)


@app.route('/set_rank_indices/<key>')
def set_rank_indices(key):
    decode_key = json.loads(key)
    seed = decode_key.get('seed', 0)
    group = decode_key.get('group', 10)
    world_size = decode_key.get("world_size", 8)
    mini_epoch = decode_key.get('mini_epoch', 1)
    mini_epoch_idx = decode_key.get('mini_epoch_idx', 0)
    global indices
    if len(indices) > 0 and mini_epoch_idx > 0:
        return json.dumps(1)
    get_group_random_indices(len(data_lst), group=group, seed=seed)
    prepare_all_rank_indices(world_size, mini_epoch, mini_epoch_idx)
    global is_init
    is_init = True
    return json.dumps(1)


@app.route('/get_rank_size/<key>')
def get_rank_size(key):
    global rank_num_samples
    return json.dumps(rank_num_samples)


@app.route('/get_rank_metas/<key>')
def get_rank_metas(key):
    global is_init, indices
    decode_key = json.loads(key)
    rank = decode_key.get('rank', 0)
    begin = decode_key.get('begin', 0)
    end = decode_key.get('end', 0)
    mini_epoch_idx = decode_key.get('mini_epoch_idx', 0)
    world_size = decode_key.get("world_size", 8)
    if not is_init:
        world_size = decode_key.get("world_size", 8)
        mini_epoch = decode_key.get('mini_epoch', 1)
        mini_epoch_idx = decode_key.get('mini_epoch_idx', 0)
        prepare_all_rank_indices(world_size, mini_epoch, mini_epoch_idx)
    metas = []
    begin_idx, end_idx = get_cur_index(
        rank, begin, end, mini_epoch_idx, world_size)
    for i in range(begin_idx, end_idx):
        metas.append(data_lst[indices[i]])
    return json.dumps(metas)


@app.route('/get_len')
def get_len():
    return json.dumps(len(data_lst))


def get_cur_index(rank, begin, end, mini_epoch_idx, world_size):
    global rank_num_samples
    mini_epoch_index = mini_epoch_idx * rank_num_samples * world_size
    rank_index = rank_num_samples * rank
    cur_begin_idx = max(0, begin + mini_epoch_index + rank_index)
    cur_end_idx = min(end + mini_epoch_index + rank_index,
                      mini_epoch_index + rank_index + rank_num_samples)
    return cur_begin_idx, cur_end_idx


def get_group_random_indices(dataset_size, group=10, seed=1):
    '''
    1. random dataset_size / group for each sub group
    2. random group
    3. extend final indices
    '''
    mini_dataset_size = int(math.ceil(dataset_size * 1.0 / group))
    group = math.ceil(dataset_size * 1.0 / mini_dataset_size)
    last_size = dataset_size - mini_dataset_size * (group - 1)
    global indices
    indices = []
    temp_indices = []
    for i in range(group):
        if i <= group - 2:
            cur_size = mini_dataset_size
        else:
            cur_size = last_size
        np.random.seed(i + int(seed) + 10000)
        _indices = np.random.permutation(cur_size).astype(np.int32)
        _indices += i * mini_dataset_size
        temp_indices.append(_indices.tolist())
    np.random.seed(int(seed) + 10000)
    for i in np.random.permutation(group):
        indices.extend(temp_indices[i])
    return indices


def prepare_all_rank_indices(world_size, mini_epoch, mini_epoch_idx):
    global indices, rank_num_samples
    dataset_size = len(data_lst)
    rank_num_samples = int(
        math.ceil(dataset_size * 1.0 / (world_size * mini_epoch)))
    total_size = rank_num_samples * world_size * mini_epoch
    indices = indices + indices[:(total_size - len(indices))]


def get_meta(meta_file):
    with open(meta_file) as f:
        for line in f.readlines():
            data_lst.append(line.strip())


if __name__ == '__main__':
    args = parser.parse_args()
    get_meta(args.meta_file)
    os.system('ifconfig')
    port = args.port
    app.run('0.0.0.0', port, threaded=True)
