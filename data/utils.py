# Copyright 2021 Sensetime Yongqiang Yao, Tianzi Xiao
import torch
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    if get_world_size() > 1:
        x = torch.cuda.IntTensor([1])
        dist.all_reduce(x)
        x.cpu()
