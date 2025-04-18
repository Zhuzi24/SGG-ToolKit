

import math
import os
import random
import sys
import time
from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, kl_divergence


class ExponentialMovingAverage(nn.Module):
    shadow: torch.Tensor

    def __init__(self, initValue, decay):
        super().__init__()
        if initValue is None:
            self.shadow = None
        else:
            self.register_buffer("shadow", initValue.clone().detach())
        self.decay = decay

    def forward(self, x):
        if self.shadow is None:
            self.register_buffer("shadow", x.clone().detach())
            return self.shadow
        self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
        return self.shadow

    @property
    def Value(self):
        return self.shadow


class NetworkExponentialMovingAverage(nn.Module):
    def __init__(self, network, decay):
        super().__init__()
        for (name, weights) in network.named_parameters():
            self.register_buffer(name.replace('.', ''), weights.clone().detach())
        self.decay = decay

    def forward(self, network):
        for (name, weights) in network.named_parameters():
            shadow = getattr(self, name.replace('.', ''))
            shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
            weights.data.copy_(shadow)
        return network


class ExponentialMovingAverage(nn.Module):
    shadow: torch.Tensor
    """
    updated sub prototype codebook
    """

    def __init__(self, init_value, decay):
        super().__init__()
        if init_value is None:
            self.shadow = None
        else:
            self.register_buffer("shadow", init_value.clone().detach())
        self.decay = decay

    def forward(self, x):
        if self.shadow is None:
            self.register_buffer("shadow", x.clone().detach())
            return self.shadow
        self.shadow.copy_((1 - self.decay) * self.shadow + self.decay * x)
        return self.shadow

    @property
    def Value(self):
        return self.shadow


class NetworkExponentialMovingAverage(nn.Module):
    def __init__(self, network, decay):
        super().__init__()
        for (name, weights) in network.named_parameters():
            self.register_buffer(name.replace('.', ''), weights.clone().detach())
        self.decay = decay

    def forward(self, network):
        for (name, weights) in network.named_parameters():
            shadow = getattr(self, name.replace('.', ''))
            shadow.copy_((1 - self.decay) * shadow + self.decay * weights)
            weights.data.copy_(shadow)
        return network


def cal_kl(distance):
    """
    计算 KL 散度
    Args:
        distance (Tensor): dist
        k (Tensor): K value

    Returns:
        Tensor: kl distance
    """
    a = Categorical(logits=distance) #[N,K]
    b = Categorical(probs=torch.ones([distance.shape[0],distance.shape[1]]).cuda() / distance.shape[1])
    loss = kl_divergence(a, b).mean()

    return loss


# refer to https://github.com/stanis-morozov/unq/blob/e8f7f43699c74be415732d914b01662ce3f60612/lib/quantizer.py#L197
def gumbel_variance(ips, p, alpha, M, square_cv=True):
    """
    计算协方差满足分布

    refer to Unsupervised Neural Quantization for Compressed-Domain Similarity Search

    Args:
        ips (_type_): _description_
        p (_type_): _description_
        alpha (_type_): _description_
        M (_type_): _description_
        square_cv (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    codes = F.gumbel_softmax(ips / alpha * M, dim=-1)  # gumbel-softmax logits
    load = torch.mean(p, dim=0)  # [..., codebook_size]
    mean = load.mean()
    variance = torch.mean((load - mean) ** 2)
    if square_cv:
        counters['cv_squared'] = variance / (mean ** 2 + eps)
        counters['reg'] += cv_coeff * counters['cv_squared']
    else:
        counters['cv'] = torch.sqrt(variance + eps) / (mean + eps)
        counters['reg'] += cv_coeff * counters['cv']

    return counters


# start uniform hypersphere
def int_sin_m(x: float, m: int) -> float:
    """
    Computes the integral of sin^m(t) dt from 0 to x recursively
    """
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


def primes() -> Iterator[int]:
    """
    Returns an infinite generator of prime numbers
    """
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10,
) -> float:
    """
    Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def uniform_hypersphere(d: int, n: int) -> List[List[float]]:  # 初始化的点>>要用的点，上下左右的点距离都比较近
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points


# end uniform hypersphere

def _cosine_similarity(codebook):
    inner_product = codebook @ codebook.T
    norm = (codebook ** 2).sum(-1).sqrt()
    return inner_product / (norm[:, None] * norm)

def get_semantic_diversity():
    freq_dict = {
        "parked alongside with": 3,
        "parking in the different apron with": 3,
        "parallelly parked on": 3,
        "parking in the same apron with": 3,
        "over": 3,
        "in the same parking with": 3,
        "connect": 3,
        "not co-storage with": 1,
        "driving in the same direction with":3,
        "parallelly docked at": 3,
        "co-storage with": 1,
        "intersect": 3,
        "within safe distance of": 3,
        "docking at the same dock with": 3,
        "driving in the same lane with": 3,
        "driving in the different lane with": 3,
        "converge": 3,
        "docked alongside with": 3,
        "within same line of": 1,
        "adjacent": 3,
        "approach":3,
        "within danger distance of": 3,
        "in the different parking with": 3,
        "not parked alongside with": 3,
        "docking at the different dock with": 3,
        "driving in the opposite direction with": 3,
        "away from": 3,
        "within different line of": 1,
        "through": 3,
        "randomly parked on": 3,
        "not docked alongside with": 3,
        "pass across": 3,
        "driving alongside with": 3,
        "randomly docked at": 1,
        "working on": 2,
        "directly connected to": 3,
        "isolatedly parked on": 3,
        "running along the different taxiway with": 1,
        "run along": 1,
        "around": 1,
        "drive toward": 1,
        "drive off": 1,
        "isolatedly docked at": 2,
        "incorrectly parked on": 2,
        "directly transmit electricity to": 1,
        "supply to": 2,
        "running along the same taxiway with": 1,
        "exhaust to": 2,
        "violently emit": 2,
        "not run along": 2,
        "slightly emit": 2,
        "docking at the same breakwater with": 1,
        "not working on": 1,
        "indirectly connected to": 1,
        "pass through": 1,
        "indirectly transmit electricity to": 1,
        "pass under": 1,
         "running along the different runway with": 1}

    pred2idx = {'parallelly docked at': 1, 
        'isolatedly docked at': 2,
        'connect': 3,
        'over': 4, 
        'co-storage with': 5, 
        'within safe distance of': 6,
        'randomly docked at': 7, 
        'docking at the same dock with': 8,
        'docked alongside with': 9, 
        'docking at the different dock with': 10, 
        'driving in the same direction with': 11,
        'parallelly parked on': 12, 
        'isolatedly parked on': 13,
        'randomly parked on': 14,
        'run along': 15, 
        'adjacent': 16, 
        'through': 17, 
        'converge': 18, 
        'intersect': 19, 
        'not run along': 20, 
        'parking in the same apron with': 21, 
        'parking in the different apron with': 22, 
        'parked alongside with': 23, 
        'not parked alongside with': 24, 
        'running along the different taxiway with': 25,
        'around': 26, 
        'not co-storage with': 27, 
        'running along the same taxiway with': 28, 
        'approach': 29,
        'away from': 30, 
        'within danger distance of': 31, 
        'incorrectly parked on': 32, 
        'in the same parking with': 33, 
        'in the different parking with': 34, 
        'not docked alongside with': 35, 
        'driving in the opposite direction with': 36, 
        'driving in the different lane with': 37, 
        'driving in the same lane with': 38, 
        'docking at the same breakwater with': 39, 
        'driving alongside with': 40,
        'running along the different runway with': 41,
        'violently emit': 42, 
        'exhaust to': 43,
        'slightly emit': 44,
        'supply to': 45, 
        'drive toward': 46, 
        'pass across': 47, 
        'drive off': 48,
        'pass under': 49,
        'within different line of': 50, 
        'within same line of': 51, 
        'directly connected to': 52, 
        'indirectly connected to': 53, 
        'pass through': 54,
        'directly transmit electricity to': 55, 
        'working on': 56, 
        'not working on': 57, 
        'indirectly transmit electricity to': 58
    }
    pred_list = [0] * 59
    pred_list[0] = 1
    for k, v in freq_dict.items():
        pred_list[int(pred2idx[k])] = v

    return pred_list

def get_semantic_diversity1():
    freq_dict = {
        "parked alongside with": 19,
        "parking in the different apron with": 4,
        "parallelly parked on": 17,
        "parking in the same apron with": 3,
        "over": 33,
        "in the same parking with": 12,
        "connect": 26,
        "not co-storage with": 1,
        "driving in the same direction with":9,
        "parallelly docked at": 12,
        "co-storage with": 1,
        "intersect": 11,
        "within safe distance of": 16,
        "docking at the same dock with": 6,
        "driving in the same lane with": 6,
        "driving in the different lane with": 5,
        "converge": 7,
        "docked alongside with": 8,
        "within same line of": 1,
        "adjacent": 20,
        "approach":24,
        "within danger distance of": 6,
        "in the different parking with": 9,
        "not parked alongside with": 4,
        "docking at the different dock with": 7,
        "driving in the opposite direction with": 9,
        "away from": 23,
        "within different line of": 1,
        "through": 4,
        "randomly parked on": 5,
        "not docked alongside with": 4,
        "pass across": 9,
        "driving alongside with": 5,
        "randomly docked at": 6,
        "working on": 7,
        "directly connected to": 8,
        "isolatedly parked on": 10,
        "running along the different taxiway with": 1,
        "run along": 4,
        "around": 8,
        "drive toward": 4,
        "drive off": 6,
        "isolatedly docked at": 7,
        "incorrectly parked on": 9,
        "directly transmit electricity to": 7,
        "supply to": 7,
        "running along the same taxiway with": 1,
        "exhaust to": 2,
        "violently emit": 2,
        "not run along": 2,
        "slightly emit": 2,
        "docking at the same breakwater with": 3,
        "not working on": 5,
        "indirectly connected to": 4,
        "pass through": 1,
        "indirectly transmit electricity to": 3,
        "pass under": 2,
         "running along the different runway with": 1}

    pred2idx = {'parallelly docked at': 1, 
        'isolatedly docked at': 2,
        'connect': 3,
        'over': 4, 
        'co-storage with': 5, 
        'within safe distance of': 6,
        'randomly docked at': 7, 
        'docking at the same dock with': 8,
        'docked alongside with': 9, 
        'docking at the different dock with': 10, 
        'driving in the same direction with': 11,
        'parallelly parked on': 12, 
        'isolatedly parked on': 13,
        'randomly parked on': 14,
        'run along': 15, 
        'adjacent': 16, 
        'through': 17, 
        'converge': 18, 
        'intersect': 19, 
        'not run along': 20, 
        'parking in the same apron with': 21, 
        'parking in the different apron with': 22, 
        'parked alongside with': 23, 
        'not parked alongside with': 24, 
        'running along the different taxiway with': 25,
        'around': 26, 
        'not co-storage with': 27, 
        'running along the same taxiway with': 28, 
        'approach': 29,
        'away from': 30, 
        'within danger distance of': 31, 
        'incorrectly parked on': 32, 
        'in the same parking with': 33, 
        'in the different parking with': 34, 
        'not docked alongside with': 35, 
        'driving in the opposite direction with': 36, 
        'driving in the different lane with': 37, 
        'driving in the same lane with': 38, 
        'docking at the same breakwater with': 39, 
        'driving alongside with': 40,
        'running along the different runway with': 41,
        'violently emit': 42, 
        'exhaust to': 43,
        'slightly emit': 44,
        'supply to': 45, 
        'drive toward': 46, 
        'pass across': 47, 
        'drive off': 48,
        'pass under': 49,
        'within different line of': 50, 
        'within same line of': 51, 
        'directly connected to': 52, 
        'indirectly connected to': 53, 
        'pass through': 54,
        'directly transmit electricity to': 55,
        'indirectly transmit electricity to': 56,
        'working on': 57, 
        'not working on': 58         
    }

    pred_list = [0] * 59
    pred_list[0] = 1
    for k, v in freq_dict.items():
        pred_list[int(pred2idx[k])] = 1

    return pred_list

def get_semantic_diversity222():
    freq_dict = {
        "parked alongside with": 19,
        "parking in the different apron with": 4,
        "parallelly parked on": 17,
        "parking in the same apron with": 3,
        "over": 33,
        "in the same parking with": 12,
        "connect": 26,
        "not co-storage with": 1,
        "driving in the same direction with":9,
        "parallelly docked at": 12,
        "co-storage with": 1,
        "intersect": 11,
        "within safe distance of": 16,
        "docking at the same dock with": 6,
        "driving in the same lane with": 6,
        "driving in the different lane with": 5,
        "converge": 7,
        "docked alongside with": 8,
        "within same line of": 1,
        "adjacent": 20,
        "approach":24,
        "within danger distance of": 6,
        "in the different parking with": 9,
        "not parked alongside with": 4,
        "docking at the different dock with": 7,
        "driving in the opposite direction with": 9,
        "away from": 23,
        "within different line of": 1,
        "through": 4,
        "randomly parked on": 5,
        "not docked alongside with": 4,
        "pass across": 9,
        "driving alongside with": 5,
        "randomly docked at": 6,
        "working on": 7,
        "directly connected to": 8,
        "isolatedly parked on": 10,
        "running along the different taxiway with": 1,
        "run along": 4,
        "around": 8,
        "drive toward": 4,
        "drive off": 6,
        "isolatedly docked at": 7,
        "incorrectly parked on": 9,
        "directly transmit electricity to": 7,
        "supply to": 7,
        "running along the same taxiway with": 1,
        "exhaust to": 2,
        "violently emit": 2,
        "not run along": 2,
        "slightly emit": 2,
        "docking at the same breakwater with": 3,
        "not working on": 5,
        "indirectly connected to": 4,
        "pass through": 1,
        "indirectly transmit electricity to": 3,
        "pass under": 2,
         "running along the different runway with": 1}

    pred2idx = {'parallelly docked at': 1, 
        'isolatedly docked at': 2,
        'connect': 3,
        'over': 4, 
        'co-storage with': 5, 
        'within safe distance of': 6,
        'randomly docked at': 7, 
        'docking at the same dock with': 8,
        'docked alongside with': 9, 
        'docking at the different dock with': 10, 
        'driving in the same direction with': 11,
        'parallelly parked on': 12, 
        'isolatedly parked on': 13,
        'randomly parked on': 14,
        'run along': 15, 
        'adjacent': 16, 
        'through': 17, 
        'converge': 18, 
        'intersect': 19, 
        'not run along': 20, 
        'parking in the same apron with': 21, 
        'parking in the different apron with': 22, 
        'parked alongside with': 23, 
        'not parked alongside with': 24, 
        'running along the different taxiway with': 25,
        'around': 26, 
        'not co-storage with': 27, 
        'running along the same taxiway with': 28, 
        'approach': 29,
        'away from': 30, 
        'within danger distance of': 31, 
        'incorrectly parked on': 32, 
        'in the same parking with': 33, 
        'in the different parking with': 34, 
        'not docked alongside with': 35, 
        'driving in the opposite direction with': 36, 
        'driving in the different lane with': 37, 
        'driving in the same lane with': 38, 
        'docking at the same breakwater with': 39, 
        'driving alongside with': 40,
        'running along the different runway with': 41,
        'violently emit': 42, 
        'exhaust to': 43,
        'slightly emit': 44,
        'supply to': 45, 
        'drive toward': 46, 
        'pass across': 47, 
        'drive off': 48,
        'pass under': 49,
        'within different line of': 50, 
        'within same line of': 51, 
        'directly connected to': 52, 
        'indirectly connected to': 53, 
        'pass through': 54,
        'directly transmit electricity to': 55, 
        'working on': 56, 
        'not working on': 57, 
        'indirectly transmit electricity to': 58
    }
    pred_list = [0] * 59
    pred_list[0] = 1
    for k, v in freq_dict.items():
        pred_list[int(pred2idx[k])] = v

    return pred_list


def get_sub_proto_label(mode='concat'):
    semantic_list = get_semantic_diversity()
    idx2concept = range(sum(semantic_list))

    if mode.lower() != "add" and mode.lower() != "concat" and mode.lower() != "clip":
        raise ValueError("Incorrect mode you input it.")
    cluster_dict = torch.load(
        f"/home/xiejunlin/workspace/Intra-Imbalanced-SGG/datasets/datafiles/intra-work/cluster_results/cluster_dict_{mode}.pt")

    return cluster_dict