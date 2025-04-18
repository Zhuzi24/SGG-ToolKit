# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler  # 随机采样
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven # False

        self.groups = torch.unique(self.group_ids).sort(0)[0]  ## tensor([0, 1])

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids) # 57723
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))  # 随机采样  len(set(tt)) == 57723 不重复
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64) # dataset_size个-1组成一个tensor all -1
        order[sampled_ids] = torch.arange(len(sampled_ids))  # 把采样的样本的index转换为每个样本对应的顺序
        #### torch.arange(len(sampled_ids)) 0-57722 
        # get a mask with the elements that were sampled
        mask = order >= 0  # all True sum(mask) 57723

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups] ## 分组
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]   # len(relative_order[0]) + len(relative_order[1] 57723
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order] ### 排序len(permutation_ids[0]) + len(permutation_ids[1]) 57723
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]   #len(permuted_clusters[0]) 42445
        # 在batch_size中拆分每个集群，并合并为张量列表
        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]   ## 对两个clusters中的值进行分组batch
        merged = tuple(itertools.chain.from_iterable(splits)) # 合并

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly  现在每个批在内部都有正确的顺序，但它们是按集群分组的。找出不同批次之间的排列，使它们尽可能接近我们在采样器中的顺序。为此，我们将考虑排序来自每个批的第一个元素，并相应地排序
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler) 从采样的索引和它们出现的位置(由采样器返回)获取和反向映射
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven: # False
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches: #False 
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()  ### 准备batchs
        self._batches = batches
        return iter(batches)  # 迭代器使用for循环遍历列表。

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
