# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations # max
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"): # False
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler: # /media/dell/data1/WTZ/BGAN-1204/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py 103  
                iteration += 1  
                if iteration > self.num_iterations:
                    break
                yield batch  # ”yield是一种能够暂时中止函数执行的语句。您可以用它返回此时的返回值并重新启动

    def __len__(self):
        return self.num_iterations
