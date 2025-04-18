import numpy as np
import torch
from torch import nn

class FrequencyBias_GCL(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """
    '''
    该函数可用来替换roi_relation_predictors.py中的self.freq_bias()方法

    我们增加了predicate_all_list，这是一个51维的向量，用来囊括所有目标的谓词，如果该谓词在这个类中，则
    设置其的值大于0.例如对于原6分类，共有6个值大于0.其中on是第31位，在6分类中是第5位，则设置
    predicate_all_list[31]=5，以此类推
    '''

    def __init__(self, cfg, statistics, Dataset_choice, eps=1e-3, predicate_all_list=None):
        super(FrequencyBias_GCL, self).__init__()
        # assert predicate_all_list is not None
        if Dataset_choice == 'VG':
            self.num_obj_cls =  151 #cfg.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
        elif Dataset_choice == 'GQA_200':
            self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
        # self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls =  len(list(filter(lambda x: x != 0, predicate_all_list))) + 1 # max(predicate_all_list) + 1
        old_matrix = statistics['fg_matrix'].float()

        fg_matrix = torch.zeros([self.num_obj_cls, self.num_obj_cls, self.num_rel_cls],
                                dtype=old_matrix.dtype, device=old_matrix.device)

        lines = 0
        # assert len(predicate_all_list) == 51 or len(predicate_all_list) == 101
        for i in range(len(predicate_all_list)):
            if i == 0 or predicate_all_list[i] > 0:
                fg_matrix[:, :, lines] = old_matrix[:, :, i]
                lines = lines + 1
        assert lines == self.num_rel_cls

        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        '''以下是原函数，以上是我改的部分'''
        # pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)

