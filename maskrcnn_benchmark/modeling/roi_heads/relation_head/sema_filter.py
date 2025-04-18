from torch.autograd import Variable
import argparse
import copy
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
import json




class sema_sx(nn.Module):
    def __init__(self, flag=None):
        super(sema_sx, self).__init__()

    
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 
        SF_path = os.path.join(current_dir,'SF_list.json')

        with open(SF_path, 'r') as f:
              self.mt = json.load(f)



    def sx(self,rel_pair_idxs,obj, flag_labels = None):
        
        cp_rel_pair_idxs = copy.deepcopy(rel_pair_idxs)
        heads = obj[rel_pair_idxs[:, 0]].long()
        tails = obj[rel_pair_idxs[:, 1]].long()
        tep = torch.tensor(self.mt)
        mt_list = tep[heads, tails]
        row_sums = torch.sum(mt_list, dim=1)
        zero_positions = torch.nonzero(row_sums == 0, as_tuple=True)[0]
        mask = torch.ones(len(rel_pair_idxs), dtype=bool)
    

   

        mask[zero_positions] = False
        filtered_rel_pair_idxs = rel_pair_idxs[mask].long()

        print("filtered / all: ", str(len(zero_positions )) +  "/" + str(len(cp_rel_pair_idxs)), " save_ratio: ", len(filtered_rel_pair_idxs) / len(cp_rel_pair_idxs))
            
        return [filtered_rel_pair_idxs.cuda()]

        


