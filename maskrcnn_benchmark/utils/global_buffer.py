from collections import defaultdict
import pickle
import torch
import os

from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size, all_gather, synchronize
from maskrcnn_benchmark.config import cfg

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


@singleton
class _GlobalBuffer():
    """a singleton buffer for store data in anywhere of program
    """
    def __init__(self ):
        self.multi_proc = (get_world_size() > 1)
        self.data = defaultdict(list)

    def add_data(self, key, val):
        if not isinstance(val, torch.Tensor):
            val = torch.Tensor(val)
        else:
            val = val.detach()

        val = torch.cat(all_gather(val))

        if not is_main_process():
            del val
            return
        self.data[key].append(val.cpu().numpy())

    def __str__(self):
        ret_str = f"Buffer contains data: (key, value type)\n"
        for k, v in self.data.items():
            ret_str += f"    {k}, {type(v).__name__}\n"
        ret_str += f"id {id(self)}"
        return ret_str


def store_data(k, v, ):
    if cfg.GLOBAL_BUFFER_ON:
        buffer = _GlobalBuffer()
        buffer.add_data(k, v)
        synchronize()


def save_buffer(output_dir):
    if cfg.GLOBAL_BUFFER_ON:
        if is_main_process():
            buffer = _GlobalBuffer()
            with open(os.path.join(output_dir, "inter_data_buffer.pkl"), 'wb') as f:
                pickle.dump(buffer.data, f)

            print("save buffer:", str(buffer))
