import torch
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['KL_divergence']

def KL_divergence(logits_p, logits_q, reduce=True):
    # p = softmax(logits_p)
    # q = softmax(logits_q)
    # KL(p||q)
    # suppose that p/q is in shape of [bs, num_classes]

    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)

    shape = list(p.size())
    _shape = list(q.size())
    assert shape == _shape
    #print(shape)
    num_classes = shape[1]
    epsilon = 1e-8
    _p = (p + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    _q = (q + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    if reduce:
        return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
    else:
        return torch.sum(_p * torch.log(_p / _q), 1)

if __name__ == '__main__':
    a = torch.zeros([3,4]).cuda()
    b = torch.ones([3,4]).cuda()
    c = KL_divergence(a, b, reduce=False)
    print(c)
    print(c.shape)