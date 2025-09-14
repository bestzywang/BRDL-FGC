import logging
import torch
import types
from .base import _BaseAggregator
#from codes.aggregator.coordinatewise_median import CM

debug_logger = logging.getLogger("debug")

def CM(inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2

class NACLP(_BaseAggregator):
    def __init__(self, n_iter, AM):
        self.tau = None
        self.n_iter = n_iter
        super(NACLP, self).__init__()
        self.momentum = None
        self.AM_max = AM
        self.AM = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau**2 / v_norm**2)
        return v * scale

    def __call__(self, inputs):

        CMT = CM(inputs)
        cmt_list = []
        for i in range(len(inputs)):
            cmt_list.append(torch.abs(inputs[i]-CMT))

        var_inputs = ((CM(cmt_list))**2).sum()

        CMT_1 = CMT

        # self.momentum = CMT
        self.momentum = torch.zeros_like(inputs[0])
        self.tau = torch.sqrt(
                (((self.momentum - CMT_1)**2).sum() + var_inputs)
            )
        
        for _ in range(self.n_iter):
            
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )

        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "NACLP (tau={}, n_iter={})".format(self.tau, self.n_iter)
