import logging
import torch
import types
from .base import _BaseAggregator

debug_logger = logging.getLogger("debug")

def CM(inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2

class NACLP_T(_BaseAggregator):
    def __init__(self, n_iter, AM, f):
        self.tau = None
        self.n_iter = n_iter
        super(NACLP_T, self).__init__()
        self.momentum = None
        self.AM_max = AM
        self.AM = None
        self.f = f

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau**2 / v_norm**2)
        return v * scale

    def __call__(self, inputs):

        num_elements = len(inputs) - self.f
        selected_inputs = torch.stack([inputs[i] for i in range(num_elements)])  # 将选中的输入堆叠成一个张量
        # 计算均值
        CMT = selected_inputs.mean(dim=0)  # 计算均值
        cmt_list = []
        for i in range(num_elements):
            cmt_list.append((inputs[i] - CMT) ** 2)

        mean_sq = torch.stack(cmt_list, dim=0).mean(dim=0)
        var_inputs = mean_sq.sum()
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
        return "NACLP_T (tau={}, n_iter={})".format(self.tau, self.n_iter)
