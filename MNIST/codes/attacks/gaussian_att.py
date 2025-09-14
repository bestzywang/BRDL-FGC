import torch
import torch.distributed as dist
import numpy as np
from scipy.stats import norm

from codes.worker import ByzantineWorker


class Gaussian_Attack(ByzantineWorker):
    """
    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """

    def __init__(self, z=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of supporters
        self.z_max = z
        

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        stacked_gradients = torch.stack(gradients, 1)
        mu = torch.mean(stacked_gradients, 1)
        std = torch.std(stacked_gradients, 1)

        # # 获取 A 中的最大元素并构成向量
        # std = torch.max(std) * torch.ones_like(std) * self.z_max
        # self._gradient = torch.normal(mean=torch.zeros_like(mu), std= std)
        self._gradient = torch.normal(mean=torch.zeros_like(mu), std= std * self.z_max)

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError
