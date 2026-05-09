import math
import torch
import torch.nn as nn


class EMA:
    """
    Exponential moving average of model parameters (target encoder).
    Momentum increases from momentum_start → 1.0 on a cosine schedule.
    """

    def __init__(self, online: nn.Module, target: nn.Module, momentum_start: float = 0.996):
        self.online = online
        self.target = target
        self.momentum = momentum_start

        # Target starts as a copy of online; gradients are never needed.
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad_(False)

    @torch.no_grad()
    def step(self):
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=1.0 - self.momentum)

    def update_momentum(self, step: int, total_steps: int, start: float, end: float = 1.0):
        """Cosine schedule: momentum goes from start → end over total_steps."""
        self.momentum = end - (end - start) * (math.cos(math.pi * step / total_steps) + 1) / 2
