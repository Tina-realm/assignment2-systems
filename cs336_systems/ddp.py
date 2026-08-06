from __future__ import annotations

import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_module_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)
