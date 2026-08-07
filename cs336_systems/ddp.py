from __future__ import annotations

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


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

    def finish_flat_gradient_synchronization(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        parameters_with_grad = [parameter for parameter in self.module.parameters() if parameter.grad is not None]
        if not parameters_with_grad:
            return

        gradients = [parameter.grad for parameter in parameters_with_grad]
        flat_gradients = _flatten_dense_tensors(gradients)
        dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM)
        flat_gradients.div_(dist.get_world_size())

        for parameter, synced_gradient in zip(parameters_with_grad, _unflatten_dense_tensors(flat_gradients, gradients), strict=True):
            parameter.grad.copy_(synced_gradient)
