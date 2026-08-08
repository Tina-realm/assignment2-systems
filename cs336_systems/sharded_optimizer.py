from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._next_parameter_index = 0
        self._parameter_owners: list[tuple[torch.Tensor, int]] = []
        self._local_param_groups: list[dict[str, Any]] = []
        self._optimizer: torch.optim.Optimizer | None = None

        super().__init__(params, defaults=kwargs)

        if not self._local_param_groups:
            self._local_param_groups.append({"params": []})
        self._optimizer = optimizer_cls(self._local_param_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        global_param_group = dict(param_group)
        global_param_group["params"] = params
        super().add_param_group(global_param_group)

        local_params = []
        for parameter in params:
            owner = self._next_parameter_index % self.world_size
            self._next_parameter_index += 1
            self._parameter_owners.append((parameter, owner))
            if owner == self.rank:
                local_params.append(parameter)

        if not local_params:
            return

        local_param_group = dict(param_group)
        local_param_group["params"] = local_params
        if self._optimizer is None:
            self._local_param_groups.append(local_param_group)
        else:
            self._optimizer.add_param_group(local_param_group)

    def step(self, closure=None, **kwargs: Any):
        if self._optimizer is None:
            raise RuntimeError("Wrapped optimizer has not been initialized.")

        loss = self._optimizer.step(closure=closure, **kwargs)
        self._synchronize_parameters()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def _synchronize_parameters(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        for parameter, owner in self._parameter_owners:
            dist.broadcast(parameter.data, src=owner)
