from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ShardedParameter:
    name: str
    module: torch.nn.Module
    full_shape: torch.Size
    full_numel: int
    shard_numel: int
    start: int
    end: int
    local_shard: torch.Tensor


class FullyShardedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._sharded_parameters: dict[str, ShardedParameter] = {}
        self._replicated_parameter_names: set[str] = set()
        self._hook_handles = []

        self._shard_module_parameters()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        for name, parameter in self.module.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue

            if name in self._sharded_parameters:
                self._reduce_sharded_gradient(self._sharded_parameters[name], parameter)
            else:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params = {}
        for name, parameter in self.module.named_parameters():
            if name in self._sharded_parameters:
                state = self._sharded_parameters[name]
                full_params[name] = self._all_gather_flat_parameter(state, dtype=state.local_shard.dtype).view(state.full_shape)
            else:
                full_params[name] = parameter.detach().clone()
        return full_params

    def _shard_module_parameters(self) -> None:
        from cs336_basics.model import Embedding, Linear

        sharded_parameter_ids = set()
        modules = dict(self.module.named_modules())

        for module_name, module in modules.items():
            if not isinstance(module, (Linear, Embedding)):
                continue

            parameter = module.weight
            parameter_id = id(parameter)
            if parameter_id in sharded_parameter_ids:
                continue
            sharded_parameter_ids.add(parameter_id)

            parameter_name = f"{module_name}.weight" if module_name else "weight"
            state = self._make_sharded_parameter(parameter_name, module, parameter)
            self._sharded_parameters[parameter_name] = state
            self._hook_handles.append(module.register_forward_pre_hook(self._make_forward_pre_hook(parameter_name)))

        for name, parameter in self.module.named_parameters():
            if id(parameter) not in sharded_parameter_ids:
                self._replicated_parameter_names.add(name)

    def _make_sharded_parameter(self, name: str, module: torch.nn.Module, parameter: torch.nn.Parameter) -> ShardedParameter:
        full_shape = parameter.data.shape
        full_numel = parameter.data.numel()
        shard_numel = (full_numel + self.world_size - 1) // self.world_size
        start = self.rank * shard_numel
        end = min(start + shard_numel, full_numel)

        flat_parameter = parameter.data.detach().reshape(-1)
        if start < full_numel:
            local_shard = flat_parameter[start:end].clone()
        else:
            local_shard = flat_parameter.new_empty((0,))
        if local_shard.numel() < shard_numel:
            local_shard = torch.cat([local_shard, flat_parameter.new_zeros(shard_numel - local_shard.numel())])

        local_shard = local_shard.to(dtype=torch.float32)
        parameter.data = local_shard

        return ShardedParameter(
            name=name,
            module=module,
            full_shape=full_shape,
            full_numel=full_numel,
            shard_numel=shard_numel,
            start=start,
            end=end,
            local_shard=local_shard,
        )

    def _make_forward_pre_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs) -> None:
            state = self._sharded_parameters[name]
            parameter = module.weight
            state.local_shard = parameter.data
            full_parameter = self._all_gather_flat_parameter(state, dtype=self.compute_dtype or state.local_shard.dtype)
            parameter.data = full_parameter.view(state.full_shape)
            parameter.grad = None

        return hook

    def _all_gather_flat_parameter(self, state: ShardedParameter, dtype: torch.dtype) -> torch.Tensor:
        local_shard = state.local_shard.to(dtype=dtype)
        if not dist.is_available() or not dist.is_initialized():
            return local_shard[: state.full_numel].detach().clone()

        gathered_shards = [torch.empty_like(local_shard) for _ in range(self.world_size)]
        if local_shard.device.type == "cuda":
            torch.cuda.nvtx.range_push(f"fsdp_all_gather:{state.name}")
        try:
            dist.all_gather(gathered_shards, local_shard)
        finally:
            if local_shard.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
        return torch.cat(gathered_shards, dim=0)[: state.full_numel]

    def _reduce_sharded_gradient(self, state: ShardedParameter, parameter: torch.nn.Parameter) -> None:
        full_grad = parameter.grad.detach().reshape(-1).to(torch.float32)
        if full_grad.numel() < state.shard_numel * self.world_size:
            full_grad = torch.cat([full_grad, full_grad.new_zeros(state.shard_numel * self.world_size - full_grad.numel())])

        local_grad = torch.empty(state.shard_numel, dtype=full_grad.dtype, device=full_grad.device)
        try:
            if full_grad.device.type == "cuda":
                torch.cuda.nvtx.range_push(f"fsdp_reduce_scatter:{state.name}")
            dist.reduce_scatter_tensor(local_grad, full_grad, op=dist.ReduceOp.SUM)
        except RuntimeError:
            if full_grad.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"fsdp_all_reduce_fallback:{state.name}")
            dist.all_reduce(full_grad, op=dist.ReduceOp.SUM)
            local_grad.copy_(full_grad[state.start : state.start + state.shard_numel])
        finally:
            if full_grad.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
        local_grad.div_(self.world_size)

        parameter.data = state.local_shard
        parameter.grad = local_grad.to(dtype=parameter.data.dtype)
