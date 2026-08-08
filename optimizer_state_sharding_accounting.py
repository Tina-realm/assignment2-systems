from __future__ import annotations

import argparse
import csv
import os
import socket
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
CS336_BASICS_SRC = REPO_ROOT / "cs336-basics"
if str(CS336_BASICS_SRC) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS_SRC))

from cs336_basics.model import BasicsTransformerLM  # noqa: E402
from cs336_systems.ddp import DistributedDataParallel  # noqa: E402
from cs336_systems.sharded_optimizer import ShardedOptimizer  # noqa: E402

GIB = 1024**3


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZES = {
    "debug": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass
class OptimizerStateShardingResult:
    optimizer_mode: str
    backend: str
    rank: int
    world_size: int
    model_size: str
    global_batch_size: int
    local_batch_size: int
    context_length: int
    vocab_size: int
    mixed_precision: bool
    warmup_steps: int
    measurement_steps: int
    parameter_count: int
    owned_parameter_count: int
    owned_parameter_fraction: float
    estimated_parameter_gib: float
    estimated_full_adam_state_gib: float
    estimated_local_adam_state_gib: float
    model_init_allocated_gib: float | None
    model_init_peak_gib: float | None
    before_step_allocated_gib: float | None
    before_step_peak_gib: float | None
    after_step_allocated_gib: float | None
    after_step_peak_gib: float | None
    step_mean_ms: float
    step_std_ms: float
    max_rank_step_mean_ms: float
    status: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_allocated_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    synchronize(device)
    return torch.cuda.memory_allocated(device) / GIB


def memory_peak_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) / GIB


def setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group() -> None:
    dist.barrier()
    dist.destroy_process_group()


def build_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    config = MODEL_SIZES[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=args.rope_theta,
    )
    return model.to(device=device)


def make_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer_mode == "regular":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer_mode == "sharded":
        return ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer mode: {args.optimizer_mode}")


def count_owned_parameters(optimizer: torch.optim.Optimizer, rank: int, parameter_count: int) -> int:
    if isinstance(optimizer, ShardedOptimizer):
        return sum(parameter.numel() for parameter, owner in optimizer._parameter_owners if owner == rank)
    return parameter_count


def make_local_batch(args: argparse.Namespace, rank: int, local_batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(local_batch_size, args.context_length),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(local_batch_size, args.context_length),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    return inputs, targets


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


def run_training_step(
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
        logits = ddp_model(inputs)
        loss = compute_loss(logits, targets)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()
    synchronize(device)


def measure_first_step_memory(
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[float | None, float | None, float | None, float | None]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
        logits = ddp_model(inputs)
        loss = compute_loss(logits, targets)
    loss.backward()
    ddp_model.finish_gradient_synchronization()

    before_step_allocated = memory_allocated_gib(device)
    before_step_peak = memory_peak_gib(device)

    optimizer.step()
    synchronize(device)

    after_step_allocated = memory_allocated_gib(device)
    after_step_peak = memory_peak_gib(device)
    return before_step_allocated, before_step_peak, after_step_allocated, after_step_peak


def worker(rank: int, world_size: int, master_port: int, args: argparse.Namespace) -> None:
    device = setup_process_group(rank, world_size, args.backend, master_port)
    local_batch_size = args.global_batch_size // world_size

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    ddp_model = DistributedDataParallel(model, overlap_gradients=False)
    parameter_count = sum(parameter.numel() for parameter in ddp_model.parameters())

    model_init_allocated = memory_allocated_gib(device)
    model_init_peak = memory_peak_gib(device)

    optimizer = make_optimizer(args, ddp_model)
    owned_parameter_count = count_owned_parameters(optimizer, rank, parameter_count)
    inputs, targets = make_local_batch(args, rank, local_batch_size, device)

    before_step_allocated, before_step_peak, after_step_allocated, after_step_peak = measure_first_step_memory(
        ddp_model=ddp_model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        device=device,
        mixed_precision=args.mixed_precision,
    )

    step_timings = []
    for _ in range(args.warmup_steps):
        run_training_step(ddp_model, optimizer, inputs, targets, device, args.mixed_precision)

    dist.barrier()
    for _ in range(args.measurement_steps):
        synchronize(device)
        dist.barrier()
        step_start = time.perf_counter()
        run_training_step(ddp_model, optimizer, inputs, targets, device, args.mixed_precision)
        step_timings.append(time.perf_counter() - step_start)

    gathered_step_timings: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_step_timings, step_timings)
    rank_step_means = [statistics.fmean(timings) for timings in gathered_step_timings if timings is not None]

    result = OptimizerStateShardingResult(
        optimizer_mode=args.optimizer_mode,
        backend=args.backend,
        rank=rank,
        world_size=world_size,
        model_size=args.model_size,
        global_batch_size=args.global_batch_size,
        local_batch_size=local_batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        mixed_precision=args.mixed_precision,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        parameter_count=parameter_count,
        owned_parameter_count=owned_parameter_count,
        owned_parameter_fraction=owned_parameter_count / parameter_count,
        estimated_parameter_gib=parameter_count * 4 / GIB,
        estimated_full_adam_state_gib=parameter_count * 8 / GIB,
        estimated_local_adam_state_gib=owned_parameter_count * 8 / GIB,
        model_init_allocated_gib=model_init_allocated,
        model_init_peak_gib=model_init_peak,
        before_step_allocated_gib=before_step_allocated,
        before_step_peak_gib=before_step_peak,
        after_step_allocated_gib=after_step_allocated,
        after_step_peak_gib=after_step_peak,
        step_mean_ms=statistics.fmean(step_timings) * 1000,
        step_std_ms=(statistics.stdev(step_timings) * 1000 if len(step_timings) > 1 else 0.0),
        max_rank_step_mean_ms=max(rank_step_means) * 1000,
        status="ok",
    )

    gathered_results: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, asdict(result))

    if rank == 0:
        rows = [row for row in gathered_results if row is not None]
        with args.output.open("a", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(OptimizerStateShardingResult.__dataclass_fields__))
            writer.writerows(rows)
            output_file.flush()
        print_summary(rows)

    cleanup_process_group()


def print_summary(rows: list[dict]) -> None:
    max_step = max(float(row["step_mean_ms"]) for row in rows)
    max_init = max_optional(row["model_init_allocated_gib"] for row in rows)
    max_before = max_optional(row["before_step_peak_gib"] for row in rows)
    max_after = max_optional(row["after_step_peak_gib"] for row in rows)
    mode = rows[0]["optimizer_mode"]
    print(f"{mode:>8} step_ms={max_step:.3f} model_init_allocated_GiB={format_optional(max_init)} before_step_peak_GiB={format_optional(max_before)} after_step_peak_GiB={format_optional(max_after)}")


def max_optional(values) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return max(float(value) for value in numeric)


def format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure optimizer state sharding memory and training-time overhead.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--optimizer-modes", choices=["regular", "sharded"], nargs="+", default=["regular", "sharded"])
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), default="xl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("optimizer_state_sharding_accounting.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.world_size <= 0:
        raise ValueError("World size must be positive.")
    if args.global_batch_size <= 0 or args.global_batch_size % args.world_size != 0:
        raise ValueError("Global batch size must be positive and divisible by world size.")
    if args.context_length <= 0 or args.vocab_size <= 0:
        raise ValueError("Context length and vocab size must be positive.")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("Warmup steps must be nonnegative and measurement steps must be positive.")
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires CUDA. Use --backend gloo with --model-size debug for local CPU smoke tests.")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(f"Requested {args.world_size} processes, but only {torch.cuda.device_count()} CUDA devices are available.")
    if args.backend == "gloo" and args.mixed_precision:
        raise ValueError("BF16 autocast smoke testing is intended for CUDA/NCCL in this script.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(OptimizerStateShardingResult.__dataclass_fields__))
        writer.writeheader()

    print(f"output={args.output.resolve()}")
    for optimizer_mode in args.optimizer_modes:
        args.optimizer_mode = optimizer_mode
        master_port = find_free_port()
        mp.spawn(worker, args=(args.world_size, master_port, args), nprocs=args.world_size, join=True)
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
