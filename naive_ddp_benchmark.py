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
class NaiveDDPBenchmarkResult:
    backend: str
    sync_mode: str
    model_size: str
    world_size: int
    global_batch_size: int
    local_batch_size: int
    context_length: int
    vocab_size: int
    optimizer: str
    mixed_precision: bool
    warmup_steps: int
    measurement_steps: int
    step_mean_ms: float
    step_std_ms: float
    comm_mean_ms: float
    comm_std_ms: float
    comm_fraction: float
    max_rank_step_mean_ms: float
    max_rank_comm_mean_ms: float
    parameter_count: int
    status: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
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
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


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
    sync_mode: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision):
        logits = ddp_model(inputs)
        loss = compute_loss(logits, targets)
    loss.backward()

    synchronize(device)
    comm_start = time.perf_counter()
    if sync_mode == "individual":
        ddp_model.finish_gradient_synchronization()
    elif sync_mode == "flat":
        ddp_model.finish_flat_gradient_synchronization()
    elif sync_mode == "overlap":
        ddp_model.finish_gradient_synchronization()
    else:
        raise ValueError(f"Unknown synchronization mode: {sync_mode}")
    synchronize(device)
    comm_seconds = time.perf_counter() - comm_start

    optimizer.step()
    return comm_seconds


def make_result(
    args: argparse.Namespace,
    world_size: int,
    local_batch_size: int,
    parameter_count: int,
    rank_step_timings: list[list[float]],
    rank_comm_timings: list[list[float]],
) -> NaiveDDPBenchmarkResult:
    rank_step_means = [statistics.fmean(timings) for timings in rank_step_timings]
    rank_comm_means = [statistics.fmean(timings) for timings in rank_comm_timings]
    all_step_timings = [timing for timings in rank_step_timings for timing in timings]
    all_comm_timings = [timing for timings in rank_comm_timings for timing in timings]

    step_mean = statistics.fmean(all_step_timings)
    comm_mean = statistics.fmean(all_comm_timings)
    return NaiveDDPBenchmarkResult(
        backend=args.backend,
        sync_mode=args.sync_mode,
        model_size=args.model_size,
        world_size=world_size,
        global_batch_size=args.global_batch_size,
        local_batch_size=local_batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        optimizer=args.optimizer,
        mixed_precision=args.mixed_precision,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        step_mean_ms=step_mean * 1000,
        step_std_ms=(statistics.stdev(all_step_timings) * 1000 if len(all_step_timings) > 1 else 0.0),
        comm_mean_ms=comm_mean * 1000,
        comm_std_ms=(statistics.stdev(all_comm_timings) * 1000 if len(all_comm_timings) > 1 else 0.0),
        comm_fraction=comm_mean / step_mean,
        max_rank_step_mean_ms=max(rank_step_means) * 1000,
        max_rank_comm_mean_ms=max(rank_comm_means) * 1000,
        parameter_count=parameter_count,
        status="ok",
    )


def benchmark_worker(rank: int, world_size: int, master_port: int, args: argparse.Namespace) -> None:
    device = setup_process_group(rank, world_size, args.backend, master_port)
    local_batch_size = args.global_batch_size // world_size

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    ddp_model = DistributedDataParallel(model, overlap_gradients=args.sync_mode == "overlap")
    optimizer = make_optimizer(args, ddp_model)
    inputs, targets = make_local_batch(args, rank, local_batch_size, device)
    parameter_count = sum(parameter.numel() for parameter in ddp_model.parameters())

    ddp_model.train()
    for _ in range(args.warmup_steps):
        run_training_step(ddp_model, optimizer, inputs, targets, device, args.mixed_precision, args.sync_mode)
        synchronize(device)

    dist.barrier()
    step_timings = []
    comm_timings = []
    for _ in range(args.measurement_steps):
        synchronize(device)
        dist.barrier()
        step_start = time.perf_counter()
        comm_seconds = run_training_step(ddp_model, optimizer, inputs, targets, device, args.mixed_precision, args.sync_mode)
        synchronize(device)
        step_seconds = time.perf_counter() - step_start
        step_timings.append(step_seconds)
        comm_timings.append(comm_seconds)

    gathered_step_timings: list[list[float] | None] = [None for _ in range(world_size)]
    gathered_comm_timings: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_step_timings, step_timings)
    dist.all_gather_object(gathered_comm_timings, comm_timings)

    if rank == 0:
        result = make_result(
            args=args,
            world_size=world_size,
            local_batch_size=local_batch_size,
            parameter_count=parameter_count,
            rank_step_timings=[timings for timings in gathered_step_timings if timings is not None],
            rank_comm_timings=[timings for timings in gathered_comm_timings if timings is not None],
        )
        with args.output.open("a", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(NaiveDDPBenchmarkResult.__dataclass_fields__))
            writer.writerow(asdict(result))
            output_file.flush()
        print_result(result)

    cleanup_process_group()


def print_result(result: NaiveDDPBenchmarkResult) -> None:
    print(
        f"{result.backend:>5} {result.sync_mode:>10} {result.model_size:>6} {result.world_size:5d} "
        f"{result.global_batch_size:8d} {result.context_length:8d} "
        f"{result.step_mean_ms:12.3f} {result.comm_mean_ms:12.3f} "
        f"{result.comm_fraction * 100:9.2f}% {result.max_rank_step_mean_ms:16.3f}  {result.status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP training with individual, flattened, or overlapped gradient all-reduces.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--sync-modes", choices=["individual", "flat", "overlap"], nargs="+", default=["individual"])
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), default="xl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("naive_ddp_benchmark.csv"))
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
        writer = csv.DictWriter(output_file, fieldnames=list(NaiveDDPBenchmarkResult.__dataclass_fields__))
        writer.writeheader()

    print(f"output={args.output.resolve()}")
    print(
        f"{'bkd':>5} {'sync':>10} {'model':>6} {'world':>5} {'glob_bs':>8} {'ctx':>8} "
        f"{'step_ms':>12} {'comm_ms':>12} {'comm_pct':>10} {'max_rank_step_ms':>16}  status"
    )

    for sync_mode in args.sync_modes:
        args.sync_mode = sync_mode
        master_port = find_free_port()
        mp.spawn(
            benchmark_worker,
            args=(args.world_size, master_port, args),
            nprocs=args.world_size,
            join=True,
        )
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
