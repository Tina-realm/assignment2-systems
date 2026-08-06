from __future__ import annotations

import argparse
import csv
import os
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DEFAULT_DATA_SIZES_MB = [1, 10, 100, 1024]
DEFAULT_WORLD_SIZES = [2, 4, 6]
BYTES_PER_FLOAT32 = torch.empty((), dtype=torch.float32).element_size()


@dataclass
class AllReduceBenchmarkResult:
    backend: str
    world_size: int
    data_size_mb: int
    num_elements: int
    warmup_steps: int
    measurement_steps: int
    mean_ms: float
    std_ms: float
    min_rank_mean_ms: float
    max_rank_mean_ms: float
    approx_bandwidth_gbps: float
    status: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def synchronize_device(device: torch.device) -> None:
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


def make_result(
    backend: str,
    world_size: int,
    data_size_mb: int,
    num_elements: int,
    warmup_steps: int,
    measurement_steps: int,
    rank_timings: list[list[float]],
) -> AllReduceBenchmarkResult:
    rank_means = [statistics.fmean(timings) for timings in rank_timings]
    all_timings = [timing for timings in rank_timings for timing in timings]
    mean_seconds = statistics.fmean(all_timings)
    std_seconds = statistics.stdev(all_timings) if len(all_timings) > 1 else 0.0
    max_rank_mean_seconds = max(rank_means)
    data_size_bytes = data_size_mb * 1024 * 1024
    approx_bandwidth_gbps = data_size_bytes / max_rank_mean_seconds / 1e9

    return AllReduceBenchmarkResult(
        backend=backend,
        world_size=world_size,
        data_size_mb=data_size_mb,
        num_elements=num_elements,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        mean_ms=mean_seconds * 1000,
        std_ms=std_seconds * 1000,
        min_rank_mean_ms=min(rank_means) * 1000,
        max_rank_mean_ms=max_rank_mean_seconds * 1000,
        approx_bandwidth_gbps=approx_bandwidth_gbps,
        status="ok",
    )


def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    master_port: int,
    data_sizes_mb: list[int],
    warmup_steps: int,
    measurement_steps: int,
    output_path: str,
) -> None:
    device = setup_process_group(rank, world_size, backend, master_port)

    for data_size_mb in data_sizes_mb:
        num_elements = data_size_mb * 1024 * 1024 // BYTES_PER_FLOAT32
        tensor = torch.full((num_elements,), fill_value=rank + 1, dtype=torch.float32, device=device)

        for _ in range(warmup_steps):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            synchronize_device(device)
            tensor.fill_(rank + 1)
            synchronize_device(device)

        dist.barrier()
        timings = []
        for _ in range(measurement_steps):
            tensor.fill_(rank + 1)
            synchronize_device(device)
            dist.barrier()

            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            synchronize_device(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        gathered_timings: list[list[float] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_timings, timings)

        if rank == 0:
            result = make_result(
                backend=backend,
                world_size=world_size,
                data_size_mb=data_size_mb,
                num_elements=num_elements,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
                rank_timings=[rank_timing for rank_timing in gathered_timings if rank_timing is not None],
            )
            with Path(output_path).open("a", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(AllReduceBenchmarkResult.__dataclass_fields__))
                writer.writerow(asdict(result))
                output_file.flush()
            print_result(result)

        del tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cleanup_process_group()


def print_result(result: AllReduceBenchmarkResult) -> None:
    print(
        f"{result.backend:>5} {result.world_size:10d} {result.data_size_mb:12d} "
        f"{result.mean_ms:10.3f} {result.std_ms:9.3f} "
        f"{result.min_rank_mean_ms:16.3f} {result.max_rank_mean_ms:16.3f} "
        f"{result.approx_bandwidth_gbps:18.3f}  {result.status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-node distributed all-reduce communication.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=DEFAULT_WORLD_SIZES)
    parser.add_argument("--data-sizes-mb", type=int, nargs="+", default=DEFAULT_DATA_SIZES_MB)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("distributed_communication_single_node.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("Warmup steps must be nonnegative and measurement steps must be positive.")
    if any(world_size <= 0 for world_size in args.world_sizes):
        raise ValueError("World sizes must be positive.")
    if any(data_size_mb <= 0 for data_size_mb in args.data_sizes_mb):
        raise ValueError("Data sizes must be positive.")
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires CUDA. Use --backend gloo for local CPU smoke tests.")
        device_count = torch.cuda.device_count()
        max_world_size = max(args.world_sizes)
        if device_count < max_world_size:
            raise RuntimeError(f"Requested up to {max_world_size} processes, but only {device_count} CUDA devices are available.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(AllReduceBenchmarkResult.__dataclass_fields__))
        writer.writeheader()

    print(f"backend={args.backend} output={args.output.resolve()}")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print(
        f"{'bkd':>5} {'world_size':>10} {'data_size_mb':>12} "
        f"{'mean_ms':>10} {'std_ms':>9} {'min_rank_mean_ms':>16} "
        f"{'max_rank_mean_ms':>16} {'approx_bw_GBps':>18}  status"
    )

    for world_size in args.world_sizes:
        master_port = find_free_port()
        mp.spawn(
            benchmark_worker,
            args=(
                world_size,
                args.backend,
                master_port,
                args.data_sizes_mb,
                args.warmup_steps,
                args.measurement_steps,
                str(args.output),
            ),
            nprocs=world_size,
            join=True,
        )

    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
