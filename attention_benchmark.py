from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.model import scaled_dot_product_attention


DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]
MIB = 1024**2


@dataclass
class BenchmarkResult:
    d_model: int
    sequence_length: int
    forward_mean_ms: float | None
    forward_std_ms: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    memory_before_backward_mib: float | None
    forward_graph_delta_mib: float | None
    attention_matrix_mib: float
    qkv_mib: float
    status: str


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def clear_cuda_memory(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)


def mean_and_std_ms(timings: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(timings) * 1000
    std = statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0
    return mean, std


def clear_input_gradients(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    q.grad = None
    k.grad = None
    v.grad = None


def allocate_inputs(
    batch_size: int, sequence_length: int, d_model: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    return q, k, v


def benchmark_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup_steps):
        output = scaled_dot_product_attention(q, k, v)
        synchronize(device)
        del output

    timings = []
    for _ in range(measurement_steps):
        synchronize(device)
        start = time.perf_counter()
        output = scaled_dot_product_attention(q, k, v)
        synchronize(device)
        timings.append(time.perf_counter() - start)
        del output
    return mean_and_std_ms(timings)


def measure_memory_before_backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, device: torch.device
) -> tuple[int, int]:
    synchronize(device)
    baseline_memory = torch.cuda.memory_allocated(device)
    output = scaled_dot_product_attention(q, k, v)
    loss = output.sum()
    synchronize(device)
    memory_before_backward = torch.cuda.memory_allocated(device)
    del loss, output
    return memory_before_backward, memory_before_backward - baseline_memory


def benchmark_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup_steps):
        output = scaled_dot_product_attention(q, k, v)
        loss = output.sum()
        loss.backward()
        synchronize(device)
        clear_input_gradients(q, k, v)
        del loss, output

    timings = []
    for _ in range(measurement_steps):
        output = scaled_dot_product_attention(q, k, v)
        loss = output.sum()
        synchronize(device)
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        timings.append(time.perf_counter() - start)
        clear_input_gradients(q, k, v)
        del loss, output
    return mean_and_std_ms(timings)


def empty_result(d_model: int, sequence_length: int, batch_size: int) -> BenchmarkResult:
    element_size = torch.empty((), dtype=torch.float32).element_size()
    return BenchmarkResult(
        d_model=d_model,
        sequence_length=sequence_length,
        forward_mean_ms=None,
        forward_std_ms=None,
        backward_mean_ms=None,
        backward_std_ms=None,
        memory_before_backward_mib=None,
        forward_graph_delta_mib=None,
        attention_matrix_mib=batch_size * sequence_length**2 * element_size / MIB,
        qkv_mib=3 * batch_size * sequence_length * d_model * element_size / MIB,
        status="not run",
    )


def run_case(
    d_model: int,
    sequence_length: int,
    batch_size: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> BenchmarkResult:
    result = empty_result(d_model, sequence_length, batch_size)

    try:
        q, k, v = allocate_inputs(batch_size, sequence_length, d_model, device)
    except torch.OutOfMemoryError:
        result.status = "OOM during input allocation"
        return result

    try:
        result.forward_mean_ms, result.forward_std_ms = benchmark_forward(
            q, k, v, warmup_steps, measurement_steps, device
        )
    except torch.OutOfMemoryError:
        result.status = "OOM during forward"
        return result

    try:
        memory_before_backward, forward_graph_delta = measure_memory_before_backward(q, k, v, device)
        result.memory_before_backward_mib = memory_before_backward / MIB
        result.forward_graph_delta_mib = forward_graph_delta / MIB
    except torch.OutOfMemoryError:
        result.status = "OOM during memory probe"
        return result

    try:
        result.backward_mean_ms, result.backward_std_ms = benchmark_backward(
            q, k, v, warmup_steps, measurement_steps, device
        )
    except torch.OutOfMemoryError:
        result.status = "OOM during backward"
        return result

    result.status = "ok"
    return result


def format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.d_model:7d} {result.sequence_length:8d} "
        f"{format_number(result.forward_mean_ms):>12} "
        f"{format_number(result.backward_mean_ms):>12} "
        f"{format_number(result.memory_before_backward_mib):>18} "
        f"{format_number(result.forward_graph_delta_mib):>17}  "
        f"{result.status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the assignment's scaled dot-product attention.")
    parser.add_argument("--device", default="cuda", help="CUDA device, such as cuda or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=int, nargs="+", default=DEFAULT_D_MODELS)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("attention_benchmark.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("This assignment benchmark requires CUDA timing and memory APIs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script on the CUDA-enabled cloud instance.")
    if args.batch_size <= 0 or args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("Batch size and measurement steps must be positive; warm-up steps cannot be negative.")
    if any(value <= 0 for value in args.d_models + args.sequence_lengths):
        raise ValueError("Embedding dimensions and sequence lengths must be positive.")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    validate_args(args, device)
    torch.cuda.set_device(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkResult.__dataclass_fields__)

    print(f"device={torch.cuda.get_device_name(device)} batch_size={args.batch_size} dtype=float32")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print(
        f"{'d_model':>7} {'seq_len':>8} {'forward_ms':>12} {'backward_ms':>12} "
        f"{'before_backward_MiB':>18} {'graph_delta_MiB':>17}  status"
    )

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for d_model in args.d_models:
            for sequence_length in args.sequence_lengths:
                clear_cuda_memory(device)
                try:
                    result = run_case(
                        d_model=d_model,
                        sequence_length=sequence_length,
                        batch_size=args.batch_size,
                        warmup_steps=args.warmup_steps,
                        measurement_steps=args.measurement_steps,
                        device=device,
                    )
                except torch.OutOfMemoryError:
                    result = empty_result(d_model, sequence_length, args.batch_size)
                    result.status = "OOM during cleanup"

                writer.writerow(asdict(result))
                output_file.flush()
                print_result(result)

    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
