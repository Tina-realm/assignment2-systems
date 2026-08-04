from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
CS336_BASICS_SRC = REPO_ROOT / "cs336-basics"
if str(CS336_BASICS_SRC) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS_SRC))

from cs336_basics.model import scaled_dot_product_attention  # noqa: E402
from cs336_systems.flash_attention import FlashAttention2Triton  # noqa: E402


DEFAULT_SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_DTYPES = ["bfloat16", "float32"]

MIB = 1024**2
AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class BenchmarkResult:
    implementation: str
    dtype: str
    sequence_length: int
    d_model: int
    forward_ms: float | None
    backward_ms: float | None
    forward_backward_ms: float | None
    peak_memory_mib: float | None
    status: str


def parse_dtype(dtype_name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtypes[dtype_name]


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def clear_cuda_memory(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)


def clear_input_gradients(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    q.grad = None
    k.grad = None
    v.grad = None


def make_inputs(
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(1, sequence_length, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    dO = torch.randn_like(q)
    return q, k, v, dO


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    token_indices = torch.arange(sequence_length, device=device)
    return token_indices[None, :, None] >= token_indices[None, None, :]


def make_pytorch_attention(sequence_length: int, device: torch.device) -> AttentionFunction:
    causal_mask = make_causal_mask(sequence_length, device)

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(q, k, v, causal_mask)

    return attention


def make_triton_attention() -> AttentionFunction:
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return FlashAttention2Triton.apply(q, k, v, True)

    return attention


def do_bench(fn: Callable[[], None], warmup_steps: int, measurement_steps: int) -> float:
    import triton.testing

    return float(triton.testing.do_bench(fn, warmup=warmup_steps, rep=measurement_steps))


def benchmark_forward(
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
) -> float:
    def fn() -> None:
        output = attention(q, k, v)
        del output

    return do_bench(fn, warmup_steps, measurement_steps)


def benchmark_backward(
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
) -> float:
    saved_output = attention(q, k, v)

    def fn() -> None:
        clear_input_gradients(q, k, v)
        saved_output.backward(dO, retain_graph=True)

    try:
        return do_bench(fn, warmup_steps, measurement_steps)
    finally:
        clear_input_gradients(q, k, v)


def benchmark_forward_backward(
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dO: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
) -> float:
    def fn() -> None:
        clear_input_gradients(q, k, v)
        output = attention(q, k, v)
        output.backward(dO)
        del output

    try:
        return do_bench(fn, warmup_steps, measurement_steps)
    finally:
        clear_input_gradients(q, k, v)


def empty_result(
    implementation: str,
    dtype_name: str,
    sequence_length: int,
    d_model: int,
    status: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        implementation=implementation,
        dtype=dtype_name,
        sequence_length=sequence_length,
        d_model=d_model,
        forward_ms=None,
        backward_ms=None,
        forward_backward_ms=None,
        peak_memory_mib=None,
        status=status,
    )


def run_case(
    implementation: str,
    attention: AttentionFunction,
    dtype_name: str,
    sequence_length: int,
    d_model: int,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> BenchmarkResult:
    dtype = parse_dtype(dtype_name)
    result = empty_result(implementation, dtype_name, sequence_length, d_model, "not run")
    q = k = v = dO = None

    try:
        q, k, v, dO = make_inputs(sequence_length, d_model, dtype, device)
        result.forward_ms = benchmark_forward(attention, q, k, v, warmup_steps, measurement_steps)
        result.backward_ms = benchmark_backward(attention, q, k, v, dO, warmup_steps, measurement_steps)
        result.forward_backward_ms = benchmark_forward_backward(attention, q, k, v, dO, warmup_steps, measurement_steps)
        synchronize(device)
        result.peak_memory_mib = torch.cuda.max_memory_allocated(device) / MIB
        result.status = "ok"
    except torch.OutOfMemoryError:
        result.status = "OOM"
        torch.cuda.synchronize(device)
    finally:
        if q is not None and k is not None and v is not None:
            clear_input_gradients(q, k, v)
        del q, k, v, dO

    return result


def format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.implementation:>8} {result.dtype:>8} {result.sequence_length:8d} {result.d_model:7d} "
        f"{format_number(result.forward_ms):>12} "
        f"{format_number(result.backward_ms):>12} "
        f"{format_number(result.forward_backward_ms):>18} "
        f"{format_number(result.peak_memory_mib):>15}  "
        f"{result.status}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 against regular PyTorch attention.")
    parser.add_argument("--device", default="cuda", help="CUDA device, such as cuda or cuda:0.")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--d-models", type=int, nargs="+", default=DEFAULT_D_MODELS)
    parser.add_argument("--dtypes", nargs="+", choices=DEFAULT_DTYPES, default=DEFAULT_DTYPES)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument(
        "--implementation",
        choices=["pytorch", "triton", "both"],
        default="both",
        help="Which attention implementation to benchmark.",
    )
    parser.add_argument("--output", type=Path, default=Path("flash_benchmark.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("FlashAttention benchmarking requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this benchmark on a CUDA-enabled machine.")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("Warmup steps must be nonnegative and measurement steps must be positive.")
    if any(value <= 0 for value in args.sequence_lengths + args.d_models):
        raise ValueError("Sequence lengths and embedding dimensions must be positive.")

    try:
        import triton  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("Triton is unavailable. Install/run through the assignment uv environment on Linux CUDA.") from exc


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    validate_args(args, device)
    torch.cuda.set_device(device.index if device.index is not None else 0)

    implementations = ["pytorch", "triton"] if args.implementation == "both" else [args.implementation]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={torch.cuda.get_device_name(device)} batch_size=1 causal=True")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print(
        f"{'impl':>8} {'dtype':>8} {'seq_len':>8} {'d_model':>7} "
        f"{'forward_ms':>12} {'backward_ms':>12} {'forward_backward_ms':>18} {'peak_mem_MiB':>15}  status"
    )

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()

        for dtype_name in args.dtypes:
            for sequence_length in args.sequence_lengths:
                for d_model in args.d_models:
                    for implementation in implementations:
                        clear_cuda_memory(device)
                        try:
                            attention = make_pytorch_attention(sequence_length, device) if implementation == "pytorch" else make_triton_attention()
                            result = run_case(
                                implementation=implementation,
                                attention=attention,
                                dtype_name=dtype_name,
                                sequence_length=sequence_length,
                                d_model=d_model,
                                warmup_steps=args.warmup_steps,
                                measurement_steps=args.measurement_steps,
                                device=device,
                            )
                        except torch.OutOfMemoryError:
                            result = empty_result(implementation, dtype_name, sequence_length, d_model, "OOM")

                        writer.writerow(asdict(result))
                        output_file.flush()
                        print_result(result)

    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
