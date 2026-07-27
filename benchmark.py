from __future__ import annotations

import argparse
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def parse_dtype(dtype_arg: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return dtypes[dtype_arg]


def build_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    config = MODEL_SIZES[args.size]
    d_model = args.d_model if args.d_model is not None else config.d_model
    d_ff = args.d_ff if args.d_ff is not None else config.d_ff
    num_layers = args.num_layers if args.num_layers is not None else config.num_layers
    num_heads = args.num_heads if args.num_heads is not None else config.num_heads

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
    )
    return model.to(device=device, dtype=parse_dtype(args.dtype))


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


def autocast_context(device: torch.device, mixed_precision: bool):
    if not mixed_precision:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def benchmark_mode(
    mode: str,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    mixed_precision: bool,
) -> list[float]:
    def run_step() -> None:
        if mode == "forward":
            with torch.no_grad(), autocast_context(device, mixed_precision):
                model(inputs)
            return

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, mixed_precision):
            logits = model(inputs)
            loss = compute_loss(logits, targets)
        loss.backward()

        if mode == "train":
            optimizer.step()

    model.train(mode != "forward")

    for _ in range(warmup_steps):
        run_step()
        synchronize(device)

    timings = []
    for _ in range(measurement_steps):
        synchronize(device)
        start = timeit.default_timer()
        run_step()
        synchronize(device)
        timings.append(timeit.default_timer() - start)

    return timings


def format_stats(mode: str, timings: list[float]) -> str:
    mean = statistics.fmean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return f"{mode:16s} mean={mean * 1000:10.3f} ms  std={std * 1000:10.3f} ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM end-to-end step times.")
    parser.add_argument("--size", choices=MODEL_SIZES.keys(), default="debug")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, etc.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Keep model parameters in FP32 and autocast the forward pass to BF16.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["all", "compare", "forward", "forward_backward", "train"],
        default="all",
        help="compare runs forward and forward+backward; train also includes the optimizer step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mixed_precision and args.dtype != "float32":
        raise ValueError("--mixed-precision requires --dtype float32 so parameters remain in FP32.")

    device = resolve_device(args.device)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    inputs, targets = make_batch(args, device)

    if args.mode == "all":
        modes = ["forward", "forward_backward", "train"]
    elif args.mode == "compare":
        modes = ["forward", "forward_backward"]
    else:
        modes = [args.mode]

    autocast_dtype = "bfloat16" if args.mixed_precision else "disabled"
    print(f"device={device} parameter_dtype={args.dtype} autocast={autocast_dtype} size={args.size}")
    print(
        f"batch_size={args.batch_size} context_length={args.context_length} "
        f"vocab_size={args.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for mode in modes:
        timings = benchmark_mode(
            mode=mode,
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
            device=device,
            mixed_precision=args.mixed_precision,
        )
        print(format_stats(mode, timings))

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak_cuda_memory={peak_gib:.3f} GiB")


if __name__ == "__main__":
    main()
