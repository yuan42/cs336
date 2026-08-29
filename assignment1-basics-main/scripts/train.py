from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer import TransformerLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model on prepared TinyStories data.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=Path, default=DATA_DIR / "tinystories_train.npy")
    data.add_argument("--valid-data", type=Path, default=DATA_DIR / "tinystories_valid.npy")
    data.add_argument("--metadata", type=Path, default=DATA_DIR / "tinystories_data.json")
    data.add_argument("--vocab-size", type=int, help="Override vocab size from the prepared-data metadata.")

    model = parser.add_argument_group("model")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--steps", type=int, default=10_000)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--max-learning-rate", type=float, default=3e-4)
    training.add_argument("--min-learning-rate", type=float, default=3e-5)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.95)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--adam-eps", type=float, default=1e-8)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)

    monitoring = parser.add_argument_group("monitoring and checkpoints")
    monitoring.add_argument("--log-interval", type=int, default=10)
    monitoring.add_argument("--eval-interval", type=int, default=100)
    monitoring.add_argument("--eval-batches", type=int, default=20)
    monitoring.add_argument("--checkpoint-interval", type=int, default=500)
    monitoring.add_argument("--run-dir", type=Path)
    monitoring.add_argument("--resume", type=Path, help="Path to a checkpoint created by this script.")
    monitoring.add_argument(
        "--device",
        default="auto",
        help="Device such as mps, cuda, or cpu. The default chooses the best available device.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "checkpoint_interval": args.checkpoint_interval,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("--warmup-steps must be non-negative and smaller than --steps")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")


def load_vocab_size(metadata_path: Path, override: int | None) -> int:
    if override is not None:
        return override
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared-data metadata not found: {metadata_path}. "
            "Run scripts/prepare_data.py first or pass --vocab-size."
        )
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return int(metadata["vocab_size"])


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Treat every batch/sequence position as one next-token classification example."""
    return cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    valid_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        inputs, targets = get_batch(valid_data, batch_size, context_length, str(device))
        losses.append(compute_loss(model(inputs), targets, vocab_size))
    mean_loss = torch.stack(losses).mean().item()
    if was_training:
        model.train()
    return mean_loss


def append_log(log_path: Path, record: dict) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    train_path = args.train_data.expanduser().resolve()
    valid_path = args.valid_data.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    for data_path in (train_path, valid_path):
        if not data_path.is_file():
            raise FileNotFoundError(f"Prepared dataset not found: {data_path}")

    vocab_size = load_vocab_size(metadata_path, args.vocab_size)
    train_data = np.load(train_path, mmap_mode="r")
    valid_data = np.load(valid_path, mmap_mode="r")
    if len(train_data) <= args.context_length or len(valid_data) <= args.context_length:
        raise ValueError("Each dataset split must contain more tokens than --context-length")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    resume_path = args.resume.expanduser().resolve() if args.resume else None
    start_step = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_step = load_checkpoint(resume_path, model, optimizer)
        if start_step > args.steps:
            raise ValueError(f"Checkpoint is already at step {start_step:,}, which is beyond --steps {args.steps:,}")

    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
    elif resume_path is not None:
        run_dir = resume_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = DATA_DIR / "runs" / f"tinystories-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    log_path = run_dir / "training_log.jsonl"
    config_path = run_dir / "config.json"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "data": {
            "train": str(train_path),
            "valid": str(valid_path),
            "metadata": str(metadata_path),
            "vocab_size": vocab_size,
        },
        "model": {
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "parameter_count": parameter_count,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_learning_rate": args.max_learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_steps": args.warmup_steps,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
        "device": str(device),
    }
    if resume_path is None:
        if config_path.exists() or log_path.exists() or checkpoint_path.exists():
            raise FileExistsError(f"Run directory already contains training files: {run_dir}")
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")

    print(f"Device: {device}", flush=True)
    print(f"Parameters: {parameter_count:,}", flush=True)
    print(f"Train tokens: {len(train_data):,}", flush=True)
    print(f"Validation tokens: {len(valid_data):,}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    if start_step:
        print(f"Resuming from step: {start_step:,}", flush=True)
    if start_step == args.steps:
        print("Checkpoint has already reached the requested number of steps; nothing to train.", flush=True)
        return

    model.train()
    loss_window: list[torch.Tensor] = []
    window_started_at = time.perf_counter()
    last_valid_loss: float | None = None

    progress = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps, desc="Training", unit="step")
    for step in progress:
        learning_rate = lr_cosine_schedule(
            step,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_steps,
            args.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            str(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(inputs), targets, vocab_size)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        loss_window.append(loss.detach())

        completed_steps = step + 1
        should_log = completed_steps % args.log_interval == 0 or completed_steps == args.steps
        if should_log:
            synchronize(device)
            elapsed = time.perf_counter() - window_started_at
            mean_train_loss = torch.stack(loss_window).mean().item()
            tokens_processed = len(loss_window) * args.batch_size * args.context_length
            tokens_per_second = tokens_processed / elapsed
            progress.set_postfix(
                loss=f"{mean_train_loss:.4f}", lr=f"{learning_rate:.2e}", tok_s=f"{tokens_per_second:,.0f}"
            )
            append_log(
                log_path,
                {
                    "type": "train",
                    "step": completed_steps,
                    "loss": mean_train_loss,
                    "learning_rate": learning_rate,
                    "tokens_per_second": tokens_per_second,
                },
            )
            loss_window.clear()
            window_started_at = time.perf_counter()

        should_evaluate = completed_steps % args.eval_interval == 0 or completed_steps == args.steps
        if should_evaluate:
            last_valid_loss = evaluate(
                model,
                valid_data,
                args.batch_size,
                args.context_length,
                vocab_size,
                args.eval_batches,
                device,
            )
            progress.write(f"step {completed_steps:,}: validation loss = {last_valid_loss:.4f}")
            append_log(
                log_path,
                {"type": "validation", "step": completed_steps, "loss": last_valid_loss},
            )
            if should_log:
                window_started_at = time.perf_counter()

        if completed_steps % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, completed_steps, checkpoint_path)

    save_checkpoint(model, optimizer, args.steps, checkpoint_path)
    print(f"Final validation loss: {last_valid_loss:.4f}", flush=True)
    print(f"Saved checkpoint to: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
