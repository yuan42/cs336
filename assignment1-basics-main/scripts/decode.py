from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from cs336_basics.nn import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TOKENIZER_DIR = DATA_DIR / "tinystories_tokenizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained TinyStories checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to a training checkpoint such as latest.pt.")
    parser.add_argument("--config", type=Path, help="Defaults to config.json beside the checkpoint.")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_tokenizer(tokenizer_dir: Path) -> tuple[Tokenizer, list[str]]:
    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"
    metadata_path = tokenizer_dir / "metadata.json"
    for path in (vocab_path, merges_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

    with vocab_path.open("rb") as file:
        vocab = pickle.load(file)
    with merges_path.open("rb") as file:
        merges = pickle.load(file)
    metadata = load_json(metadata_path)
    special_tokens = metadata.get("special_tokens", ["<|endoftext|>"])
    return Tokenizer(vocab, merges, special_tokens), special_tokens


def load_model(
    checkpoint_path: Path,
    config: dict,
    device: torch.device,
) -> tuple[TransformerLM, int]:
    model_config = config["model"]
    model = TransformerLM(
        vocab_size=config["data"]["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device,
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint["iteration"])


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    eos_token_id: int | None,
    device: torch.device,
) -> str:
    """Generate from one prompt using temperature sampling."""
    generated_ids = tokenizer.encode(prompt)
    if not generated_ids:
        raise ValueError("The prompt must encode to at least one token")

    for _ in range(max_new_tokens):
        context_ids = generated_ids[-context_length:]
        input_ids = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = softmax(next_token_logits, dim=-1)
        next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_ids.append(next_token_id)

    return tokenizer.decode(generated_ids)



def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else checkpoint_path.parent / "config.json"
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    config = load_json(config_path)
    tokenizer, special_tokens = load_tokenizer(tokenizer_dir)
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    model, checkpoint_iteration = load_model(checkpoint_path, config, device)

    eos_token_id = None
    if special_tokens:
        eos_token_id = tokenizer.vocab2Id.get(special_tokens[0].encode("utf-8"))

    print(f"Device: {device}", flush=True)
    print(f"Checkpoint iteration: {checkpoint_iteration:,}")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=config["model"]["context_length"],
        temperature=args.temperature,
        eos_token_id=eos_token_id,
        device=device,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
