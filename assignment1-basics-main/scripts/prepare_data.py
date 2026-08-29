from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TOKENIZER_DIR = DATA_DIR / "tinystories_tokenizer"
DEFAULT_TRAIN_INPUT = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VALID_INPUT = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize TinyStories into memory-mappable NumPy arrays.")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--valid-input", type=Path, default=DEFAULT_VALID_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=1_000_000,
        help="Number of token IDs to buffer before writing them to disk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prepared arrays and metadata that already exist.",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_dir: Path) -> tuple[Tokenizer, dict]:
    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"
    metadata_path = tokenizer_dir / "metadata.json"

    missing_paths = [path for path in (vocab_path, merges_path, metadata_path) if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Tokenizer is incomplete; missing: {missing}")

    with vocab_path.open("rb") as file:
        vocab = pickle.load(file)
    with merges_path.open("rb") as file:
        merges = pickle.load(file)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    special_tokens = metadata.get("special_tokens", ["<|endoftext|>"])
    return Tokenizer(vocab, merges, special_tokens), metadata


def choose_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError(f"Vocabulary of size {vocab_size:,} is too large for uint32 token IDs")


def tokenize_to_npy(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    buffer_tokens: int,
) -> int:
    """Encode a text file without holding the corpus or all token IDs in memory."""
    input_bytes = input_path.stat().st_size
    token_count = 0
    token_buffer: list[int] = []
    raw_temp_path: Path | None = None
    npy_temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tokens.tmp",
            delete=False,
        ) as raw_file:
            raw_temp_path = Path(raw_file.name)
            with input_path.open(encoding="utf-8") as input_file:
                with tqdm(
                    total=input_bytes,
                    desc=f"Tokenizing {input_path.stem}",
                    unit="B",
                    unit_scale=True,
                ) as progress:
                    for text in input_file:
                        token_buffer.extend(tokenizer.encode(text))
                        progress.update(len(text.encode("utf-8")))

                        if len(token_buffer) >= buffer_tokens:
                            tokens = np.asarray(token_buffer, dtype=dtype)
                            tokens.tofile(raw_file)
                            token_count += len(tokens)
                            token_buffer.clear()

                    if token_buffer:
                        tokens = np.asarray(token_buffer, dtype=dtype)
                        tokens.tofile(raw_file)
                        token_count += len(tokens)
                        token_buffer.clear()

        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".npy.tmp",
            delete=False,
        ) as npy_temp_file:
            npy_temp_path = Path(npy_temp_file.name)

        output_array = np.lib.format.open_memmap(
            npy_temp_path,
            mode="w+",
            dtype=dtype,
            shape=(token_count,),
        )
        if token_count:
            raw_array = np.memmap(raw_temp_path, mode="r", dtype=dtype, shape=(token_count,))
            copy_size = max(1, buffer_tokens)
            with tqdm(total=token_count, desc=f"Writing {output_path.name}", unit="token") as progress:
                for start in range(0, token_count, copy_size):
                    end = min(start + copy_size, token_count)
                    output_array[start:end] = raw_array[start:end]
                    progress.update(end - start)
            del raw_array
        output_array.flush()
        del output_array

        os.replace(npy_temp_path, output_path)
        npy_temp_path = None
        return token_count
    finally:
        if raw_temp_path is not None:
            raw_temp_path.unlink(missing_ok=True)
        if npy_temp_path is not None:
            npy_temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_inputs = {
        "train": args.train_input.expanduser().resolve(),
        "valid": args.valid_input.expanduser().resolve(),
    }
    split_outputs = {name: output_dir / f"tinystories_{name}.npy" for name in split_inputs}
    metadata_path = output_dir / "tinystories_data.json"

    if args.buffer_tokens <= 0:
        raise ValueError("--buffer-tokens must be positive")
    for input_path in split_inputs.values():
        if not input_path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {input_path}")

    existing_paths = [path for path in (*split_outputs.values(), metadata_path) if path.exists()]
    if existing_paths and not args.overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Prepared data already exists: {existing}. Pass --overwrite to replace it.")

    tokenizer, tokenizer_metadata = load_tokenizer(tokenizer_dir)
    dtype = choose_dtype(len(tokenizer.vocab))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizer: {tokenizer_dir}", flush=True)
    print(f"Vocabulary size: {len(tokenizer.vocab):,}", flush=True)
    print(f"Token dtype: {dtype.name} ({dtype.itemsize} bytes/token)", flush=True)

    started_at = time.perf_counter()
    split_metadata = {}
    for split_name, input_path in split_inputs.items():
        output_path = split_outputs[split_name]
        split_started_at = time.perf_counter()
        token_count = tokenize_to_npy(
            input_path,
            output_path,
            tokenizer,
            dtype,
            args.buffer_tokens,
        )
        split_metadata[split_name] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_bytes": input_path.stat().st_size,
            "token_count": token_count,
            "elapsed_seconds": time.perf_counter() - split_started_at,
        }
        print(f"{split_name}: wrote {token_count:,} tokens to {output_path}", flush=True)

    metadata = {
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": tokenizer_metadata.get("special_tokens", []),
        "dtype": dtype.name,
        "total_elapsed_seconds": time.perf_counter() - started_at,
        "splits": split_metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Saved metadata to: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
