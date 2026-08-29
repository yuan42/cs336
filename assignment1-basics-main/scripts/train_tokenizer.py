from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from cs336_basics.bpe import DEFAULT_CHUNK_SIZE, train_bpe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tinystories_tokenizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE // (1024 * 1024),
        help="Approximate amount of corpus text to read at once (default: 64 MiB).",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help="Special token to add. Repeat this option to add more than one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace tokenizer files that already exist in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")

    output_paths = {
        "vocab": output_dir / "vocab.pkl",
        "merges": output_dir / "merges.pkl",
        "metadata": output_dir / "metadata.json",
    }
    existing_paths = [path for path in output_paths.values() if path.exists()]
    if existing_paths and not args.overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Output already exists: {existing}. Pass --overwrite to replace it.")

    corpus_size = input_path.stat().st_size
    chunk_size = args.chunk_size_mb * 1024 * 1024

    print(f"Training from: {input_path}", flush=True)
    print(f"Corpus size: {corpus_size / (1024**3):.2f} GiB", flush=True)
    print(f"Vocabulary size: {args.vocab_size:,}", flush=True)
    print(f"Streaming chunk size: {args.chunk_size_mb} MiB", flush=True)
    started_at = time.perf_counter()
    vocab, merges = train_bpe(
        input_path,
        args.vocab_size,
        special_tokens,
        show_progress=True,
        chunk_size=chunk_size,
    )
    elapsed_seconds = time.perf_counter() - started_at

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_paths["vocab"].open("wb") as file:
        pickle.dump(vocab, file)
    with output_paths["merges"].open("wb") as file:
        pickle.dump(merges, file)

    metadata = {
        "input_path": str(input_path),
        "input_bytes": corpus_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "number_of_merges": len(merges),
        "special_tokens": special_tokens,
        "chunk_size_bytes": chunk_size,
        "elapsed_seconds": elapsed_seconds,
    }
    with output_paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Finished in {elapsed_seconds:.2f} seconds")
    print(f"Saved tokenizer files to: {output_dir}")


if __name__ == "__main__":
    main()
