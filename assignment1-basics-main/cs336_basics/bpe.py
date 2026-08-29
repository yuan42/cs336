from __future__ import annotations

import heapq
import math
import os
from collections import defaultdict
from collections.abc import Iterator
from typing import BinaryIO

import regex as re
from tqdm.auto import tqdm

DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024


class PreToken:
    def __init__(self, tokens: tuple[bytes, ...], count: int):
        self.tokens = tokens
        self.count = count


class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: ReversePair) -> bool:
        return self.pair > other.pair


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_token: bytes,
) -> list[int]:
    """Find independent corpus chunks whose boundaries start at a special token."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    search_size = 4096
    for boundary_index in range(1, len(boundaries) - 1):
        position = boundaries[boundary_index]
        file.seek(position)
        overlap = b""

        while True:
            block = file.read(search_size)
            if not block:
                boundaries[boundary_index] = file_size
                break

            searchable = overlap + block
            token_index = searchable.find(split_token)
            if token_index != -1:
                boundaries[boundary_index] = position - len(overlap) + token_index
                break
            overlap = searchable[-(len(split_token) - 1) :] if len(split_token) > 1 else b""
            position += len(block)

    return sorted(set(boundaries))


def _iter_corpus_chunks(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    chunk_size: int,
) -> Iterator[tuple[str, int]]:
    """Yield independently pre-tokenizable text chunks and their byte sizes."""
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as file:
        if special_tokens and file_size > chunk_size:
            desired_num_chunks = math.ceil(file_size / chunk_size)
            boundaries = _find_chunk_boundaries(
                file,
                desired_num_chunks,
                special_tokens[0].encode("utf-8"),
            )
        else:
            boundaries = [0, file_size]

        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            file.seek(start)
            chunk_bytes = file.read(end - start)
            yield chunk_bytes.decode("utf-8"), len(chunk_bytes)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    show_progress: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merges."""
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size} to fit all byte tokens and special tokens")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pre_token_count: dict[tuple[bytes, ...], int] = defaultdict(int)
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    special_regex = re.compile(special_pattern) if special_pattern else None
    pre_token_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    file_size = os.path.getsize(input_path)
    with tqdm(
        total=file_size,
        desc="Pre-tokenizing",
        unit="B",
        unit_scale=True,
        disable=not show_progress,
    ) as progress:
        for text, bytes_read in _iter_corpus_chunks(input_path, special_tokens, chunk_size):
            chunks = special_regex.split(text) if special_regex else [text]
            for chunk in chunks:
                for match in pre_token_regex.finditer(chunk):
                    pre_token = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
                    pre_token_count[pre_token] += 1
            progress.update(bytes_read)

    pre_tokens = [PreToken(tokens, count) for tokens, count in pre_token_count.items()]

    vocab = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    pair_locations: dict[tuple[bytes, bytes], set[PreToken]] = defaultdict(set)
    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
    heap: list[tuple[int, ReversePair]] = []

    for pre_token in tqdm(
        pre_tokens,
        desc="Counting pairs",
        unit="pre-token",
        disable=not show_progress,
    ):
        for index in range(len(pre_token.tokens) - 1):
            pair = (pre_token.tokens[index], pre_token.tokens[index + 1])
            pair_count[pair] += pre_token.count
            pair_locations[pair].add(pre_token)

    for pair, count in pair_count.items():
        heapq.heappush(heap, (-count, ReversePair(pair)))

    def merge_pair(pair: tuple[bytes, bytes]) -> None:
        merged_token = pair[0] + pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(pair)

        for pre_token in pair_locations[pair]:
            count = pre_token.count
            tokens = pre_token.tokens
            index = 0

            while index < len(tokens) - 1:
                if tokens[index] != pair[0] or tokens[index + 1] != pair[1]:
                    index += 1
                    continue

                if index > 0:
                    old_pair = (tokens[index - 1], tokens[index])
                    pair_count[old_pair] -= count
                    heapq.heappush(heap, (-pair_count[old_pair], ReversePair(old_pair)))
                if index < len(tokens) - 2:
                    old_pair = (tokens[index + 1], tokens[index + 2])
                    pair_count[old_pair] -= count
                    heapq.heappush(heap, (-pair_count[old_pair], ReversePair(old_pair)))
                pair_count[pair] -= count
                heapq.heappush(heap, (-pair_count[pair], ReversePair(pair)))

                tokens = tokens[:index] + (merged_token,) + tokens[index + 2 :]
                if index > 0:
                    new_pair = (tokens[index - 1], tokens[index])
                    pair_count[new_pair] += count
                    heapq.heappush(heap, (-pair_count[new_pair], ReversePair(new_pair)))
                    pair_locations[new_pair].add(pre_token)
                if index < len(tokens) - 1:
                    new_pair = (tokens[index], tokens[index + 1])
                    pair_count[new_pair] += count
                    heapq.heappush(heap, (-pair_count[new_pair], ReversePair(new_pair)))
                    pair_locations[new_pair].add(pre_token)

                pre_token.tokens = tokens
                index += 1

    with tqdm(
        total=vocab_size - len(vocab),
        desc="Learning merges",
        unit="merge",
        disable=not show_progress,
    ) as progress:
        while len(vocab) < vocab_size and heap:
            reverse_count, reverse_pair = heapq.heappop(heap)
            count = -reverse_count
            pair = reverse_pair.pair
            if count != pair_count[pair] or count == 0:
                continue
            merge_pair(pair)
            progress.update()

    return vocab, merges
