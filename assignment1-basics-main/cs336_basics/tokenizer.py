from __future__ import annotations
import json
import regex as re
from typing import Iterable, Iterator

class Tokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)
        self.special_pattern = ""
        if special_tokens:
            sorted_special_tokens = sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
            self.special_pattern = "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token_bytes

        self.vocab2Id = {val: key for key, val in vocab.items()}
        self.merge_map = {}
        for rank, t in enumerate(self.merges):
            self.merge_map[t] = rank
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, val in data.items():
                vocab[int(val)] = str(key).encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 2:
                    continue

                token1, token2 = parts
                merges.append(
                    (
                        token1.encode("utf-8"),
                        token2.encode("utf-8"),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _match_merge(self, seq: tuple[bytes, ...]) -> int:
        rank = float('inf')
        idx = -1
        for i in range(len(seq) - 1):
            temp = (seq[i], seq[i+1])
            if temp in self.merge_map and self.merge_map[temp] < rank:
                rank = self.merge_map[temp]
                idx = i

        return idx

    def encode(self, text: str) -> list[int]:
        tokens = [] 
        chunks = re.split(self.special_pattern, text) if self.special_pattern else [text]
        for chunk in chunks:
            if chunk in self.special_token_set:
                tokens.append(self.vocab2Id[chunk.encode("utf-8")])
                continue
            for match in re.finditer(self.PAT, chunk):
                temp = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                while (idx := self._match_merge(temp)) >= 0:
                    temp = temp[:idx] + (temp[idx] + temp[idx + 1],) + temp[idx + 2:]
                for token in temp:
                    tokens.append(self.vocab2Id[token])

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[id] for id in ids)
        return byte_seq.decode("utf-8", errors="replace")

    