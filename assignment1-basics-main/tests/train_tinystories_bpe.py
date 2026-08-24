import pickle
import time

from adapters import run_train_bpe

start = time.time()

vocab, merges = run_train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=1_000,
    special_tokens=["<|endoftext|>"],
)

elapsed = time.time() - start

print(f"Training time: {elapsed:.2f}s")

longest_id, longest_token = max(
    vocab.items(),
    key=lambda item: len(item[1]),
)

print("Longest token ID:", longest_id)
print("Longest token:", repr(longest_token))
print("Longest token length:", len(longest_token))

with open("tinystories_bpe.pkl", "wb") as f:
    pickle.dump((vocab, merges), f)
