import json
import pickle
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe_tokenizer

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

def main():
    start = time.time()
    vocab, merges = train_bpe_tokenizer(
        input_path=TRAIN_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed = time.time() - start
    print(f"训练耗时 {elapsed/60:.2f} 分钟，生成 vocab 中最长的 token 是 {max(vocab.values(), key=len)}")

    # 序列化保存，便于后续作业检查
    vocab_path = Path("artifacts") / "tinystories_vocab.json"
    merges_path = Path("artifacts") / "tinystories_merges.pkl"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({idx: token.decode("utf-8", errors="replace") for idx, token in vocab.items()}, f)

    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    main()