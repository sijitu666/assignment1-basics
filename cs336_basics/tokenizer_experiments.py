import argparse
import time
from pathlib import Path
from typing import List

from cs336_basics.bpe import Tokenizer

# 数据与 artifact 路径（相对项目根目录）
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")

TINYSTORIES_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
OWT_TRAIN = DATA_DIR / "owt_train.txt"

TINYSTORIES_VOCAB_PKL = ARTIFACTS_DIR / "tinystories_vocab.pkl"
TINYSTORIES_MERGES_PKL = ARTIFACTS_DIR / "tinystories_merges.pkl"

OWT_VOCAB_PKL = ARTIFACTS_DIR / "owt_vocab.pkl"
OWT_MERGES_PKL = ARTIFACTS_DIR / "owt_merges.pkl"

SPECIAL_TOKENS = ["<|endoftext|>"]


def load_tokenizer(kind: str) -> Tokenizer:
    """加载指定类型的 tokenizer。
    kind: 'tinystories' 或 'owt'
    """
    kind = kind.lower()
    if kind == "tinystories":
        vocab_path, merges_path = TINYSTORIES_VOCAB_PKL, TINYSTORIES_MERGES_PKL
    elif kind == "owt":
        vocab_path, merges_path = OWT_VOCAB_PKL, OWT_MERGES_PKL
    else:
        raise ValueError("kind 必须是 'tinystories' 或 'owt'")

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"未找到 tokenizer artifact，请先运行训练脚本生成：\n"
            f"  vocab: {vocab_path}\n  merges: {merges_path}"
        )

    return Tokenizer.from_files(str(vocab_path), str(merges_path), SPECIAL_TOKENS)


def sample_documents(file_path: Path, n: int = 10, delimiter: str = "<|endoftext|>") -> List[str]:
    """从语料中抽样 n 个文档（优先使用特殊 token 边界，否则退化为按空行或单行）。"""
    with file_path.open("r", encoding="utf-8") as f:
        text = f.read()

    docs: List[str]
    if delimiter in text:
        docs = [d for d in text.split(delimiter) if d.strip()]
    else:
        # 退化策略：按双换行或单行分段
        parts = [p.strip("\n") for p in text.split("\n\n") if p.strip()]
        if len(parts) >= n:
            docs = parts
        else:
            docs = [line for line in text.splitlines() if line.strip()]

    return docs[:n]


def compute_compression_ratio(texts: List[str], tokenizer: Tokenizer) -> tuple[int, int, float]:
    """计算压缩比：bytes/token。
    返回 (总字节数, 总 token 数, 压缩比)。"""
    total_bytes = 0
    total_tokens = 0
    for t in texts:
        total_bytes += len(t.encode("utf-8"))
        total_tokens += len(tokenizer.encode(t))
    ratio = (total_bytes / total_tokens) if total_tokens > 0 else float("inf")
    return total_bytes, total_tokens, ratio


def estimate_throughput(texts: List[str], tokenizer: Tokenizer) -> tuple[float, float, int]:
    """估算吞吐：返回 (bytes/sec, tokens/sec, 总 tokens)。"""
    total_bytes = 0
    start = time.time()
    total_tokens = 0
    for t in texts:
        total_bytes += len(t.encode("utf-8"))
        total_tokens += len(tokenizer.encode(t))
    elapsed = max(time.time() - start, 1e-9)
    return total_bytes / elapsed, total_tokens / elapsed, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenizer 实验脚本")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], required=True, help="选择要抽样的语料")
    parser.add_argument("--tokenizer", choices=["tinystories", "owt"], required=True, help="选择使用的 tokenizer")
    parser.add_argument("--num_docs", type=int, default=10, help="抽样文档数量")
    parser.add_argument("--throughput", action="store_true", help="是否测吞吐")
    args = parser.parse_args()

    # 选择语料路径
    if args.dataset == "tinystories":
        corpus_path = TINYSTORIES_TRAIN
    else:
        corpus_path = OWT_TRAIN

    if not corpus_path.exists():
        raise FileNotFoundError(f"未找到语料文件：{corpus_path}")

    tokenizer = load_tokenizer(args.tokenizer)
    docs = sample_documents(corpus_path, n=args.num_docs)

    total_bytes, total_tokens, ratio = compute_compression_ratio(docs, tokenizer)
    print(f"样本文档数: {len(docs)}")
    print(f"总字节数: {total_bytes}")
    print(f"总 token 数: {total_tokens}")
    print(f"压缩比 (bytes/token): {ratio:.4f}")

    if args.throughput:
        bps, tps, ntok = estimate_throughput(docs, tokenizer)
        print(f"吞吐: {bps:,.2f} bytes/s, {tps:,.2f} tokens/s (总 tokens={ntok})")


if __name__ == "__main__":
    main()
