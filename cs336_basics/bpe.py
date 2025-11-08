import os
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import functools

def initialize_stats(word_freqs: dict[tuple[int, ...], int]) -> tuple[
    defaultdict[tuple[int, int], int],
    defaultdict[tuple[int, int], set[tuple[int, ...]]],
]:
    """Initialize pair frequencies and the inverted index."""
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            pair_freqs[p] += freq
            pair_to_words[p].add(word)
    return pair_freqs, pair_to_words

def remove_word_stats(word, freq, pair_freqs, pair_to_words):
    """Remove a word's contribution from global stats."""
    for i in range(len(word) - 1):
        p = (word[i], word[i+1])
        pair_freqs[p] -= freq
        if pair_freqs[p] == 0:
            del pair_freqs[p]
        pair_to_words[p].discard(word)
        if not pair_to_words[p]:
             del pair_to_words[p]

def add_word_stats(word, freq, pair_freqs, pair_to_words):
    """Add a word's contribution to global stats."""
    for i in range(len(word) - 1):
        p = (word[i], word[i+1])
        pair_freqs[p] += freq
        pair_to_words[p].add(word)

def merge_pair_in_word(word: tuple[int, ...], pair: tuple[int, int], new_token_id: int) -> tuple[int, ...]:
    """Merge a specific pair in a single word."""
    new_word = []
    i = 0
    p1, p2 = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def _pretokenize_chunk(text_chunk: str) -> Counter:
    """Worker function for parallel pre-tokenization."""
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    chunk_freqs = Counter()
    for match in PAT.finditer(text_chunk):
        word_bytes = match.group(0).encode('utf-8')
        chunk_freqs[tuple(word_bytes)] += 1
    return chunk_freqs

def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_proc: int | None = None, # 新增可选参数控制并行
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Step 1: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        vocab[len(vocab)] = token_str.encode("utf-8")
    
    # Step 2: Pre-tokenization (支持并行)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 使用特殊 token 作为天然的分块边界
    delimiter = "<|endoftext|>"
    if delimiter in special_tokens:
        # 使用正则split以确保特殊token被正确处理（虽然这里我们只用它来分块）
        # 如果能确定输入格式良好，简单的 text.split(delimiter) 可能也够用
        # 为了稳妥，复用之前的特殊token隔离逻辑来分块
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_special_tokens = [re.escape(st) for st in sorted_special_tokens]
        special_token_pattern = re.compile("|".join(escaped_special_tokens))
        chunks = special_token_pattern.split(text)
    else:
        chunks = [text]
    
    # 去除空块
    chunks = [c for c in chunks if c]

    word_freqs = Counter()
    if num_proc is not None and num_proc > 1:
        # 并行路径
        with Pool(processes=num_proc) as pool:
            chunk_counters = pool.map(_pretokenize_chunk, chunks)
        for chunk_counter in chunk_counters:
            word_freqs.update(chunk_counter)
    else:
        # 单进程路径 (复用 worker 函数逻辑或保持原样)
        for chunk in chunks:
            word_freqs.update(_pretokenize_chunk(chunk))

    # Step 3: Initialize stats
    pair_freqs, pair_to_words = initialize_stats(word_freqs)

    # Step 4: Main merging loop
    merges = []
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break

        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        
        new_token_id = len(vocab)
        p1, p2 = best_pair
        vocab[new_token_id] = vocab[p1] + vocab[p2]
        merges.append((vocab[p1], vocab[p2]))

        affected_words = list(pair_to_words[best_pair])
        for old_word in affected_words:
            freq = word_freqs[old_word]
            remove_word_stats(old_word, freq, pair_freqs, pair_to_words)
            new_word = merge_pair_in_word(old_word, best_pair, new_token_id)
            add_word_stats(new_word, freq, pair_freqs, pair_to_words)
            del word_freqs[old_word]
            word_freqs[new_word] += freq

    return vocab, merges