import os
import json
import pickle
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from typing import Iterable, Iterator

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
    num_proc: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Step 1: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        vocab[len(vocab)] = token_str.encode("utf-8")
    
    # Step 2: Pre-tokenization (支持并行)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    delimiter = "<|endoftext|>"
    if delimiter in special_tokens:
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_special_tokens = [re.escape(st) for st in sorted_special_tokens]
        special_token_pattern = re.compile("|".join(escaped_special_tokens))
        chunks = special_token_pattern.split(text)
    else:
        chunks = [text]
    
    chunks = [c for c in chunks if c]

    word_freqs = Counter()
    if num_proc is not None and num_proc > 1:
        with Pool(processes=num_proc) as pool:
            chunk_counters = pool.map(_pretokenize_chunk, chunks)
        for chunk_counter in chunk_counters:
            word_freqs.update(chunk_counter)
    else:
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
        # print(f"合并 token {p1} ({vocab[p1]!r}) + token {p2} ({vocab[p2]!r}) => 新 token id {new_token_id}")
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


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 1、构建反向词汇表，用于 encode 时查找 ID
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        
        # 2、确保所有特殊 Token 都在词汇表中，并建立快速查找表
        self.special_token_ids = {}
        for st in self.special_tokens:
            st_bytes = st.encode('utf-8')
            if st_bytes not in self.vocab_reverse:
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self.vocab_reverse[st_bytes] = new_id
            self.special_token_ids[st] = self.vocab_reverse[st_bytes]

        # 3. 预编译特殊 Token 的正则模式，用于 encode 时的文本切分
        if self.special_tokens:
            sorted_st = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = re.compile("|".join(re.escape(st) for st in sorted_st))
        else:
            self.special_pat = None
        
        # 4. 预编译 GPT-2 预分词正则
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # 假设 vocab 用 pickle 保存以保持二进制安全
        with open(vocab_filepath, "rb") as f:
             vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
             merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _tokenize_word(self, word_bytes: bytes) -> list[int]:
        # 初始状态：将单词视为单字节 ID 列表
        ids = [self.vocab_reverse[bytes([b])] for b in word_bytes]
        
        while len(ids) >= 2:
            # 找出当前序列中所有可能的 pair，并找到在 merges 中最早出现的那个
            stats = {}
            for i in range(len(ids) - 1):
                pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                stats[pair] = i

            best_pair = None
            
            # 在训练好的 merges 列表中，找到第一个（优先级最高）且当前实际存在的 pair
            for i, merge in enumerate(self.merges):
                if merge in stats:
                    # 找到了一个可合并的 pair
                    best_pair = merge
                    break
            
            if best_pair is None:
                break # 没有更多可合并的 pair

            # 执行合并
            p1, p2 = best_pair
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and self.vocab[ids[i]] == p1 and self.vocab[ids[i+1]] == p2:
                    new_ids.append(self.vocab_reverse[p1 + p2])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids

    def encode(self, text: str) -> list[int]:
        final_ids = []
        
        # 1. 使用特殊 token 切分文本
        if self.special_pat:
            segments = self.special_pat.split(text)
            # split 结果中会包含分隔符（如果用了捕获组），这里需要重新确认一下哪些是特殊token
            # 更简单的方法是手动遍历或使用 finditer 来定位特殊 token
            # 这里采用一种简化的两步法：先 split，再检查
            
            # 为了保留特殊 token，使用捕获组
            special_pat_with_group = re.compile(f"({self.special_pat.pattern})")
            segments = special_pat_with_group.split(text)
        else:
            segments = [text]
            
        for segment in segments:
            if not segment: continue
            
            if segment in self.special_token_ids:
                final_ids.append(self.special_token_ids[segment])
            else:
                # 普通文本：使用 GPT-2 正则进行预分词
                for match in self.pat.finditer(segment):
                    word_bytes = match.group(0).encode('utf-8')
                    final_ids.extend(self._tokenize_word(word_bytes))
                    
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            # 可以选择在这里处理未知的 ID，或者忽略
        
        full_bytes = b"".join(byte_parts)
        return full_bytes.decode('utf-8', errors='replace')