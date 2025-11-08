import os
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool

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
        # 临时日志：输出合并信息
        print(f"合并 token {p1} ({vocab[p1]!r}) + token {p2} ({vocab[p2]!r}) => 新 token id {new_token_id}")
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
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens.
        Args:
            vocab (dict[int, bytes]): The vocabulary to use.
            merges (list[tuple[bytes, bytes]]): The list of merges to use.
            special_tokens (list[str] | None, optional): The list of special tokens to use. Defaults to None.
        Returns:
            Tokenizer: The tokenizer.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_reverse = {v: k for k, v in self.vocab.items}
        
        # 对于不存在的special token，放入vocab
        for special_token in self.special_tokens:
            if not vocab_reverse[special_token]:
                token_id = len(self.vocab)
                vocab[token_id] = special_token.encode('utf-8')
        
        self.special_token_ids = {token: self.vocab_reverse[token.encode('utf-8')] for token in self.special_tokens}

        
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary file and merges file.
        Args:
            vocab_filepath (str): The path to the vocabulary file.
            merges_filepath (str): The path to the merges file.
            special_tokens (list[str] | None, optional): The list of special tokens to use. Defaults to None.
        Returns:
            Tokenizer: The tokenizer.
        """
        # # 方案一：对于latin-1存储，兼容控制字符存储在json中的情况
        # with open(vocab_filepath, "r", encoding="utf-8") as f:
        #     raw_vocab = json.load(f)

        # vocab = {}
        # for token_id_str, token_val in raw_vocab.items():
        #     # 用 latin-1 编码回去，完美还原原始 bytes
        #     vocab[int(token_id_str)] = token_val.encode("latin-1")
        
        # # 方案二：# 在加载后手动修复前 256 个，这种方案是错误的，因为没办法保证后续被合并的bytes里面是否也有非法的utf-8字符、组合
        # for i in range(256):
        #     vocab[i] = bytes([i])

        vocab = pickle.load(open(vocab_filepath, "rb"))
        merges = pickle.load(open(merges_filepath, "rb"))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a sequence of token IDs.
        Args:
            text (str): The string to encode.
        Returns:
            list[int]: The list of token IDs.
        """
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-effcient tokenization of large files that we cannot directly load into memory.
        Args:
            iterable (Iterable[str]): The iterable of strings to encode.
        Returns:
            Iterator[int]: The iterator of token IDs.
        """
        pass
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        Args:
            ids (list[int]): The list of token IDs to decode.
        Returns:
            str: The decoded string.
        """
        pass
        