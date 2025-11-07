import os
import regex as re
from collections import defaultdict

def get_pair_freqs(word_freqs: dict[tuple[int, ...], int]) -> defaultdict[tuple[int, int], int]:
    """Helper function to calculate frequencies of adjacent byte pairs."""
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i+1])] += freq
    return pair_freqs

def merge_pair(
    pair: tuple[int, int],
    word_freqs_in: dict[tuple[int, ...], int],
    new_token_id: int
) -> dict[tuple[int, ...], int]:
    """Helper function to merge a pair in all words and return the new word frequencies."""
    word_freqs_out = {}
    (p1, p2) = pair
    for word, freq in word_freqs_in.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word_freqs_out[tuple(new_word)] = freq
    return word_freqs_out


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Step 1: Initialize vocabulary with basic bytes and special tokens.
    # The first 256 tokens are the raw bytes.
    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = token_str.encode("utf-8")
    
    # Step 2: Read data and perform pre-tokenization.
    # Remember to handle splitting by special tokens first, then use the regex.
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # The GPT-2 pre-tokenization regex pattern
    pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # === 新增：处理特殊 Token 的隔离 ===
    if special_tokens:
        # 创建一个匹配任何特殊 token 的正则表达式，注意要转义(escape)特殊字符
        # 按照长度降序排列，以确保先匹配较长的特殊 token（例如 '<|endoftext|>' 优于 '<|'）
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_special_tokens = [re.escape(st) for st in sorted_special_tokens]
        special_token_pattern = re.compile("|".join(escaped_special_tokens))
        
        # 使用特殊 token 将文本切分为若干片段
        text_segments = special_token_pattern.split(text)
    else:
        text_segments = [text]

    # === 修改：只对普通文本片段进行预分词统计 ===
    word_freqs = defaultdict(int)
    for segment in text_segments:
        if not segment: continue # 跳过空片段
        for match in pat.finditer(segment):
            word_bytes = match.group(0).encode('utf-8')
            word_tuple = tuple(b for b in word_bytes)
            word_freqs[word_tuple] += 1

    # Step 3: Calculate initial pair frequencies from the pre-tokenized words.
    # This is the "indexing" step.
    pair_freqs = get_pair_freqs(word_freqs)

    # Step 4: The main merging loop.
    # Loop until the vocabulary reaches the desired size.
    merges = []
    while len(vocab) < vocab_size:
        # 4a. Find the most frequent pair.
        if not pair_freqs:
            break # No more pairs to merge
        # In case of ties, Python's max() on tuples will handle lexicographical breaking
        # best_pair = max(pair_freqs, key=pair_freqs.get)
        # 错误的写法：best_pair = max(pair_freqs, key=lambda x: (pair_freqs[x], x)) # 构建(它的频率, 它本身)元组作为比较依据
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]])) # 相同频率依次比较第一个 Token 和第二个 Token 的实际内容
        
        # 4b. Create the new merged token.
        new_token_id = len(vocab)
        p1, p2 = best_pair
        new_token_bytes = vocab[p1] + vocab[p2]
        
        # 4c. Update vocabulary and merges list.
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[p1], vocab[p2]))

        # 4d. Update the word frequencies by replacing the merged pair.
        word_freqs = merge_pair(best_pair, word_freqs, new_token_id)
        
        # 4e. Recalculate pair frequencies for the next iteration.
        # This is the "incremental update" part. For simplicity here we recalculate,
        # but the optimal solution would update counts incrementally.
        pair_freqs = get_pair_freqs(word_freqs)
        
    # Step 5: Return the final vocabulary and merges.
    return vocab, merges