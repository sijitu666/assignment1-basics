import os
import regex as re
from collections import defaultdict

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
        # FIX: Use discard instead of remove to handle words with duplicate pairs
        # 当一个词中包含多次相同的字节对时（例如单词中出现两次 ('t', 'h')），remove_word_stats 函数在第二次尝试
        # 从 pair_to_words 集合中移除同一个词时会抛出 KeyError，因为第一次操作已经将其移除了
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

def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Step 1: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = token_str.encode("utf-8")
    
    # Step 2: Pre-tokenization with special token handling
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    if special_tokens:
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_special_tokens = [re.escape(st) for st in sorted_special_tokens]
        special_token_pattern = re.compile("|".join(escaped_special_tokens))
        text_segments = special_token_pattern.split(text)
    else:
        text_segments = [text]

    word_freqs = defaultdict(int)
    for segment in text_segments:
        if not segment: continue
        for match in pat.finditer(segment):
            word_bytes = match.group(0).encode('utf-8')
            word_tuple = tuple(b for b in word_bytes)
            word_freqs[word_tuple] += 1

    # Step 3: Initialize stats once
    pair_freqs, pair_to_words = initialize_stats(word_freqs)

    # Step 4: Main merging loop with incremental updates
    merges = []
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break

        # 4a. Find best pair with correct tie-breaking
        # Use a tuple of (freq, p0_bytes, p1_bytes) for comparison to ensure deterministic tie-breaking
        # by actual byte content, not just integer IDs.
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        
        # 4b. Create new token
        new_token_id = len(vocab)
        p1, p2 = best_pair
        vocab[new_token_id] = vocab[p1] + vocab[p2]
        merges.append((vocab[p1], vocab[p2]))

        # 4c. Incremental update for affected words
        affected_words = list(pair_to_words[best_pair])
        
        for old_word in affected_words:
            freq = word_freqs[old_word]
            
            # 1. Remove old word stats
            remove_word_stats(old_word, freq, pair_freqs, pair_to_words)
            
            # 2. Compute new word structure
            new_word = merge_pair_in_word(old_word, best_pair, new_token_id)
            
            # 3. Add new word stats
            add_word_stats(new_word, freq, pair_freqs, pair_to_words)
            
            # 4. Update word_freqs dictionary
            del word_freqs[old_word]
            word_freqs[new_word] += freq

    return vocab, merges