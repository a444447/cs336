# own impl

import regex as re
from collections import Counter
import os
from typing import Iterable, Iterator
import multiprocessing as mp
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)


def iter_pretokens(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    if not special_tokens:
        for m in _PAT.finditer(text):
            yield m.group(0)
        return
    if not any(st in text for st in special_tokens):
        for m in _PAT.finditer(text):
            yield m.group(0)
        return

    # Prefer longer matches when special tokens overlap.
    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    special_set = set(special_tokens)
    pattern = "|".join(re.escape(st) for st in special_tokens)
    for chunk in re.split(f"({pattern})", text):
        if not chunk:
            continue
        if chunk in special_set:
            yield chunk
            continue
        for m in _PAT.finditer(chunk):
            yield m.group(0)


def pretoken_to_bytes_seq(pretoken: str, special_tokens: set[str] | None = None) -> tuple[bytes, ...]:
    # 类似于，比如pretoken = "text"，这个函数的作用是把它拆成[b"t", b"e", b"x", b"t"]
    if special_tokens and pretoken in special_tokens:
        return (pretoken.encode("utf-8"),)
    b = pretoken.encode("utf-8")
    return tuple(bytes([x]) for x in b)


def _count_pretokens_in_chunk(args):
    input_path, start, end, special_tokens = args
    local = Counter()
    special_set = set(special_tokens) if special_tokens else None

    # 注意：用 rb + seek/read，和 find_chunk_boundaries 的字节 offset 对齐
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    # Normalize line endings to match the reference fixtures.
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    # 这里才是真正的 pre-tokenization（regex PAT + finditer）
    for pre in iter_pretokens(chunk, special_tokens=special_tokens):
        local[pretoken_to_bytes_seq(pre, special_tokens=special_set)] += 1

    return local


def _merge_counters(counters):
    out = Counter()
    for c in counters:
        out.update(c)
    return out


def get_pair_counts(word_counts):
    """
    假设语料是:
    text text text text text
    text text text text text
    那么pre-token = "text",出现次数 = 10 次
    count就会是word_counts[("t","e","x","t")] = 10
    get_pair_counts函数就是统计相邻 token pair 的频次，只不过因为 "text" 出现了 10 次而 "text" 里 确实有一对 't' 'e'，所以：'t','e' 这个 pair 的计数 +10
    不需要在原始语料里扫 10 次
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, c in word_counts.items():
        if c <= 0 or len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += c
    return pair_counts


def merge_word_once(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of `pair` inside `word` in a single left-to-right pass."""
    a, b = pair
    out: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def apply_merge_to_corpus(
    word_counts,
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    """Apply one merge to every word type in the corpus, preserving counts."""
    new_counts: Counter[tuple[bytes, ...]] = Counter()
    for word, c in word_counts.items():
        new_counts[merge_word_once(word, pair)] += c
    return new_counts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must more than zero !")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    next_id = 256  # 下一个新的token 分配的id

    # 先把special token加入到vocab里面
    for st in special_tokens:
        sb = st.encode("utf-8")
        if sb in vocab.values():
            continue
        vocab[next_id] = sb
        next_id += 1

    if len(vocab) >= vocab_size:
        return vocab, merges

    num_processes = 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = list(
        zip(
            [input_path] * (len(boundaries) - 1),
            boundaries[:-1],
            boundaries[1:],
            [special_tokens] * (len(boundaries) - 1),
        )
    )

    if num_processes <= 1:
        parts = [_count_pretokens_in_chunk(t) for t in tasks]
    else:
        with mp.Pool(processes=num_processes) as pool:
            parts = pool.map(_count_pretokens_in_chunk, tasks)
    word_counts = _merge_counters(parts)

    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(word_counts)
        if not pair_counts:
            break
        max_freq = max(pair_counts.values())
        # Deterministic tie-break: choose lexicographically greatest pair among max-frequency pairs
        best_candidates = [p for p, v in pair_counts.items() if v == max_freq]
        best_pair = max(best_candidates)

        new_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        word_counts = apply_merge_to_corpus(word_counts, best_pair)
        vocab[next_id] = new_token
        next_id += 1

    return vocab, merges

if __name__ == '__main__':
    vocab, merges = run_train_bpe("E:\githome\cs336\data\debug_corpus.txt", vocab_size=256+6, special_tokens=["<|endoftext|>"])
    print(merges[:5])