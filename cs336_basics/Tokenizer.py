import os
import json
import regex as re
from typing import Iterable, Iterator, Any
from .bpe import iter_pretokens 

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}

        # 合并优先级
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        # 保证special token被处理
        self.special_tokens = special_tokens or []
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.token_to_id[st_bytes] = new_id
                self.id_to_token[new_id] = st_bytes
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从本地文件加载 vocab 和 merges [cite: 307]"""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            # 注意：JSON key 总是字符串，需要转回 int
            vocab = {int(k): v.encode("utf-8") if isinstance(v, str) else bytes(v) 
                     for k, v in vocab_json.items()}
        
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        
        return cls(vocab, merges, special_tokens)
    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码回文本 [cite: 315]"""
        # 拼接字节序列 
        byte_segments = [self.id_to_token[i] for i in ids]
        all_bytes = b"".join(byte_segments)
        
        # 解码为字符串，错误处替换为 U+FFFD 
        return all_bytes.decode("utf-8", errors="replace")
    def _bpe_merge(self, word_bytes: list[bytes]) -> list[bytes]:
        """对单个 pre-token 内部的字节序列应用合并规则 """
        if len(word_bytes) <= 1:
            return word_bytes

        while True:
            # 找出当前序列中所有可合并的 pair 及其在 merges_rank 中的优先级 
            pairs = []
            for i in range(len(word_bytes) - 1):
                pairs.append((word_bytes[i], word_bytes[i+1]))
            
            if not pairs:
                break

            # 选出优先级最高（Rank 最小）的 pair [cite: 282]
            bigram = min(pairs, key=lambda p: self.merges_rank.get(p, float("inf")))

            if bigram not in self.merges_rank:
                break  # 没有更多可以合并的规则了 [cite: 282]

            # 执行合并：将序列中的 (a, b) 替换为 a+b
            new_word_bytes = []
            i = 0
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and word_bytes[i] == bigram[0] and word_bytes[i+1] == bigram[1]:
                    new_word_bytes.append(word_bytes[i] + word_bytes[i+1])
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1
            word_bytes = new_word_bytes
            
        return word_bytes
    def encode(self, text: str) -> list[int]:
        """将文本编码为 ID 序列 [cite: 312]"""
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码，内存占用恒定 [cite: 313, 314]"""
        special_set = set(st.encode("utf-8") for st in self.special_tokens)

        for text_chunk in iterable:
            # 1. 预分词：处理特殊 token 并根据 PAT 切分 [cite: 272, 287]
            for pre_token in iter_pretokens(text_chunk, self.special_tokens):
                pre_token_bytes = pre_token.encode("utf-8")

                # 如果是特殊 token，直接产出 ID [cite: 287]
                if pre_token_bytes in special_set:
                    yield self.token_to_id[pre_token_bytes]
                    continue

                # 2. 将普通片段拆为单字节序列 [cite: 272]
                word_bytes = [bytes([b]) for b in pre_token_bytes]
                
                # 3. 应用 BPE 合并规则 
                merged_segments = self._bpe_merge(word_bytes)
                
                # 4. 映射到 ID
                for token in merged_segments:
                    yield self.token_to_id[token]