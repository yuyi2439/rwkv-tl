from __future__ import annotations

# This file is adapted from ChatRWKV tokenizer implementation:
# https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_tokenizer.py
# Original implementation by BlinkDL/ChatRWKV.


class TRIE:
    __slots__ = ("to", "token")
    to: list[TRIE | None]
    token: int

    def __init__(self) -> None:
        self.to = [None for _ in range(256)]
        self.token = 0

    def add(self, key: bytes, val: int) -> None:
        u = self
        for ch in key:
            v = u.to[ch]
            if v is None:
                v = TRIE()
                u.to[ch] = v
            u = v
        u.token = val + 1


class Tokenizer:
    def __init__(self, vocab_path: str) -> None:
        idx2token: dict[int, bytes] = {}
        sorted: list[bytes] = []  # must be already sorted
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            idx2token[idx] = x

        self.token2idx: dict[bytes, int] = {}
        for k, v in idx2token.items():
            self.token2idx[v] = int(k)
        self.idx2token: list[bytes] = [b"" for _ in range(max(idx2token) + 1)]
        for idx, token in idx2token.items():
            self.idx2token[idx] = token

        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=i)
        for ch in range(256):
            assert self.root.to[ch] is not None

    def encodeBytes(self, src: bytes) -> list[int]:
        tokens: list[int] = []
        append = tokens.append
        root_to = self.root.to
        idx = 0
        src_len = len(src)
        while idx < src_len:
            u = root_to[src[idx]]
            assert u is not None
            j = idx + 1
            token = u.token
            end = j
            to = u.to
            while j < src_len:
                u = to[src[j]]
                if u is None:
                    break
                j += 1
                tok = u.token
                if tok:
                    token = tok
                    end = j
                to = u.to
            append(token - 1)
            idx = end
        return tokens

    def decodeBytes(self, tokens: list[int] | tuple[int, ...]) -> bytes:
        return b"".join(map(self.idx2token.__getitem__, tokens))

    def encode(self, src: str) -> list[int]:
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
        return self.decodeBytes(tokens).decode("utf-8")
