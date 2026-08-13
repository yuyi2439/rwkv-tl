"""Application-layer text wrapper: tokenizer, token generation, and chat.

``RWKV7TextModel`` composes a token-level ``RWKV7Model`` (held as
``self.model``) with a tokenizer. It deliberately does NOT inherit
``RWKV7Model``: the inference layer and the text layer stay decoupled, and
core must not reference this module. This is the class the library exposes to
users; ``rwkv7()`` returns one.

``generate`` is text-level: it takes a string prompt plus decoding
parameters, iterates autoregressively, and returns the generated text
(stopping early when a ``stop`` string appears). ``chat`` renders a message
list through the packaged chat template (jinja2) and runs ``generate``
underneath.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files

import torch
from jinja2 import Environment
from torch import Tensor

from rwkv_tl.core.model import RWKV7Model
from rwkv_tl.core.state import State
from rwkv_tl.core.tokenizer import Tokenizer
from rwkv_tl.sampling import sample_logits

_CHAT_TEMPLATE_NAME = "rwkv_chat_template_v20260805.jinja"
_CHAT_TEMPLATE = None


def _chat_template():
    """Compile the packaged chat template once (lazily)."""
    global _CHAT_TEMPLATE
    if _CHAT_TEMPLATE is None:
        src = (
            files("rwkv_tl")
            .joinpath("asset", _CHAT_TEMPLATE_NAME)
            .read_text(encoding="utf-8")
        )
        _CHAT_TEMPLATE = Environment().from_string(src)
    return _CHAT_TEMPLATE


class RWKV7TextModel:
    """Application-layer wrapper: tokenizer + text API over a token model.

    Holds the token-level model as ``self.model`` (composition, no
    inheritance) plus a decoupled tokenizer. Exposes the full inference
    surface (``decode`` / ``prefill`` / ``forward`` / ``w`` / ``L`` / ``C`` /
    ``N`` / ``H`` delegate to ``self.model``) and adds ``tokenize`` /
    ``detokenize`` / ``generate`` / ``chat``.
    """

    def __init__(self, model: RWKV7Model) -> None:
        self.model = model
        self.tokenizer = Tokenizer()
        self.w = model.w
        self.L = model.L
        self.C = model.C
        self.N = model.N
        self.H = model.H

    # ------------------------------------------------------------------
    # token-level inference surface (delegates to self.model)
    # ------------------------------------------------------------------

    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; returns ``(logits, state)``."""
        return self.model.decode(token, S)

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a 1-D token sequence, updating ``S`` in place."""
        return self.model.prefill(tokens, S)

    def forward(self, tokens: Tensor, S: State) -> tuple[Tensor, State]:
        """Run inference over a token sequence; returns ``(logits, state)``."""
        return self.model.forward(tokens, S)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token ids."""
        return self.tokenizer.encode(text)

    def detokenize(self, ids: Sequence[int]) -> str:
        """Decode token ids back into text."""
        return self.tokenizer.decode(list(ids))

    def generate(
        self,
        input: str,
        *,
        max_new_tokens: int = 256,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop: str | None = None,
    ) -> str:
        """Autoregressively generate text from a text prompt.

        The prompt is tokenized and processed first (prefill + final-token
        decode); only the newly generated text is returned. Generation stops
        early when the decoded output contains ``stop``, and the match is
        truncated away. A fresh zero ``State`` is created internally.

        Args:
            input: Text prompt.
            max_new_tokens: Number of tokens to sample after the prompt.
            temperature: Sampling temperature (None or <= 0 = greedy).
            top_k: Top-k restriction (0 = off).
            top_p: Nucleus threshold (1.0 = off).
            repetition_penalty: Penalty applied to tokens already generated
                (1.0 = off).
            stop: Stop generating as soon as the generated text contains this
                string (the match itself is excluded from the result).

        Returns:
            The generated text (excluding the prompt).
        """
        tokens = self.tokenize(input)
        if not tokens:
            raise ValueError("generate() requires a non-empty prompt")
        ids = torch.tensor(tokens, dtype=torch.long, device=self.w.device)

        S = State(self.L, self.C, self.N, device=self.w.device, dtype=self.w.dtype)
        if ids.numel() == 1:
            logits, S = self.model.decode(ids, S)
        else:
            S = self.model.prefill(ids[:-1], S)
            logits, S = self.model.decode(ids[-1:], S)

        out: list[int] = []
        for _ in range(max_new_tokens):
            token = sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen=torch.tensor(out, dtype=torch.long, device=self.w.device),
            )
            out.append(int(token.item()))
            if stop is not None:
                text = self.detokenize(out)
                i = text.find(stop)
                if i >= 0:
                    return text[:i]
            logits, S = self.model.decode(token, S)
        return self.detokenize(out)

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 128,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop: str | None = None,
        add_generation_prompt: bool = True,
        thinking: bool = False,
        bos_token: str = "",
        tools: list | None = None,
    ) -> str:
        """Chat: render ``messages`` through the chat template and generate.

        The template parameters (``add_generation_prompt``, ``thinking``,
        ``bos_token``, ``tools``) are managed here and passed to the jinja
        renderer. The rendered prompt is tokenized and fed to :meth:`generate`;
        the generated tokens are decoded back into the assistant's reply.

        Args:
            messages: Message dicts with ``role`` ("system" / "user" /
                "assistant" / "tool") and ``content``; assistant messages may
                also carry ``tool_calls`` (see the template).
            max_new_tokens: Maximum length of the assistant reply.
            temperature: Sampling temperature (None or <= 0 = greedy).
            top_k: Top-k restriction (0 = off).
            top_p: Nucleus threshold (1.0 = off).
            repetition_penalty: Penalty applied to already generated tokens.
            stop: Stop string passed to :meth:`generate` (e.g. the next
                speaker marker).
            add_generation_prompt: Append the "Assistant:" generation prompt.
            thinking: Enable the "<think" thinking prefix.
            bos_token: Optional BOS token prepended to the prompt.
            tools: Optional tool schemas rendered into the system block.

        Returns:
            The assistant's reply text (decoded generated tokens).
        """
        prompt = _chat_template().render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            thinking=thinking,
            bos_token=bos_token,
            tools=tools if tools is not None else [],
        )
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )


RWKV7Model.register(RWKV7TextModel)
