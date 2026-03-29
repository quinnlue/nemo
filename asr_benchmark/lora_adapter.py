"""Lightweight LoRA utilities for NeMo ASR models.

Provides ``LoRALinear``, a drop-in wrapper around ``nn.Linear`` that adds a
trainable low-rank residual (A · B · scaling) while keeping the original
weight frozen, and ``inject_lora``, which walks a model and replaces target
linear layers in-place.
"""

import math
import re
from typing import Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that adds a low-rank adapter.

    The original weight is frozen; only ``lora_down`` (A) and ``lora_up`` (B)
    are trainable.  Output = ``original(x) + (x @ A^T @ B^T) * (alpha / rank)``.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features

        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Init: A ~ kaiming-uniform, B = 0  ⟹  adapter starts as identity
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        return base + lora


# ── Default target patterns ─────────────────────────────────────────────────
# Each pattern is matched against the *full dotted parameter name* relative to
# the module passed to ``inject_lora``.
ATTN_TARGETS = [
    r"self_attn\.linear_q",
    r"self_attn\.linear_k",
    r"self_attn\.linear_v",
    r"self_attn\.linear_out",
]

FFN_TARGETS = [
    r"feed_forward1\.linear1",
    r"feed_forward1\.linear2",
    r"feed_forward2\.linear1",
    r"feed_forward2\.linear2",
]

DEFAULT_TARGETS = ATTN_TARGETS + FFN_TARGETS


def inject_lora(
    model: nn.Module,
    targets: Sequence[str] = DEFAULT_TARGETS,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear`` wrappers.

    Parameters
    ----------
    model:
        The model (or sub-module) to modify **in-place**.
    targets:
        Regex patterns matched against the full dotted name of each module.
        Only ``nn.Linear`` modules whose name matches at least one pattern
        are wrapped.
    rank:
        LoRA rank for every injected adapter.
    alpha:
        LoRA scaling factor (defaults to *rank*).
    dropout:
        Dropout applied to LoRA input.

    Returns
    -------
    list[str]
        Names of the modules that were wrapped.
    """
    replaced: list[str] = []
    compiled = [re.compile(p) for p in targets]

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(p.search(name) for p in compiled):
            continue

        # Walk to the parent so we can swap the attribute
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            attr = parts[0]
            parent = model

        lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_linear)
        replaced.append(name)

    return replaced
