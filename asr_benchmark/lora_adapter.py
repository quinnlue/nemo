"""Lightweight LoRA adapter compatible with NeMo's ASR adapter framework.

Subclasses ``LinearAdapter`` so it passes the Conformer encoder's
``get_accepted_adapter_types`` check, but replaces the bottleneck MLP with
LoRA's low-rank decomposition:  ``x → A(x) → B(…) * (alpha / rank)``.
"""

import math

import torch
from torch import nn

from nemo.collections.common.parts.adapter_modules import LinearAdapter
from nemo.core.classes.mixins import adapter_mixin_strategies


class LoRAAdapter(LinearAdapter):
    """Drop-in LoRA replacement for NeMo's ``LinearAdapter``.

    Parameters
    ----------
    in_features:
        Input (and output) dimension — must match the encoder block dim.
    rank:
        Rank of the low-rank decomposition (analogous to ``dim`` in
        ``LinearAdapter``).  Small values (4–32) are typical.
    alpha:
        LoRA scaling factor.  The adapter output is multiplied by
        ``alpha / rank``.  Defaults to ``rank`` (i.e.\ scale = 1).
    dropout:
        Dropout applied *before* the down-projection.
    adapter_strategy:
        Adapter merge strategy config — defaults to ``ResidualAddAdapterStrategy``.
    """

    def __init__(
        self,
        in_features: int,
        rank: int = 16,
        alpha: float | None = None,
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
        **kwargs,
    ):
        # Bypass LinearAdapter.__init__ entirely — we build our own layers.
        # Call nn.Module + AdapterModuleUtil init directly.
        nn.Module.__init__(self)

        self.in_features = in_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        # LoRA layers (no bias, no activation)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, in_features, bias=False)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Initialisation: A ~ kaiming-uniform, B = 0  ⟹  adapter starts as identity
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Setup adapter strategy (inherited from AdapterModuleUtil)
        self.setup_adapter_strategy(adapter_strategy)

    # ------------------------------------------------------------------
    # Override LinearAdapter.reset_parameters (not needed for LoRA)
    # ------------------------------------------------------------------
    def reset_parameters(self):
        pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.lora_down(x)
        x = self.lora_up(x)
        return x * self.scaling
