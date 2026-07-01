"""CNN-LSTM backbone with a growable, task-partitioned head (PIPELINE.md §5).

Architecture (paper IV-B):
    X [B, C, T]  --1D conv stack-->  H [B, F, T']  --LSTM over time-->  h [B, H]
    h  --Linear head-->  logits [B, |C|]      where |C| = 1 + sum_t U_t

The head is *grown* as tasks arrive: :meth:`grow_head` widens the output layer by
appending zero-initialised columns for the new task's block while preserving the
existing columns bit-for-bit (freeze test in selftest.py). At inference a
``task_id`` restricts the logits to that task's columns via eq-(6) masking:
non-visible columns are set to ``-inf`` so argmax can never leave the task block.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


class CNNLSTMBackbone(nn.Module):
    def __init__(self, n_channels: int, conv_channels=(64, 64, 128), kernel: int = 5,
                 lstm_hidden: int = 128, lstm_layers: int = 1, dropout: float = 0.2,
                 n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.lstm_hidden = lstm_hidden

        # --- 1D conv stack over time (padding='same' keeps T so masking is simple) ---
        convs, in_ch = [], n_channels
        for out_ch in conv_channels:
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)

        # --- LSTM over conv features (batch_first: [B, T, F]) ---
        self.lstm = nn.LSTM(in_ch, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=(dropout if lstm_layers > 1 else 0.0))

        # --- growable linear head; starts with the shared `normal` column only ---
        self.head = nn.Linear(lstm_hidden, max(1, n_classes))

    # -- feature extractor (shared trunk) -------------------------------------
    def embed(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the sequence embedding h [B, lstm_hidden].

        ``mask`` [B, T] (1=valid) zeroes masked timesteps before the LSTM and
        selects the last *valid* step as the sequence summary, so masked padding
        never contributes to the representation.
        """
        h = self.conv(x)                       # [B, F, T]
        h = h.transpose(1, 2)                  # [B, T, F]
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        out, _ = self.lstm(h)                  # [B, T, H]
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).long() - 1   # last valid index
            idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
            return out.gather(1, idx).squeeze(1)
        return out[:, -1, :]

    # -- classification with optional eq-(6) task masking ---------------------
    def forward(self, x, mask=None, task_columns: list[int] | None = None):
        """Logits [B, |C|]. If ``task_columns`` is given, mask every other column
        to ``-inf`` (eq-6): the model may only predict {normal} u block(task)."""
        logits = self.head(self.embed(x, mask))
        if task_columns is not None:
            keep = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
            keep[torch.as_tensor(task_columns, device=logits.device)] = True
            logits = logits.masked_fill(~keep.unsqueeze(0), NEG_INF)
        return logits

    @property
    def n_classes(self) -> int:
        return self.head.out_features

    @torch.no_grad()
    def grow_head(self, new_total: int):
        """Widen the head to ``new_total`` columns, preserving existing weights.

        New columns are zero-initialised so a fixed input's existing logits are
        unchanged immediately after growth (§5 DoD freeze test).
        """
        old = self.head
        if new_total <= old.out_features:
            return
        new = nn.Linear(old.in_features, new_total).to(old.weight.device)
        nn.init.zeros_(new.weight)
        nn.init.zeros_(new.bias)
        new.weight.data[:old.out_features] = old.weight.data
        new.bias.data[:old.out_features] = old.bias.data
        self.head = new


def masked_cross_entropy(logits, targets, task_columns=None):
    """Cross-entropy that respects eq-(6) masking (targets are global indices)."""
    if task_columns is not None:
        keep = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
        keep[torch.as_tensor(task_columns, device=logits.device)] = True
        logits = logits.masked_fill(~keep.unsqueeze(0), NEG_INF)
    return F.cross_entropy(logits, targets)
