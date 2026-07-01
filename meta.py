"""Reptile meta-trainer + task-aware inference (PIPELINE.md §6).

§6.1 Incremental meta-training (eq 3, 4)
    Init Phi <- Phi_{t-1}; remember Phi_base = Phi.
    For each seen task i <= t: reset the model to Phi_base, gather a batch from
    task i's block (drawn from the current task's data D_t and the replay memory
    M), run `r` inner SGD steps on masked cross-entropy (eq 3) -> Phi_i.
    Meta-update (config meta.update_rule):
      * reptile_canonical: Phi <- Phi + eps * mean_i(Phi_i - Phi)
      * paper_eq4_avg    : eta = exp(-beta * t/S);
                           Phi <- eta * (1/t) sum_i Phi_i + (1-eta) * Phi_base

§6.2 Task-aware inference (eq 5, 6)
    Given known task t: clone Phi, adapt `r` steps on M_t (memory filtered to t)
    with out-of-task labels relabelled to `normal` (eq 5), then restrict the
    prediction to C_t and argmax over that block (eq 6, masking in backbone).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from backbone import masked_cross_entropy
from dataloader import WindowSet, iterate_batches


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_tensors(ws: WindowSet, device):
    X = torch.as_tensor(ws.X, dtype=torch.float32, device=device)
    m = torch.as_tensor(ws.mask, dtype=torch.float32, device=device)
    y = torch.as_tensor(ws.y, dtype=torch.long, device=device)
    return X, m, y


def clone_state(model):
    return copy.deepcopy(model.state_dict())


def _flat_params(state):
    return {k: v.clone() for k, v in state.items() if v.dtype.is_floating_point}


# ---------------------------------------------------------------------------
# inner-loop adaptation (eq 3 / eq 5)
# ---------------------------------------------------------------------------
def inner_adapt(model, data: WindowSet, task_columns, *, steps: int, lr: float,
                batch: int, device, rng, relabel_out_of_task: int | None = None):
    """Run exactly ``steps`` SGD steps of masked cross-entropy, each on one freshly
    sampled mini-batch (Reptile's `r` inner steps, eq 3 -- NOT `r` full epochs).

    Sampling a fixed-size batch per step keeps cost independent of the task's size:
    a real task's pool can hold tens of thousands of windows, so iterating the
    whole set every step would be prohibitively slow.

    ``relabel_out_of_task`` (eq 5): any target not in ``task_columns`` is mapped to
    this column (the shared background/normal symbol) before the loss.
    """
    n = len(data)
    if n == 0:
        return
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    keep = set(task_columns)
    model.train()
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        b = data.subset(idx)
        X, m, y = _to_tensors(b, device)
        if relabel_out_of_task is not None:
            y = torch.where(
                torch.tensor([int(t) in keep for t in y.tolist()], device=device),
                y, torch.full_like(y, relabel_out_of_task))
        opt.zero_grad()
        loss = masked_cross_entropy(model(X, m), y, task_columns)
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# §6.1 Reptile incremental meta-training
# ---------------------------------------------------------------------------
class ReptileMetaTrainer:
    def __init__(self, model, cfg, device, seed: int = 0):
        self.model = model
        self.device = device
        m = cfg["meta"]
        self.update_rule = m.get("update_rule", "reptile_canonical")
        self.r = int(m.get("inner_steps_r", 5))
        self.alpha = float(m.get("inner_lr_alpha", 1e-3))
        self.eps = float(m.get("meta_step_eps", 0.1))
        self.beta = float(m.get("beta", 2.0))
        self.epochs = int(m.get("base_task_epochs", 30))
        self.batch = int(m.get("inner_batch", 32))
        self.rng = np.random.default_rng(seed)

    def _task_batch(self, task, memory, label_space, use_pool: bool = True) -> WindowSet:
        """Windows for task i's block. The CURRENT task may use its full pool D_t;
        PAST tasks (use_pool=False) survive only through the bounded memory M --
        this is the continual-learning constraint that makes curation matter."""
        parts = [task.support]
        if use_pool and task.pool is not None and len(task.pool) > 0:
            parts.append(task.pool)
        mem = memory.windows_for_task(task.tid) if memory is not None else None
        if mem is not None and len(mem) > 0:
            parts.append(mem)
        return WindowSet.concat(parts)

    def meta_train_step(self, seen_tasks, memory, label_space, S: int):
        """One incremental meta-update over all tasks seen so far (eq 3, 4)."""
        phi_base = _flat_params(clone_state(self.model))
        t = len(seen_tasks)
        adapted: list[dict] = []

        current = seen_tasks[-1]
        for task in seen_tasks:
            self.model.load_state_dict(_merge(self.model.state_dict(), phi_base))
            cols = label_space.task_columns(task.key)
            # only the current task sees its full pool; past tasks rely on memory M
            data = self._task_batch(task, memory, label_space, use_pool=(task is current))
            for _ in range(self.epochs):
                inner_adapt(self.model, data, cols, steps=self.r, lr=self.alpha,
                            batch=self.batch, device=self.device, rng=self.rng)
            adapted.append(_flat_params(self.model.state_dict()))

        # aggregate the adapted parameter sets into the new Phi
        if self.update_rule == "paper_eq4_avg":
            eta = math.exp(-self.beta * t / max(1, S))
            new = {k: eta * (sum(a[k] for a in adapted) / t) + (1 - eta) * phi_base[k]
                   for k in phi_base}
        else:  # reptile_canonical
            new = {k: phi_base[k] + self.eps * (sum(a[k] - phi_base[k] for a in adapted) / t)
                   for k in phi_base}
        self.model.load_state_dict(_merge(self.model.state_dict(), new))


def _merge(full_state, float_updates):
    """Overlay floating-point updates onto a full state_dict (keeps buffers)."""
    out = {k: v.clone() for k, v in full_state.items()}
    for k, v in float_updates.items():
        out[k] = v.clone()
    return out


# ---------------------------------------------------------------------------
# §6.2 Task-aware inference (eq 5, 6)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict(model, ws: WindowSet, task_columns, device, batch: int = 256):
    """Batched argmax prediction (real query sets have tens of thousands of
    windows, so a single forward pass would exhaust GPU memory)."""
    model.eval()
    preds = []
    for s in range(0, len(ws), batch):
        b = ws.subset(np.arange(s, min(s + batch, len(ws))))
        X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
        m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
        preds.append(model(X, m, task_columns).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)


def task_aware_eval(model, task, memory, label_space, cfg, device, seed=0,
                    adapt: bool = True, max_query: int | None = None) -> dict:
    """Evaluate one task with eq-(5) adaptation on its memory + eq-(6) masking.

    Returns per-window predictions and accuracy on ``task.query``. Adaptation runs
    on a *clone* so the shared Phi is never mutated by evaluation. ``max_query``
    subsamples the query set (used to keep the RL reward cheap)."""
    m = cfg["meta"]
    rng = np.random.default_rng(seed)
    cols = label_space.task_columns(task.key)
    eval_batch = int(cfg.get("eval", {}).get("batch", 256))

    scratch = copy.deepcopy(model).to(device)
    if adapt and memory is not None:
        mem_t = memory.windows_for_task(task.tid)
        adapt_data = WindowSet.concat([task.support, mem_t]) if len(mem_t) else task.support
        inner_adapt(scratch, adapt_data, cols, steps=int(m.get("inner_steps_r", 5)),
                    lr=float(m.get("inner_lr_alpha", 1e-3)),
                    batch=int(m.get("inner_batch", 32)), device=device, rng=rng,
                    relabel_out_of_task=label_space.normal_index)

    query = task.query
    if max_query is not None and len(query) > max_query:
        query = query.subset(rng.choice(len(query), max_query, replace=False))
    preds = _predict(scratch, query, cols, device, batch=eval_batch)
    y = query.y
    acc = float((preds == y).mean()) if len(y) else 0.0
    return {"acc": acc, "preds": preds, "y": y, "task_columns": cols, "model": scratch}
