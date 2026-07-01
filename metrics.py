"""Continual-learning metrics (PIPELINE.md §9).

All metrics are computed from the stage x task accuracy matrix

    R[i][j] = accuracy on task j after training through stage i   (i >= j)

measured *after* task-aware adaptation (eq 5). ``R`` is a dict-of-dicts keyed by
task id; entries are absent before a task is first seen.
"""
from __future__ import annotations

import numpy as np


def average_incremental_accuracy(R: dict, task_ids: list[int]) -> float:
    """Mean over stages of the average accuracy on all tasks seen up to that stage."""
    per_stage = []
    for i in task_ids:
        seen = [R[i][j] for j in task_ids if j <= i and j in R.get(i, {})]
        if seen:
            per_stage.append(float(np.mean(seen)))
    return float(np.mean(per_stage)) if per_stage else 0.0


def average_forgetting(R: dict, task_ids: list[int]) -> float:
    """Mean over tasks of (best past accuracy - final accuracy).

    forgetting(j) = max_{i < last} R[i][j] - R[last][j], for j != last.
    Positive = the model degraded on task j after learning later tasks.
    """
    last = task_ids[-1]
    vals = []
    for j in task_ids[:-1]:
        past = [R[i][j] for i in task_ids if j <= i < last and j in R.get(i, {})]
        final = R.get(last, {}).get(j)
        if past and final is not None:
            vals.append(max(past) - final)
    return float(np.mean(vals)) if vals else 0.0


def harmonic_mean_base_novel(R: dict, task_ids: list[int]) -> float:
    """Harmonic mean of final accuracy on the base task vs the mean of novel tasks.

    Rewards models that are good on both old and new -- a single strong side does
    not carry the score (unlike the arithmetic mean).
    """
    last = task_ids[-1]
    base = R.get(last, {}).get(task_ids[0])
    novel = [R[last][j] for j in task_ids[1:] if j in R.get(last, {})]
    if base is None or not novel:
        return 0.0
    nov = float(np.mean(novel))
    if base + nov <= 0:
        return 0.0
    return float(2 * base * nov / (base + nov))


def summarize(R: dict, task_ids: list[int]) -> dict:
    return {
        "avg_incremental_accuracy": average_incremental_accuracy(R, task_ids),
        "avg_forgetting": average_forgetting(R, task_ids),
        "harmonic_mean_base_novel": harmonic_mean_base_novel(R, task_ids),
        "final_avg_accuracy": (
            float(np.mean([R[task_ids[-1]][j] for j in task_ids
                           if j in R.get(task_ids[-1], {})]))
            if R.get(task_ids[-1]) else 0.0),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    """mean +/- std over seeds (PIPELINE.md §9 reporting)."""
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))
