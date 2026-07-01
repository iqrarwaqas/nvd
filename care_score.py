"""Official CARE scorer (PIPELINE.md §8).

Sub-scores (Gueck, Roelofs & Faulstich, "CARE to Compare", *Data* 2024):
  * Coverage    - per-anomaly-dataset F_beta (beta=0.5) on the normal-status points.
  * Earliness   - per-anomaly-dataset detection weighted 1 over the first half of the
                  event window, decaying linearly to 0 over the second half.
  * Reliability - event-level F_beta across ALL datasets, where each dataset yields a
                  single anomaly/normal verdict via a criticality counter
                  (Algorithm 1, threshold t_c = 72 = 12h of consecutive 10-min anomalies).
  * Accuracy    - per-normal-dataset tn/(fp+tn) on normal-status points.

**Critical (§8):** omega = (1, 1, 1, 0). Accuracy is a **gate condition**, NOT a
weighted term. Weighting Accuracy is the known bug that inflates trivial models
(e.g. IsolationForest ~0.50 vs published ~0.14). Here:

    WA   = (w_cov*Cov + w_ear*Ear + w_rel*Rel) / (w_cov + w_ear + w_rel)
    CARE = 0        if no anomaly event was detected at all
           0        if mean normal-accuracy < accuracy_gate   (FAILS the gate)
           WA       otherwise
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def f_beta(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return 0.0 if denom <= 0 else float((1 + b2) * tp / denom)


def criticality_max(pred: np.ndarray, is_normal_status: np.ndarray) -> int:
    """Algorithm 1: max of a criticality counter over the prediction frame.
    +1 when status is abnormal, +1 when normal-status but anomaly predicted,
    decrement (floored at 0) when normal-status and no anomaly predicted."""
    crit = cmax = 0
    for pi, normal in zip(pred, is_normal_status):
        if not normal or pi:
            crit += 1
        else:
            crit = max(crit - 1, 0)
        cmax = max(cmax, crit)
    return cmax


def earliness_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    tau = np.linspace(0.0, 1.0, n)
    return np.clip(np.where(tau <= 0.5, 1.0, 2.0 * (1.0 - tau)), 0.0, 1.0)


@dataclass
class DatasetEval:
    dataset_id: str
    is_normal_only: bool
    coverage_fbeta: float | None
    earliness_ws: float | None
    accuracy: float | None
    event_true: int
    event_pred: int


def evaluate_dataset(pred: np.ndarray, status_ids: np.ndarray, *,
                     is_normal_only: bool, event_window, normal_status_ids,
                     dataset_id: str = "", beta: float = 0.5,
                     criticality_threshold: int = 72) -> DatasetEval:
    """Score one dataset's prediction frame (binary per-row anomaly predictions)."""
    pred = np.asarray(pred).astype(bool)
    status_ids = np.asarray(status_ids)
    normal_mask = np.isin(status_ids, list(normal_status_ids))

    cmax = criticality_max(pred, normal_mask)
    event_pred = int(cmax >= criticality_threshold)

    if is_normal_only or event_window is None:
        p = pred[normal_mask]
        tn, fp = int((~p).sum()), int(p.sum())
        acc = tn / (fp + tn) if (fp + tn) > 0 else 1.0
        return DatasetEval(dataset_id, True, None, None, acc, 0, event_pred)

    start, end = event_window
    gt = np.zeros(pred.size, bool)
    s, e = max(0, start), min(pred.size - 1, end)
    if e >= s:
        gt[s:e + 1] = True

    g, p = gt[normal_mask], pred[normal_mask]
    tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
    coverage = f_beta(tp, fp, fn, beta)

    win = np.arange(s, e + 1)
    win_normal = normal_mask[win]
    w = earliness_weights(len(win))[win_normal]
    p_win = pred[win][win_normal].astype(float)
    ws = float((w * p_win).sum() / w.sum()) if w.sum() > 0 else 0.0

    return DatasetEval(dataset_id, False, coverage, ws, None, 1, event_pred)


def care_score(evals: list[DatasetEval], *, beta: float = 0.5,
               weights: dict | None = None, accuracy_gate: float = 0.5) -> dict:
    """Aggregate per-dataset evals into the CARE score (§8, omega=(1,1,1,0))."""
    w = weights or {"coverage": 1.0, "earliness": 1.0, "reliability": 1.0}
    w1, w2, w3 = w["coverage"], w["earliness"], w["reliability"]

    cov = [e.coverage_fbeta for e in evals if e.coverage_fbeta is not None]
    ear = [e.earliness_ws for e in evals if e.earliness_ws is not None]
    acc = [e.accuracy for e in evals if e.accuracy is not None]

    Cov = float(np.mean(cov)) if cov else 0.0
    Ear = float(np.mean(ear)) if ear else 0.0
    Acc = float(np.mean(acc)) if acc else 1.0     # no normal datasets -> no false alarms

    # Reliability: event-level F_beta across ALL datasets.
    tp = sum(1 for e in evals if e.event_true and e.event_pred)
    fp = sum(1 for e in evals if not e.event_true and e.event_pred)
    fn = sum(1 for e in evals if e.event_true and not e.event_pred)
    Rel = f_beta(tp, fp, fn, beta)

    WA = (w1 * Cov + w2 * Ear + w3 * Rel) / (w1 + w2 + w3)

    if not any(e.event_pred for e in evals):
        care = 0.0                                 # nothing detected at all
    elif Acc < accuracy_gate:
        care = 0.0                                 # FAILS the accuracy gate (§8)
    else:
        care = WA

    return {"CARE": float(care), "coverage": Cov, "earliness": Ear,
            "reliability": Rel, "accuracy_gate_value": Acc, "WA": float(WA),
            "gate_passed": bool(Acc >= accuracy_gate),
            "n_anomaly_datasets": len(cov), "n_normal_datasets": len(acc),
            "event_tp": tp, "event_fp": fp, "event_fn": fn}
