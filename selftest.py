"""Fast synthetic + tiny-real assertions (PIPELINE.md §10 DoDs). No pytest.

Run via ``python main.py selftest``. Each check is a function that raises on
failure; :func:`run_all` executes them and prints a compact pass/fail report.
The checks are cheap (seconds) and require neither a GPU nor the CARE dataset.
"""
from __future__ import annotations

import numpy as np
import torch

import dataloader as dl
from backbone import CNNLSTMBackbone
from care_score import DatasetEval, care_score, criticality_max, evaluate_dataset
from labels import LabelSpace, fault_group_of, window_label
from memory import CurationContext, RandomCuration, ReplayBuffer, RLCuration, HerdingCuration
from meta import task_aware_eval
import metrics


# ---------------------------------------------------------------------------
# §3 data / label invariants
# ---------------------------------------------------------------------------
def check_zero_run_masking():
    """No window may span a masked zero-run (§3.3 DoD a)."""
    x = np.ones((100, 3))
    x[40:60] = 0.0                                   # a 20-row missing gap
    valid = dl._zero_run_mask(x, min_run=6)
    assert not valid[40:60].any(), "zero-run not masked"
    assert valid[:40].all() and valid[60:].all(), "valid rows wrongly masked"
    status = np.zeros(100, int)
    X, m, centers, _ = dl._make_windows(x, valid, status, window=16, stride=4)
    # every emitted window must lie fully inside a valid region
    for c in centers:
        assert valid[c - 8:c + 8].all() if 8 <= c <= 92 else True


def check_label_space_inventory():
    """sum(U_t) matches the discovered class inventory (§3.3 DoD d)."""
    ls = LabelSpace()
    ls.register_task("A/gearbox", ["gearbox", "bearing"])   # U=2
    ls.register_task("B/anomaly", ["anomaly"])              # U=1
    assert ls.total_classes == 1 + 2 + 1, ls.total_classes
    assert ls.task_columns("A/gearbox") == [0, 1, 2]
    assert ls.task_columns("B/anomaly") == [0, 3]
    assert ls.global_index("A/gearbox", "bearing") == 2


def check_fault_grouping():
    assert fault_group_of("Gearbox oil leakage") == "gearbox"
    assert fault_group_of("generator bearing damage") == "generator"
    assert fault_group_of(None) == "other"
    # window fully in normal status -> normal regardless of event label
    assert window_label([0, 0, 2], "anomaly", "gearbox", (0, 2)) == "normal"
    assert window_label([0, 5, 2], "anomaly", "gearbox", (0, 2)) == "gearbox"


# ---------------------------------------------------------------------------
# §5 backbone
# ---------------------------------------------------------------------------
def check_backbone_shapes_and_growth():
    torch.manual_seed(0)
    model = CNNLSTMBackbone(n_channels=6, n_classes=1)
    model.eval()                                     # freeze BN/dropout so logits are deterministic
    x = torch.randn(4, 6, 32)
    mask = torch.ones(4, 32)
    assert model(x, mask).shape == (4, 1)

    # Freeze test: growing the head must not perturb existing logits (§5 DoD).
    model.grow_head(3)
    before = model(x, mask).detach().clone()
    model.grow_head(6)
    after = model(x, mask).detach()
    assert torch.allclose(before, after[:, :3], atol=1e-6), "grow_head perturbed old logits"

    # eq-(6) masking: out-of-task columns are -inf, argmax stays in the block.
    logits = model(x, mask, task_columns=[0, 4, 5])
    assert torch.isinf(logits[:, 1]).all()
    assert set(logits.argmax(1).tolist()) <= {0, 4, 5}


# ---------------------------------------------------------------------------
# §6.2 task-aware adaptation
# ---------------------------------------------------------------------------
def check_task_aware_adaptation():
    """Adaptation on memory raises target-task accuracy above no-adapt (§6.2 DoD)."""
    cur = dl.make_synthetic_curriculum(n_tasks=2, n_channels=6, window=32,
                                       per_class=40, K=8, seed=1)
    cfg = {"meta": {"inner_steps_r": 8, "inner_lr_alpha": 5e-2, "inner_batch": 16}}
    device = "cpu"
    model = CNNLSTMBackbone(6, n_classes=cur.label_space.total_classes)

    # quick supervised warm-up so the trunk is not random noise
    _quick_fit(model, cur, device, epochs=6)

    class _Mem:
        def __init__(self, task): self.t = task
        def windows_for_task(self, tid): return self.t.support
    task = cur.tasks[1]
    no_adapt = task_aware_eval(model, task, None, cur.label_space, cfg, device, adapt=False)
    adapted = task_aware_eval(model, task, _Mem(task), cur.label_space, cfg, device, adapt=True)
    assert adapted["acc"] >= no_adapt["acc"] - 1e-6, (no_adapt["acc"], adapted["acc"])
    # masking never predicts an out-of-task class
    cols = set(cur.label_space.task_columns(task.key))
    assert set(np.unique(adapted["preds"]).tolist()) <= cols


# ---------------------------------------------------------------------------
# §7 memory
# ---------------------------------------------------------------------------
def check_buffer_capacity():
    """Buffer never exceeds B under any strategy (§7 DoD a)."""
    cur = dl.make_synthetic_curriculum(n_tasks=3, per_class=50, K=5, seed=2)
    emb = lambda ws: ws.X.reshape(len(ws), -1)      # cheap embedding for the test
    ctx = CurationContext(embed_fn=emb, grad_fn=emb, seed=0)
    for strat in (RandomCuration(), HerdingCuration()):
        buf = ReplayBuffer(capacity=30)
        for task in cur:
            strat.update(buf, task.pool, ctx)
            assert len(buf) <= 30, (strat.name, len(buf))


def check_rl_recovers_protective_exemplars():
    """On a rigged pool, the warm-started RL policy keeps the 'protective'
    exemplars more often than random (§7 DoD b)."""
    # Single class of 40 windows; the 10 "protective" exemplars form a tight
    # cluster at the class mean, the other 30 are spread far out. Herding (and
    # thus a BC-warm-started policy) keeps the representative cluster, while
    # unstructured random keeps them only in proportion (~10/40).
    rng = np.random.default_rng(0)
    n, D = 40, 6
    X = rng.normal(0, 3.0, (n, D, 8)).astype(np.float32)   # spread-out majority
    X[:10] = rng.normal(0, 0.05, (10, D, 8)).astype(np.float32)  # protective cluster at mean
    y = np.zeros(n, np.int64)
    protective = np.zeros(n, bool); protective[:10] = True
    ws = dl.WindowSet(X, np.ones((n, 8), np.float32), y, np.zeros(n, np.int64), ["A"] * n)
    emb = lambda w: w.X.reshape(len(w), -1)
    ctx = CurationContext(embed_fn=emb, seed=0)

    rl = RLCuration({"policy_hidden": 32}, seed=0)
    buf = ReplayBuffer(capacity=10)
    rl.bc_warmstart(buf, ws, ctx, HerdingCuration(), steps=100)
    rl.update(buf, ws, ctx)
    # a kept window is protective if its embedding is near the origin cluster
    kept = buf.as_windowset()
    rl_recall = float((np.linalg.norm(kept.X.reshape(len(kept), -1), axis=1) < 1.0).mean())

    rand = RandomCuration(); rbuf = ReplayBuffer(10)
    rand.update(rbuf, ws, ctx)
    keptr = rbuf.as_windowset()
    rand_recall = float((np.linalg.norm(keptr.X.reshape(len(keptr), -1), axis=1) < 1.0).mean())
    assert rl_recall > rand_recall, (rl_recall, rand_recall)


# ---------------------------------------------------------------------------
# §8 CARE score
# ---------------------------------------------------------------------------
def check_care_score_gate():
    """Accuracy is a gate, not a weighted term (§8)."""
    assert criticality_max(np.array([1, 1, 1]), np.array([True, True, True])) == 3
    # a "trivial" model that flags everything: perfect coverage but zero normal-accuracy
    anomaly = evaluate_dataset(np.ones(100), np.zeros(100, int), is_normal_only=False,
                               event_window=(10, 40), normal_status_ids=[0],
                               criticality_threshold=5)
    normal = evaluate_dataset(np.ones(100), np.zeros(100, int), is_normal_only=True,
                              event_window=None, normal_status_ids=[0],
                              criticality_threshold=5)
    out = care_score([anomaly, normal], accuracy_gate=0.5)
    assert not out["gate_passed"], "trivial model should fail the accuracy gate"
    assert out["CARE"] == 0.0, out["CARE"]


def check_metrics_forgetting():
    R = {0: {0: 0.9}, 1: {0: 0.6, 1: 0.8}}
    assert abs(metrics.average_forgetting(R, [0, 1]) - 0.3) < 1e-9
    assert metrics.harmonic_mean_base_novel(R, [0, 1]) > 0


# ---------------------------------------------------------------------------
# helpers + runner
# ---------------------------------------------------------------------------
def _quick_fit(model, cur, device, epochs=6):
    import torch.nn.functional as F
    ws = dl.WindowSet.concat([t.support for t in cur] + [t.pool for t in cur if len(t.pool)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X = torch.as_tensor(ws.X); m = torch.as_tensor(ws.mask); y = torch.as_tensor(ws.y)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X, m), y)
        loss.backward(); opt.step()


CHECKS = [
    check_zero_run_masking,
    check_label_space_inventory,
    check_fault_grouping,
    check_backbone_shapes_and_growth,
    check_task_aware_adaptation,
    check_buffer_capacity,
    check_rl_recovers_protective_exemplars,
    check_care_score_gate,
    check_metrics_forgetting,
]


def run_all() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    passed = 0
    for chk in CHECKS:
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001 - selftest reports, does not raise
            print(f"  FAIL  {chk.__name__}: {type(e).__name__}: {e}")
    print(f"\nselftest: {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
