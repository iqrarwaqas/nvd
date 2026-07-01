"""CARE-IFD entry point -- run everything from here.

RL-guided incremental fault diagnosis for wind turbines on the CARE benchmark.
This single file orchestrates the modular components (dataloader, labels, backbone,
meta, memory, care_score, metrics) into four runnable commands:

    python main.py selftest        # cheap invariants (§10 DoDs); no data/GPU needed
    python main.py baselines       # baseline ladder on a small subset -> scoreboard
    python main.py train           # full incremental run of the proposed method
    python main.py eval            # score a trained run over seeds

Design notes
------------
* The heavy lifting lives in the modules; this file only wires them together and
  is where the continual-learning *control flow* is spelled out and commented.
* If the CARE dataset is not present at ``data.root``, the data-dependent commands
  fall back to a synthetic curriculum so the pipeline still runs end-to-end. The
  fallback is announced loudly -- synthetic numbers are NOT benchmark numbers.
* Nothing here trains automatically on import; training only happens when you
  invoke ``train`` / ``baselines`` explicitly.

Method presets (the Table-I ladder, PIPELINE.md §9):
    fine_tuning        - lower bound: plain SGD, no memory.
    ewc                - SGD + Elastic Weight Consolidation penalty.
    icarl              - herding replay + distillation (nearest-mean flavour).
    reptile_random     - Reptile + task-aware backbone + random memory.
    reptile_herding    - Reptile + task-aware backbone + herding memory.
    reptile_reservoir  - Reptile + task-aware backbone + reservoir memory.
    reptile_gss        - Reptile + task-aware backbone + GSS memory (Table II).
    proposed           - Reptile + task-aware backbone + RL-curated memory (ours).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn.functional as F

# --- modular building blocks -------------------------------------------------
import dataloader as dl
import metrics
from backbone import CNNLSTMBackbone, masked_cross_entropy
from care_score import DatasetEval, care_score, evaluate_dataset
from memory import CurationContext, ReplayBuffer, make_strategy
from meta import ReptileMetaTrainer, inner_adapt, task_aware_eval

HERE = os.path.dirname(os.path.abspath(__file__))

# Preset -> (learning mode, memory strategy). None strategy = no replay buffer.
METHOD_PRESETS = {
    "fine_tuning":       ("finetune", None),
    "ewc":               ("ewc",      None),
    "icarl":             ("distill",  "herding"),
    "reptile_random":    ("meta",     "random"),
    "reptile_herding":   ("meta",     "herding"),
    "reptile_reservoir": ("meta",     "reservoir"),
    "reptile_gss":       ("meta",     "gss"),
    "proposed":          ("meta",     "rl"),
}


# ===========================================================================
# Config + environment
# ===========================================================================
def load_config(path: str | None) -> dict:
    """Load config.yaml (falls back to the bundled default next to main.py)."""
    import yaml
    path = path or os.path.join(HERE, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve dataset root relative to the config file, per PIPELINE.md's fixed path
    root = cfg["data"]["root"]
    if not os.path.isabs(root):
        cfg["data"]["root"] = os.path.normpath(os.path.join(HERE, root))
    return cfg


def resolve_device(cfg: dict) -> str:
    want = cfg.get("train", {}).get("device", "auto")
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# Curriculum construction (real CARE, or synthetic fallback)
# ===========================================================================
def build_curriculum(cfg: dict, seed: int, small: bool = False) -> tuple[dl.Curriculum, bool]:
    """Return (curriculum, is_synthetic).

    Tries the real CARE loader first; if the dataset is missing (or pandas is
    unavailable) it falls back to a synthetic curriculum and flags it clearly.
    """
    root = cfg["data"]["root"]
    if os.path.isdir(root):
        try:
            if small:
                # cap datasets/farm so a real-data smoke test loads in seconds,
                # not minutes (Farm C alone is ~17 GB across 58 CSVs).
                cfg = copy.deepcopy(cfg)
                cap = cfg["data"].get("max_datasets_per_farm")
                cfg["data"]["max_datasets_per_farm"] = min(cap or 4, 4)
            cur = dl.build_care_curriculum(cfg, seed=seed)
            return cur, False
        except Exception as e:  # noqa: BLE001
            print(f"[warn] real CARE loader failed ({type(e).__name__}: {e});"
                  f" falling back to synthetic curriculum.")
    else:
        print(f"[warn] CARE dataset not found at {root!r}; using SYNTHETIC curriculum."
              f" Numbers below are for pipeline validation only, NOT benchmark results.")
    n_tasks = 6 if not small else 4
    cur = dl.make_synthetic_curriculum(n_tasks=n_tasks, n_channels=6, window=32,
                                       per_class=(40 if small else 80),
                                       K=int(cfg["task"]["shots_K"]), seed=seed)
    return cur, True


# ===========================================================================
# Backbone helpers: embeddings, gradients, and the memory context
# ===========================================================================
def make_embed_fn(model: CNNLSTMBackbone, device: str):
    """Closure: WindowSet -> [N, D] backbone embedding (used by herding/GSS/RL state)."""
    @torch.no_grad()
    def embed(ws: dl.WindowSet, batch: int = 256) -> np.ndarray:
        if len(ws) == 0:
            return np.zeros((0, model.lstm_hidden), np.float32)
        model.eval()
        out = []
        for s in range(0, len(ws), batch):     # batched: pools can be tens of thousands
            b = ws.subset(np.arange(s, min(s + batch, len(ws))))
            X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
            m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
            out.append(model.embed(X, m).cpu().numpy())
        return np.concatenate(out, axis=0)
    return embed


def make_grad_fn(model: CNNLSTMBackbone, label_space, device: str):
    """Closure: WindowSet -> per-sample last-layer gradient sketch (for GSS).

    Uses the gradient of the masked CE loss w.r.t. the head bias as a cheap,
    fixed-width signature of each sample's learning direction.
    """
    def grad(ws: dl.WindowSet) -> np.ndarray:
        if len(ws) == 0:
            return np.zeros((0, model.n_classes), np.float32)
        was_training = model.training
        model.train()          # cuDNN requires train mode for RNN backward
        out = np.zeros((len(ws), model.n_classes), np.float32)
        for i in range(len(ws)):
            model.zero_grad(set_to_none=True)
            X = torch.as_tensor(ws.X[i:i + 1], dtype=torch.float32, device=device)
            m = torch.as_tensor(ws.mask[i:i + 1], dtype=torch.float32, device=device)
            y = torch.as_tensor(ws.y[i:i + 1], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(X, m), y)
            g = torch.autograd.grad(loss, model.head.bias, retain_graph=False)[0]
            out[i] = g.detach().cpu().numpy()
        model.train(was_training)
        return out
    return grad


# ===========================================================================
# Per-task learning modes
# ===========================================================================
def _task_training_windows(task, buffer):
    """D_t (support + pool) union the replay memory M."""
    parts = [task.support]
    if len(task.pool):
        parts.append(task.pool)
    if buffer is not None and len(buffer):
        parts.append(buffer.as_windowset())
    return dl.WindowSet.concat(parts)


def _sampled_batches(data, batch, rng, max_batches):
    """Yield up to ``max_batches`` freshly sampled mini-batches from ``data``.

    Bounds cost independent of task size: a real task can hold tens of thousands
    of windows, so a full epoch per step would be far too slow.
    """
    n = len(data)
    for _ in range(max_batches):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        yield data.subset(idx)


def learn_finetune(model, task, label_space, buffer, cfg, device, rng, ewc=None):
    """Plain supervised SGD on the task (lower bound / EWC / iCaRL trunk)."""
    cols = label_space.task_columns(task.key)
    data = _task_training_windows(task, buffer)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["meta"]["inner_lr_alpha"]) * 10)
    spe = int(cfg["meta"].get("steps_per_epoch", 50))
    model.train()
    for _ in range(int(cfg["meta"]["base_task_epochs"])):
        for b in _sampled_batches(data, int(cfg["meta"]["inner_batch"]), rng, spe):
            X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
            m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
            y = torch.as_tensor(b.y, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = masked_cross_entropy(model(X, m), y, cols)
            if ewc is not None:
                loss = loss + ewc.penalty(model)      # Elastic Weight Consolidation
            loss.backward()
            opt.step()


def learn_distill(model, task, label_space, buffer, cfg, device, rng, teacher):
    """iCaRL-flavoured learning: CE on new data + KL distillation from the old
    model on replayed exemplars (keeps old-class responses stable)."""
    cols = label_space.task_columns(task.key)
    data = _task_training_windows(task, buffer)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["meta"]["inner_lr_alpha"]) * 10)
    spe = int(cfg["meta"].get("steps_per_epoch", 50))
    lam = 0.5
    model.train()
    for _ in range(int(cfg["meta"]["base_task_epochs"])):
        for b in _sampled_batches(data, int(cfg["meta"]["inner_batch"]), rng, spe):
            X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
            m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
            y = torch.as_tensor(b.y, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(X, m)
            loss = F.cross_entropy(logits, y)
            if teacher is not None:                   # distill on the shared old columns
                with torch.no_grad():
                    t_logits = teacher(X, m)
                k = min(t_logits.size(1), logits.size(1))
                loss = loss + lam * F.kl_div(
                    F.log_softmax(logits[:, :k], 1),
                    F.softmax(t_logits[:, :k], 1), reduction="batchmean")
            loss.backward()
            opt.step()


class EWC:
    """Elastic Weight Consolidation: quadratic penalty anchoring params to their
    post-task values, weighted by a diagonal Fisher estimate (ref [13])."""

    def __init__(self, lam: float = 5.0):
        self.lam = lam
        self.anchors: list[tuple[dict, dict]] = []   # (params, fisher) per consolidated task

    def penalty(self, model) -> torch.Tensor:
        loss = torch.zeros((), device=next(model.parameters()).device)
        cur = dict(model.named_parameters())
        for params, fisher in self.anchors:
            for n, p in cur.items():
                if n in fisher and p.shape == params[n].shape:
                    loss = loss + (fisher[n] * (p - params[n]) ** 2).sum()
        return self.lam * loss

    @torch.no_grad()
    def _snapshot_params(self, model):
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    def consolidate(self, model, task, label_space, device, rng):
        """Estimate the diagonal Fisher on this task and store an anchor."""
        cols = label_space.task_columns(task.key)
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        data = task.support
        model.train()          # cuDNN requires train mode for RNN backward
        count = 0
        for b in dl.iterate_batches(data, 8, rng):
            X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
            m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
            y = torch.as_tensor(b.y, dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            masked_cross_entropy(model(X, m), y, cols).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        if count:
            for n in fisher:
                fisher[n] /= count
        self.anchors.append((self._snapshot_params(model), fisher))


# ===========================================================================
# The incremental continual-learning loop (the heart of the pipeline)
# ===========================================================================
def run_incremental(cfg: dict, seed: int, method: str, small: bool = False,
                    return_state: bool = False):
    """Run one A->B->C incremental pass for ``method`` and return its scoreboard.

    If ``return_state`` is set, returns ``(scoreboard, state)`` where ``state``
    holds the trained ``model``, replay ``buffer``, ``label_space`` and
    ``curriculum`` -- used by ``train`` to save a checkpoint for later ``eval``.

    Steps per task t (PIPELINE.md §5-§7):
      1. Grow the head to reveal task t's block (old columns preserved).
      2. Learn task t (meta / finetune / ewc / distill), replaying memory M.
      3. Curate M with task t's candidates under the chosen strategy (|M| <= B).
      4. Task-aware-evaluate every seen task (eq 5-6) -> accuracy matrix R.
    """
    set_seed(seed)
    device = resolve_device(cfg)
    learn_mode, strat_name = METHOD_PRESETS[method]

    cur, synthetic = build_curriculum(cfg, seed, small=small)
    ls = cur.label_space
    rng = np.random.default_rng(seed)

    # Backbone starts with only the shared `normal` column; it grows per task.
    model = CNNLSTMBackbone(
        n_channels=cur.n_channels,
        conv_channels=tuple(cfg["backbone"]["conv_channels"]),
        kernel=int(cfg["backbone"]["kernel"]),
        lstm_hidden=int(cfg["backbone"]["lstm_hidden"]),
        lstm_layers=int(cfg["backbone"]["lstm_layers"]),
        dropout=float(cfg["backbone"]["dropout"]),
        n_classes=1,
    ).to(device)

    # Memory + curation strategy (only for replay methods).
    buffer = None
    strategy = None
    if strat_name is not None:
        buffer = ReplayBuffer(int(cfg["memory"]["budget_B"]))
        strat_cfg = copy.deepcopy(cfg)
        strat_cfg["memory"]["strategy"] = strat_name
        strategy = make_strategy(strat_cfg, seed=seed)

    trainer = ReptileMetaTrainer(model, cfg, device, seed) if learn_mode == "meta" else None
    ewc = EWC(lam=float(cfg.get("cl", {}).get("lambda_ewc", 5.0))) if learn_mode == "ewc" else None

    R: dict[int, dict[int, float]] = {}
    seen: list = []
    S = len(cur)

    for task in cur:
        seen.append(task)

        # 1) grow head to include this task's columns -----------------------
        block_start, block_w = ls.block(task.key)
        model.grow_head(block_start + block_w)

        # 2) learn the task --------------------------------------------------
        teacher = copy.deepcopy(model) if learn_mode == "distill" else None
        if learn_mode == "meta":
            trainer.meta_train_step(seen, buffer, ls, S)
            # also fit the new block on its support so fresh columns are trained
            inner_adapt(model, _task_training_windows(task, buffer),
                        ls.task_columns(task.key),
                        steps=int(cfg["meta"]["inner_steps_r"]),
                        lr=float(cfg["meta"]["inner_lr_alpha"]),
                        batch=int(cfg["meta"]["inner_batch"]), device=device, rng=rng)
        elif learn_mode == "ewc":
            learn_finetune(model, task, ls, buffer, cfg, device, rng, ewc=ewc)
            ewc.consolidate(model, task, ls, device, rng)
        elif learn_mode == "distill":
            learn_distill(model, task, ls, buffer, cfg, device, rng, teacher)
        else:  # finetune
            learn_finetune(model, task, ls, buffer, cfg, device, rng)

        # 3) curate the replay memory ---------------------------------------
        if strategy is not None:
            ctx = CurationContext(
                embed_fn=make_embed_fn(model, device),
                grad_fn=make_grad_fn(model, ls, device),
                reward_fn=make_reward_fn(model, seen, buffer, ls, cfg, device, seed),
                seed=seed + task.tid)
            # Candidates = the new task's K-shot support + a capped sample of its
            # labelled pool. Capping keeps curation (esp. the RL episodes) tractable
            # -- a real task's pool can hold tens of thousands of windows.
            cap = int(cfg["memory"].get("candidate_pool_cap", 512))
            pool = task.pool
            if len(pool) > cap:
                pool = pool.subset(rng.choice(len(pool), cap, replace=False))
            candidates = dl.WindowSet.concat([task.support, pool]) \
                if len(pool) else task.support
            # For the RL policy: BC warm-start on the first task, then REINFORCE.
            if strat_name == "rl":
                run_rl_curation(strategy, buffer, candidates, ctx, cfg, first=(task.tid == 0))
            else:
                strategy.update(buffer, candidates, ctx)

        # 4) task-aware evaluation on all tasks seen so far -----------------
        R[task.tid] = {}
        for prev in seen:
            res = task_aware_eval(model, prev, buffer, ls, cfg, device,
                                  seed=seed, adapt=(buffer is not None))
            R[task.tid][prev.tid] = res["acc"]

    task_ids = [t.tid for t in cur]
    scoreboard = metrics.summarize(R, task_ids)
    scoreboard["method"] = method
    scoreboard["seed"] = seed
    scoreboard["synthetic"] = synthetic
    scoreboard["n_tasks"] = S
    scoreboard["total_classes"] = ls.total_classes
    scoreboard["device"] = device

    # Full learning curve + per-task breakdown so another session can reconstruct
    # exactly what happened at every stage (not just the headline averages).
    scoreboard["tasks"] = [
        {"tid": t.tid, "key": t.key, "farm": t.farm, "U": t.U,
         "local_classes": t.local_classes, "n_support": len(t.support),
         "n_query": len(t.query), "n_pool": len(t.pool)}
        for t in cur]
    scoreboard["accuracy_matrix"] = {str(i): {str(j): v for j, v in row.items()}
                                     for i, row in R.items()}
    scoreboard["per_task"] = _per_task_breakdown(R, cur)

    # native CARE score in binary detection mode (real data only)
    if cfg["eval"].get("report_care_score", True) and not synthetic:
        scoreboard["care"] = compute_care_score(model, cur, cfg, device)

    if return_state:
        return scoreboard, {"model": model, "buffer": buffer,
                            "label_space": ls, "curriculum": cur,
                            "synthetic": synthetic}
    return scoreboard


def _per_task_breakdown(R: dict, cur) -> list[dict]:
    """Per-task first/final accuracy + forgetting, for the human-readable report."""
    task_ids = [t.tid for t in cur]
    last = task_ids[-1]
    key = {t.tid: t.key for t in cur}
    out = []
    for j in task_ids:
        first = R.get(j, {}).get(j)                       # accuracy right after learning j
        final = R.get(last, {}).get(j)                    # accuracy at end of sequence
        past = [R[i][j] for i in task_ids if j <= i and j in R.get(i, {})]
        forget = (max(past) - final) if (past and final is not None and j != last) else None
        out.append({"tid": j, "key": key[j],
                    "acc_after_learning": first, "acc_final": final,
                    "forgetting": forget})
    return out


def make_reward_fn(model, seen, buffer, label_space, cfg, device, seed):
    """Reward closure for the RL policy (eq 7), measured AFTER task-aware
    adaptation: r = alpha*Acc_old + (1-alpha)*Acc_new (occupancy term added in
    memory.py). Old = mean accuracy over previously-seen tasks; new = current."""
    alpha = float(cfg["memory"]["rl"].get("reward_alpha", 0.5))
    # cap the query used per reward: the RL loop calls this many times, so a full
    # tens-of-thousands query eval per episode would be prohibitively slow.
    mq = int(cfg["memory"]["rl"].get("reward_max_query", 512))

    def reward(candidate_ws: dl.WindowSet) -> float:
        probe = ReplayBuffer(buffer.capacity)
        probe.set_contents(candidate_ws)
        new_task = seen[-1]
        acc_new = task_aware_eval(model, new_task, probe, label_space, cfg,
                                  device, seed=seed, adapt=True, max_query=mq)["acc"]
        if len(seen) > 1:
            olds = [task_aware_eval(model, t, probe, label_space, cfg, device,
                                    seed=seed, adapt=True, max_query=mq)["acc"]
                    for t in seen[:-1]]
            acc_old = float(np.mean(olds))
        else:
            acc_old = acc_new
        return alpha * acc_old + (1 - alpha) * acc_new
    return reward


def run_rl_curation(strategy, buffer, candidates, ctx, cfg, first: bool):
    """Drive the RL policy: BC warm-start (once) then a short REINFORCE loop,
    with the contextual-bandit fallback already built into the strategy."""
    from memory import HerdingCuration
    if first:
        strategy.bc_warmstart(buffer, candidates, ctx, HerdingCuration(), steps=50)
    episodes = int(cfg["memory"]["rl"].get("episodes", 400))
    # cap episodes for tractability; each episode measures a post-adaptation reward
    for _ in range(min(episodes, 30)):
        strategy.train_episode(buffer, candidates, ctx)
    strategy.update(buffer, candidates, ctx)          # final greedy selection


# ===========================================================================
# Native CARE score (binary detection mode, §8)
# ===========================================================================
def compute_care_score(model, cur, cfg, device) -> dict:
    """Score the model as an anomaly detector over each farm's prediction frames.

    A window is 'anomaly' if the model's task-masked argmax leaves the normal
    column. Per-window predictions are expanded back to per-row over the
    prediction frame, then scored with the official CARE metric.
    """
    ls = cur.label_space
    normal_ids = cfg["data"].get("normal_status_ids", [0, 2])
    evals: list[DatasetEval] = []

    # collect the prediction frames gathered by the real loader
    frames = []
    for task in cur:
        frames.extend(task.care_frames)
    seen_ids = set()
    for fr in frames:
        if fr.dataset_id in seen_ids:
            continue
        seen_ids.add(fr.dataset_id)
        if len(fr.windows) == 0:
            continue
        # binary detection: any non-normal argmax over the full head (batched)
        with torch.no_grad():
            model.eval()
            ws = fr.windows
            bs, chunks = 256, []
            for s in range(0, len(ws), bs):
                b = ws.subset(np.arange(s, min(s + bs, len(ws))))
                X = torch.as_tensor(b.X, dtype=torch.float32, device=device)
                m = torch.as_tensor(b.mask, dtype=torch.float32, device=device)
                chunks.append(model(X, m).argmax(1).cpu().numpy())
            preds_win = (np.concatenate(chunks) != ls.normal_index)
        # expand window predictions to per-row via each window's center row
        n_rows = len(fr.status_ids)
        row_pred = np.zeros(n_rows, bool)
        for w, center in enumerate(fr.scores_index):
            if 0 <= center < n_rows:
                row_pred[center] = row_pred[center] or bool(preds_win[w])
        evals.append(evaluate_dataset(
            row_pred, fr.status_ids, is_normal_only=fr.is_normal_only,
            event_window=fr.event_window, normal_status_ids=normal_ids,
            dataset_id=fr.dataset_id, beta=float(cfg["eval"]["beta"]),
            criticality_threshold=int(cfg["eval"]["criticality_threshold"])))

    if not evals:
        return {"CARE": None, "note": "no prediction frames available"}
    return care_score(evals, beta=float(cfg["eval"]["beta"]),
                      weights=cfg["eval"]["care_weights"],
                      accuracy_gate=float(cfg["eval"]["accuracy_gate"]))


# ===========================================================================
# Reporting
# ===========================================================================
def print_scoreboard(rows: list[dict]):
    print("\n" + "=" * 78)
    print(f"{'method':<20}{'inc_acc':>10}{'forget':>10}{'harm':>10}{'CARE':>10}")
    print("-" * 78)
    for r in rows:
        care = r.get("care", {})
        care_v = care.get("CARE") if isinstance(care, dict) else None
        care_s = f"{care_v:.3f}" if isinstance(care_v, float) else "  -  "
        print(f"{r['method']:<20}{r['avg_incremental_accuracy']:>10.3f}"
              f"{r['avg_forgetting']:>10.3f}{r['harmonic_mean_base_novel']:>10.3f}"
              f"{care_s:>10}")
    print("=" * 78)


# ===========================================================================
# Result persistence
# ---------------------------------------------------------------------------
# Every experiment (one method x one seed x one command) is written to its own
# self-describing folder under results/runs/, plus two master index files:
#
#   results/
#     runs/<timestamp>__<command>__<method>__seed<seed>/
#         summary.json         # everything: metrics, CARE, learning curve, per-task
#         insights.md          # human-readable narrative of the run
#         config.snapshot.yaml # exact config that produced it
#     INDEX.md                 # master table, one row per run (human)
#     index.jsonl              # one JSON line per run (machine)
#
# The layout is designed so that in a *fresh chat session* you can read
# results/INDEX.md for the overview and then a single summary.json / insights.md
# for the complete detail of any run -- no need to re-run anything.
# ===========================================================================
def results_root(cfg: dict) -> str:
    root = os.path.join(HERE, cfg["paths"]["results_dir"])
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    return root


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _env_info(cfg: dict) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "platform": platform.platform(),
        "dataset_root": cfg["data"]["root"],
    }


def save_experiment(cfg: dict, command: str, scoreboard: dict, ts: str) -> str:
    """Persist one run to its own folder and append it to the master index."""
    root = results_root(cfg)
    method, seed = scoreboard["method"], scoreboard["seed"]
    run_id = f"{ts}__{command}__{method}__seed{seed}"
    run_dir = os.path.join(root, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    record = dict(scoreboard)
    record["run_id"] = run_id
    record["command"] = command
    record["timestamp"] = ts
    record["env"] = _env_info(cfg)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=float)
    with open(os.path.join(run_dir, "insights.md"), "w", encoding="utf-8") as f:
        f.write(_render_insights(record))
    try:
        import yaml
        with open(os.path.join(run_dir, "config.snapshot.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:  # noqa: BLE001 - config snapshot is best-effort
        pass

    _append_index(root, record)
    print(f"[saved] {os.path.relpath(run_dir, HERE)}")
    return run_dir


def _append_index(root: str, record: dict):
    """Append a compact line to index.jsonl and a table row to INDEX.md."""
    care = record.get("care") if isinstance(record.get("care"), dict) else {}
    entry = {
        "run_id": record["run_id"], "timestamp": record["timestamp"],
        "command": record["command"], "method": record["method"],
        "seed": record["seed"], "synthetic": record.get("synthetic"),
        "inc_acc": record.get("avg_incremental_accuracy"),
        "forgetting": record.get("avg_forgetting"),
        "harmonic": record.get("harmonic_mean_base_novel"),
        "final_acc": record.get("final_avg_accuracy"),
        "CARE": care.get("CARE"),
    }
    with open(os.path.join(root, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=float) + "\n")

    index_md = os.path.join(root, "INDEX.md")
    if not os.path.exists(index_md):
        with open(index_md, "w", encoding="utf-8") as f:
            f.write("# CARE-IFD experiment index\n\n"
                    "One row per run. Full detail in `runs/<run_id>/summary.json` "
                    "and `runs/<run_id>/insights.md`.\n\n"
                    "| run_id | method | seed | synth | inc_acc | forget | harm | CARE |\n"
                    "|---|---|---|---|---|---|---|---|\n")
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
    with open(index_md, "a", encoding="utf-8") as f:
        f.write(f"| [{entry['run_id']}](runs/{entry['run_id']}/insights.md) "
                f"| {entry['method']} | {entry['seed']} | "
                f"{'Y' if entry['synthetic'] else 'N'} | {fmt(entry['inc_acc'])} | "
                f"{fmt(entry['forgetting'])} | {fmt(entry['harmonic'])} | "
                f"{fmt(entry['CARE'])} |\n")


def _render_insights(r: dict) -> str:
    """Human-readable single-run report (Markdown)."""
    L = []
    L.append(f"# Run `{r['run_id']}`\n")
    L.append(f"- **method**: {r['method']}")
    L.append(f"- **seed**: {r['seed']}")
    L.append(f"- **command**: {r['command']}")
    L.append(f"- **timestamp**: {r['timestamp']}")
    L.append(f"- **data**: {'SYNTHETIC (not benchmark)' if r.get('synthetic') else 'real CARE'}")
    L.append(f"- **device**: {r.get('device')}")
    L.append(f"- **tasks**: {r.get('n_tasks')} | **total classes**: {r.get('total_classes')}\n")

    L.append("## Headline metrics")
    L.append(f"- avg incremental accuracy : {r.get('avg_incremental_accuracy'):.4f}")
    L.append(f"- avg forgetting           : {r.get('avg_forgetting'):.4f}")
    L.append(f"- harmonic mean base/novel : {r.get('harmonic_mean_base_novel'):.4f}")
    L.append(f"- final avg accuracy       : {r.get('final_avg_accuracy'):.4f}\n")

    care = r.get("care")
    if isinstance(care, dict):
        L.append("## Native CARE score (binary detection mode)")
        for k, v in care.items():
            L.append(f"- {k:<22}: {v}")
        L.append("")

    L.append("## Per-task breakdown")
    L.append("| tid | task | after-learning acc | final acc | forgetting |")
    L.append("|---|---|---|---|---|")
    for t in r.get("per_task", []):
        def f(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        L.append(f"| {t['tid']} | {t['key']} | {f(t['acc_after_learning'])} "
                 f"| {f(t['acc_final'])} | {f(t['forgetting'])} |")
    L.append("")

    L.append("## Curriculum")
    for t in r.get("tasks", []):
        L.append(f"- T{t['tid']} `{t['key']}` (farm {t['farm']}, U={t['U']}, "
                 f"support={t['n_support']}, query={t['n_query']}, pool={t['n_pool']}): "
                 f"{', '.join(t['local_classes'])}")
    L.append("\n## Full learning curve (accuracy_matrix)")
    L.append("`accuracy_matrix[stage][task]` = accuracy on `task` after training "
             "through `stage` (see summary.json for the raw numbers).")
    return "\n".join(L) + "\n"


def save_group_summary(cfg: dict, command: str, label: str, runs: list[dict], ts: str):
    """Write an aggregate (mean +/- std over runs) alongside the per-run folders."""
    root = results_root(cfg)
    agg = {
        "label": label, "command": command, "timestamp": ts,
        "n_runs": len(runs),
        "methods": sorted({r["method"] for r in runs}),
        "seeds": sorted({r["seed"] for r in runs}),
        "per_method": {},
    }
    for method in agg["methods"]:
        rs = [r for r in runs if r["method"] == method]
        inc = metrics.mean_std([r["avg_incremental_accuracy"] for r in rs])
        fgt = metrics.mean_std([r["avg_forgetting"] for r in rs])
        har = metrics.mean_std([r["harmonic_mean_base_novel"] for r in rs])
        cares = [r["care"]["CARE"] for r in rs
                 if isinstance(r.get("care"), dict) and isinstance(r["care"].get("CARE"), (int, float))]
        agg["per_method"][method] = {
            "inc_acc_mean": inc[0], "inc_acc_std": inc[1],
            "forgetting_mean": fgt[0], "forgetting_std": fgt[1],
            "harmonic_mean": har[0], "harmonic_std": har[1],
            "care_mean": (metrics.mean_std(cares)[0] if cares else None),
            "run_ids": [r.get("run_id") for r in rs],
        }
    path = os.path.join(root, "runs", f"{ts}__{command}__{label}__AGGREGATE.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=float)
    print(f"[saved] {os.path.relpath(path, HERE)}")


# ===========================================================================
# Model checkpointing (train -> save; eval -> load, no retraining)
# ---------------------------------------------------------------------------
# A checkpoint captures everything needed to reproduce task-aware evaluation:
#   model.pt   - backbone weights (grown to the final head width)
#   memory.npz - the replay buffer (needed for eq-5 task-aware adaptation)
#   ckpt.json  - method, seed, config snapshot, backbone dims, curriculum keys
# The query sets are rebuilt deterministically from (config, seed), so they are
# not stored -- only what training produced that eval cannot recompute.
# ===========================================================================
def save_checkpoint(run_dir: str, state: dict, cfg: dict, method: str, seed: int):
    model, buffer, ls, cur = (state["model"], state["buffer"],
                              state["label_space"], state["curriculum"])
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    if buffer is not None and len(buffer):
        ws = buffer.as_windowset()
        np.savez(os.path.join(run_dir, "memory.npz"),
                 X=ws.X, mask=ws.mask, y=ws.y, task=ws.task,
                 farm=np.array(ws.farm, dtype=object))
    ckpt = {
        "method": method, "seed": seed, "synthetic": state.get("synthetic"),
        "n_channels": cur.n_channels, "window": cur.n_timesteps,
        "total_classes": ls.total_classes, "budget_B": int(cfg["memory"]["budget_B"]),
        "has_memory": bool(buffer is not None and len(buffer)),
        "tasks": [{"tid": t.tid, "key": t.key} for t in cur],
    }
    with open(os.path.join(run_dir, "ckpt.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    print(f"[checkpoint] model saved to {os.path.relpath(run_dir, HERE)}/model.pt")


def evaluate_checkpoint(cfg: dict, ckpt_dir: str, small: bool = False) -> dict:
    """Load a trained checkpoint and run ONLY task-aware evaluation + CARE score
    (no training). The curriculum is rebuilt deterministically from (config,seed)
    so query sets match those the model was trained/evaluated on."""
    with open(os.path.join(ckpt_dir, "ckpt.json"), "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    method, seed = ckpt["method"], ckpt["seed"]
    device = resolve_device(cfg)
    set_seed(seed)

    cur, synthetic = build_curriculum(cfg, seed, small=small)
    ls = cur.label_space

    # rebuild the model at the final (grown) width and load the trained weights
    model = CNNLSTMBackbone(
        n_channels=cur.n_channels,
        conv_channels=tuple(cfg["backbone"]["conv_channels"]),
        kernel=int(cfg["backbone"]["kernel"]),
        lstm_hidden=int(cfg["backbone"]["lstm_hidden"]),
        lstm_layers=int(cfg["backbone"]["lstm_layers"]),
        dropout=float(cfg["backbone"]["dropout"]),
        n_classes=ls.total_classes,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"),
                                     map_location=device))

    # restore the replay memory (needed for eq-5 adaptation)
    buffer = None
    mem_path = os.path.join(ckpt_dir, "memory.npz")
    if ckpt.get("has_memory") and os.path.exists(mem_path):
        z = np.load(mem_path, allow_pickle=True)
        buffer = ReplayBuffer(int(ckpt["budget_B"]))
        buffer.set_contents(dl.WindowSet(z["X"], z["mask"], z["y"], z["task"],
                                         list(z["farm"])))

    # task-aware evaluation across all tasks (final row of the accuracy matrix)
    R = {}
    task_ids = [t.tid for t in cur]
    last = task_ids[-1]
    R[last] = {}
    for t in cur:
        R[last][t.tid] = task_aware_eval(model, t, buffer, ls, cfg, device,
                                         seed=seed, adapt=(buffer is not None))["acc"]

    scoreboard = {
        "method": method, "seed": seed, "synthetic": synthetic,
        "n_tasks": len(cur), "total_classes": ls.total_classes, "device": device,
        "loaded_from": os.path.relpath(ckpt_dir, HERE),
        "avg_incremental_accuracy": float(np.mean(list(R[last].values()))),
        "avg_forgetting": float("nan"),   # forgetting needs the full training curve
        "harmonic_mean_base_novel": metrics.harmonic_mean_base_novel(R, task_ids),
        "final_avg_accuracy": float(np.mean(list(R[last].values()))),
        "tasks": [{"tid": t.tid, "key": t.key, "farm": t.farm, "U": t.U,
                   "local_classes": t.local_classes, "n_support": len(t.support),
                   "n_query": len(t.query), "n_pool": len(t.pool)} for t in cur],
        "accuracy_matrix": {str(last): {str(j): v for j, v in R[last].items()}},
        "per_task": [{"tid": t.tid, "key": t.key,
                      "acc_after_learning": None, "acc_final": R[last][t.tid],
                      "forgetting": None} for t in cur],
    }
    if cfg["eval"].get("report_care_score", True) and not synthetic:
        scoreboard["care"] = compute_care_score(model, cur, cfg, device)
    return scoreboard


# ===========================================================================
# CLI commands
# ===========================================================================
def cmd_selftest(args, cfg):
    import selftest
    return selftest.run_all()


def cmd_baselines(args, cfg):
    """Run the baseline ladder (Table I + Table II) on a SMALL subset."""
    ladder = args.methods or [
        "fine_tuning", "ewc", "icarl",
        "reptile_random", "reptile_herding", "reptile_reservoir",
        "reptile_gss", "proposed",
    ]
    ts = timestamp()
    rows = []
    for method in ladder:
        print(f"\n>>> running baseline: {method}")
        r = run_incremental(cfg, seed=cfg["seed"], method=method, small=True)
        save_experiment(cfg, "baselines", r, ts)     # each method -> its own run folder
        rows.append(r)
    print_scoreboard(rows)
    save_group_summary(cfg, "baselines", "ladder", rows, ts)
    return 0


def cmd_train(args, cfg):
    """Full incremental run of one method over the eval seeds (default: proposed)."""
    method = args.method or "proposed"
    seeds = cfg["eval"]["seeds"] if args.all_seeds else [cfg["seed"]]
    ts = timestamp()
    runs = []
    for s in seeds:
        r, state = run_incremental(cfg, seed=s, method=method, small=args.small,
                                   return_state=True)
        run_dir = save_experiment(cfg, "train", r, ts)   # one folder per seed
        save_checkpoint(run_dir, state, cfg, method, s)  # + trained model for later eval
        runs.append(r)
    for r in runs:
        print(f"  seed {r['seed']}: inc_acc={r['avg_incremental_accuracy']:.3f} "
              f"forget={r['avg_forgetting']:.3f} harm={r['harmonic_mean_base_novel']:.3f}")
    inc = metrics.mean_std([r["avg_incremental_accuracy"] for r in runs])
    fgt = metrics.mean_std([r["avg_forgetting"] for r in runs])
    print(f"\n{method}: inc_acc = {inc[0]:.3f} +/- {inc[1]:.3f} | "
          f"forgetting = {fgt[0]:.3f} +/- {fgt[1]:.3f}")
    if len(runs) > 1:
        save_group_summary(cfg, "train", method, runs, ts)
    return 0


def cmd_eval(args, cfg):
    """Evaluate a trained model. With --checkpoint, load a saved model and only
    run task-aware eval + CARE (no retraining); otherwise train-then-eval in one
    pass (convenience)."""
    if args.checkpoint:
        ckpt_dir = args.checkpoint if os.path.isabs(args.checkpoint) \
            else os.path.join(HERE, args.checkpoint)
        if not os.path.isfile(os.path.join(ckpt_dir, "ckpt.json")):
            print(f"[error] no checkpoint (ckpt.json) found in {ckpt_dir!r}")
            return 1
        print(f">>> evaluating checkpoint: {ckpt_dir}")
        r = evaluate_checkpoint(cfg, ckpt_dir, small=args.small)
    else:
        method = args.method or "proposed"
        r = run_incremental(cfg, seed=cfg["seed"], method=method, small=args.small)
    print_scoreboard([r])
    if isinstance(r.get("care"), dict):
        print("\nCARE sub-scores:")
        for k, v in r["care"].items():
            print(f"  {k:<22}: {v}")
    save_experiment(cfg, "eval", r, timestamp())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CARE-IFD: RL-guided incremental fault diagnosis")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("selftest", help="cheap invariants (no data/GPU needed)")

    b = sub.add_parser("baselines", help="run the baseline ladder on a small subset")
    b.add_argument("--methods", nargs="*", help="subset of methods to run")

    t = sub.add_parser("train", help="full incremental run of one method")
    t.add_argument("--method", default=None, choices=list(METHOD_PRESETS))
    t.add_argument("--all-seeds", action="store_true", help="run over eval.seeds")
    t.add_argument("--small", action="store_true", help="use the small/fast subset")

    e = sub.add_parser("eval", help="evaluate a trained model + native CARE score")
    e.add_argument("--checkpoint", default=None,
                   help="path to a saved train run folder (contains model.pt/ckpt.json); "
                        "loads the trained model instead of retraining")
    e.add_argument("--method", default=None, choices=list(METHOD_PRESETS),
                   help="method to train-then-eval when --checkpoint is not given")
    e.add_argument("--small", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    dispatch = {"selftest": cmd_selftest, "baselines": cmd_baselines,
                "train": cmd_train, "eval": cmd_eval}
    return dispatch[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
