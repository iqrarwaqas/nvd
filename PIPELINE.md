# Learning Not to Forget — Implementation Brief

RL-guided incremental fault diagnosis for wind turbines on the **CARE** benchmark.
This document is the build spec for a coding agent. It corrects the paper where CARE's
label reality forces it, and flags every deviation from the manuscript.

Dataset root (fixed): `../Datasets/Care`
Framework: PyTorch. Single flat folder, modular `.py` files. No package tree.

---

## 0. Corrections to the paper (read first)

The paper is implementable, but two things in it do **not** survive contact with CARE.
Both are handled here; do not silently revert them.

1. **CARE has no per-farm fault-class taxonomy.** It has dataset-level event labels
   (anomaly / normal) and per-timestamp `status_type_id`. Only Farm A (EDP, onshore)
   has real component fault types; farms B and C (offshore) are anonymised → one usable
   label (anomaly). So the setting is **domain-incremental with a variable-width task head**,
   not pure class-incremental.
   - **Fix:** each task adds `U_t >= 1` fault channels + one shared `normal` channel.
     Farm A tasks: `U_t` = number of distinct logbook fault types present.
     Farms B, C: `U_t = 1` (a single `anomaly` channel). Eq. (2) head and eq. (6) masking
     are preserved exactly; only `U` becomes per-task.

2. **REINFORCE over 3 farm-tasks will not learn.** Length-3 curricula give the policy
   almost no signal.
   - **Fix (all enabled by default):** define tasks at `farm x fault_group` granularity to
     get 8-15 tasks; resample task orderings each episode; behavior-clone the policy from
     `herding` before REINFORCE; keep a contextual-bandit fallback for the curation head.

Smaller: add **GSS** (gradient-based sample selection, ref [16]) to the curation ablation —
random/herding/reservoir alone are too weak a comparison for a "learned curation" claim.
Meta-update eq. (4) is a nonstandard Reptile variant; `meta.py` implements both it and
canonical Reptile behind a config flag (`meta.update_rule`).

---

## 1. Repo layout (one folder)

```
care_ifd/
  config.yaml          # all hyperparameters + dataset path
  dataloader.py        # CARE parsing, windowing, task/curriculum construction
  labels.py            # label definition + variable-U task partition (the swappable layer)
  backbone.py          # CNN-LSTM + task-partitioned head
  meta.py              # Reptile meta-train (eq 3,4) + task-aware inference (eq 5,6)
  memory.py            # replay buffer + curation strategies (heuristics, GSS, RL policy)
  care_score.py        # official CARE scorer (Coverage, Earliness, Reliability; Acc as gate)
  metrics.py           # avg incremental accuracy, avg forgetting, harmonic mean
  main.py              # CLI: selftest | baselines | train | eval
  selftest.py          # fast synthetic + tiny-real assertions (no pytest)
```

Two-tier CLI verification (no pytest fixtures):
`python main.py selftest` → cheap invariants; `python main.py baselines` → runs the baseline
ladder end-to-end on a small subset and prints the scoreboard.

---

## 2. config.yaml (defaults)

```yaml
data:
  root: "../Datasets/Care"
  window: 144            # 24h at 10-min resolution
  stride: 6              # 1h hop; tune vs class balance
  farms: [A, B, C]
  zero_mask: true        # farms B,C encode missing as 0 -> mask, do NOT feed as real
  standardize: per_channel_per_farm
  lazy: true             # stream CSVs, never load all farms into RAM

task:
  granularity: farm_x_faultgroup   # {farm | farm_x_faultgroup | turbine}
  shots_K: 5                       # sweep {1,5,10}
  order: fixed                     # fixed | resampled (resampled for RL training)
  protocol: incremental_farms      # incremental_farms | leave_one_event_out

backbone:
  conv_channels: [64, 64, 128]
  kernel: 5
  lstm_hidden: 128
  lstm_layers: 1
  dropout: 0.2

meta:
  update_rule: reptile_canonical   # reptile_canonical | paper_eq4_avg
  inner_steps_r: 5
  inner_lr_alpha: 1.0e-3
  beta: 2.0            # meta-step decay (paper eq 4)
  base_task_epochs: 30

memory:
  budget_B: 200
  strategy: rl        # random | herding | reservoir | gss | rl
  rl:
    reward_alpha: 0.5   # alpha_r in eq (7): old vs new accuracy weight
    reward_lambda: 0.1  # occupancy penalty
    bc_warmstart_from: herding
    episodes: 400
    baseline: moving_avg   # variance reduction for REINFORCE
    fallback: bandit       # bandit | none

eval:
  seeds: [0, 1, 2]
  report_care_score: true
```

---

## 3. Data layer (`dataloader.py`, `labels.py`)

### 3.1 CARE structure and known gotchas
Each dataset = one CSV (one turbine, ~1yr train + prediction window) plus event metadata.
Columns present (confirm names against the local copy — they vary slightly):
`id`, `time_stamp`, `asset`/turbine id, `train_test`, `status_type_id`, then `sensor_*`.

Bake in these findings so the agent does not rediscover them:
- **Locate events by the `id` column, not by timestamp.** Timestamps have gaps; the id
  ordering is the reliable index for event start/end.
- **`pred_features` bug:** the prediction-window feature slice is the known trap — validate
  that prediction rows align with the event id range before scoring.
- **Zero-masking (farms B, C):** long runs of consecutive `0` are missing-value fill, not
  real readings. Mask them (attention/loss mask), never standardize or window over them as data.
- **`status_type_id` mapping is explicit and farm-specific** — load from a verified mapping
  table, do not assume ids mean the same thing across farms. Statuses for B/C are logged only
  on change → forward-fill within a turbine before use.
- **Lazy loading:** iterate CSV -> windows on demand; hold only the active task's windows +
  the fixed memory `M` in RAM.

### 3.2 Windowing
Per-channel, per-farm standardization fit on that farm's *training* year only (no leakage
from prediction windows). Slice `X in R^{T x C}` with `window`/`stride`. Channel counts differ
per farm → maintain a per-farm channel index; the backbone sees a farm-max `C` with unused
channels zero-masked, OR align to a shared subset (document which).

### 3.3 Label definition — the swappable layer (`labels.py`)
This is where the paper's "task owns U fault channels" is realised honestly.

- `window_label(window, status_ids, event_meta)` returns one of:
  `normal`, or a fault-class id for that task's block.
- **Farm A:** map event `fault_description`/logbook to a small fixed set of component classes
  (generator, gearbox, transformer, hydraulic, bearing, ...). Read the actual present classes
  at build time; do not hardcode counts. `U_A` = number present.
- **Farms B, C:** single `anomaly` class each → `U_B = U_C = 1`.
- Output space is the union: `{normal} ∪ block(task_1) ∪ ... ∪ block(task_S)` where
  `block(t)` has width `U_t`. This is the variable-width version of eq. (2)/(6).
- A `binary` mode (all faults collapse to `anomaly`) must exist for the detection ablation
  and for computing the native CARE score.

Definition of done for section 3: `selftest` asserts (a) no window spans a masked zero-run,
(b) every anomaly event is recovered by id range with correct label, (c) standardization
stats were fit on train rows only, (d) `sum(U_t)` matches the discovered class inventory.

---

## 4. Task / curriculum construction (`dataloader.py`)

- `granularity = farm_x_faultgroup` splits each farm into sub-tasks per fault group → longer
  curriculum (needed for RL). `farm` and `turbine` also selectable.
- Base task = most populous regime/classes (per paper). Subsequent tasks introduce their
  `U_t` classes with `K in {1,5,10}` labelled windows each.
- Task identity is provided at inference (operationally, the source farm is known).
- Memory budget `B` is identical across all compared methods (fairness).
- Two eval protocols:
  - `incremental_farms`: A -> B -> C sub-tasks in sequence; evaluate on all seen tasks.
  - `leave_one_event_out`: hold one anomaly event out, train on the rest (matches CARE's
    event-based evaluation; use for the CARE-score numbers).

---

## 5. Backbone (`backbone.py`)

CNN-LSTM, task-partitioned head (paper IV-B).
- 1D conv stack over time → `H = CNN(X)`; LSTM over conv features → `h_t`.
- Linear head to `|C| = 1 + sum_t U_t` logits.
- Head is grown as tasks arrive (add columns for the new block); old columns preserved.
- `forward(X, task_id=None)`: if `task_id` given, apply eq-(6) masking to
  `C_t = {normal} ∪ block(task_t)` (set others to `-inf`).

Definition of done: forward pass shape checks; growing the head does not perturb existing
logits for a fixed input (freeze test).

---

## 6. Meta-trainer + task-aware inference (`meta.py`)

### 6.1 Reptile incremental meta-training (eq 3, 4)
- Init `Phi <- Phi_{t-1}`, store `Phi_base`.
- For each seen task `i <= t`: reset to base, gather batch `B_i` (labels in task i's block
  from `D_t ∪ M`), run `r` inner SGD steps on cross-entropy (eq 3).
- Meta-update (config `meta.update_rule`):
  - `reptile_canonical`: `Phi <- Phi + eps * mean_i(Phi_i - Phi)`.
  - `paper_eq4_avg`: `eta = exp(-beta * t/S); Phi <- eta * (1/t) sum_i Phi_i + (1-eta) Phi_base`.
- Default to `reptile_canonical`; report the paper variant as an ablation.

### 6.2 Task-aware inference (eq 5, 6)
- Given known `t`: `M_t = filter(M, t)`, clone weights, `r` steps of adaptation on `M_t`
  with out-of-task labels relabelled to a shared background/`normal` symbol (eq 5).
- Restrict prediction to `C_t`, mask the rest to `-inf`, argmax (eq 6).

Definition of done: on a synthetic 2-task problem, task-aware adaptation raises target-task
accuracy above the no-adapt baseline; masking never predicts an out-of-task class.

---

## 7. Memory + curation (`memory.py`) — the contribution

Fixed buffer `M`, capacity `B`, identical across all strategies.

Strategies (all selectable via `memory.strategy`):
- `random`, `herding` (toward class means), `reservoir` — heuristic baselines.
- `gss` — gradient-based sample selection (ref [16]); the strong learned-ish baseline.
- `rl` — the policy `pi_phi`:
  - **State** (eq): `[ mean new-task embedding, per-class occupancy {n_c}, running forgetting
    estimate delta_forget on a held-out split ]`. Keep it dimension-independent so it
    transfers across farms with different `C`.
  - **Action:** per candidate (incumbents + new K-shot), keep/replace, subject to `|M| <= B`.
  - **Reward** (eq 7), measured *after* task-aware adaptation (eq 5):
    `r = alpha_r * Acc_old + (1 - alpha_r) * Acc_new - lambda * |M|/B`.
  - **Optimisation:** REINFORCE (eq 8) with moving-average baseline `b`.
  - **Feasibility scaffolding (required):** BC warm-start from `herding`; resampled task
    orderings across episodes; bandit fallback (`memory.rl.fallback`) if REINFORCE variance
    stalls — treat each curation decision as a contextual bandit with the same reward.

Definition of done: (a) buffer never exceeds `B`; (b) on a rigged toy task where the
old-task-protective exemplars are identifiable, the RL policy recovers them with higher
frequency than random after warm-start; (c) reward is computed post-adaptation, not pre.

---

## 8. CARE scorer (`care_score.py`)

Official scoring for the detection evaluation. **Critical:** use `omega = (1, 1, 1, 0)` —
Accuracy is a **gate condition**, not a weighted term. (Using a weighted Accuracy term is the
known bug that inflates trivial models, e.g. IsolationForest ~0.50 vs published ~0.14.)
Implement Coverage, Earliness, Reliability; apply Accuracy as the pass/fail gate on normal
datasets. Validate against the published reference score on a known baseline before trusting it.

---

## 9. Baseline ladder (`main.py baselines`) and metrics (`metrics.py`)

Table I (main comparison):
1. Fine-tuning (lower bound)
2. EWC [13]
3. iCaRL [14]
4. Reptile + task-aware backbone + heuristic memory (random / herding / reservoir)
5. **Proposed:** Reptile + task-aware + RL-curated memory

Table II (the crux ablation — identical backbone + budget `B`):
`random | herding | reservoir | gss | rl(ours)`.

Metrics: average incremental accuracy, average forgetting, harmonic mean of base/novel
accuracy, all **after** task-aware adaptation; plus the native CARE score in `binary` label
mode. Report mean +/- std over `seeds`.

---

## 10. Milestones (each with a definition of done)

- **M1 Data:** `dataloader`+`labels` produce correct windows/labels; selftest passes 3.3 DoD.
- **M2 Backbone+meta:** synthetic 2-task learning + task-aware adaptation works (6.1/6.2 DoD).
- **M3 CARE score:** matches published reference on a known baseline (section 8 DoD).
- **M4 Heuristic replay:** full pipeline runs A->B->C, produces Table I rows 1-4 + Table II
  heuristic rows. **This is the honest baseline the paper must beat.**
- **M5 GSS:** GSS row added to Table II.
- **M6 RL curation:** policy trains (with BC warm-start); RL row filled. **Go/no-go:** if RL
  does not beat herding+GSS on avg forgetting at equal `B`, the paper's core claim fails —
  report that outcome honestly rather than tuning until it wins.
- **M7 Sweeps:** K in {1,5,10}, beta, alpha_r; write-up.

---

## 11. Known risks

- **RL underperforms heuristics.** Most likely failure. M6 is the go/no-go; do not p-hack.
- **Too few tasks for "forgetting" to be meaningful.** Mitigated by `farm_x_faultgroup`
  granularity; if forgetting is near-zero everywhere, the incremental framing is weak and the
  paper should pivot to the domain-shift/earliness story.
- **Offshore U_t = 1.** The multi-class-diagnosis novelty lives mostly on Farm A; be explicit
  in the paper that B/C are detection tasks, or a reviewer will read the offshore title as
  overclaiming.
- **Positioning.** Must differentiate from GSS [16] and any prior RL-for-replay work; the delta
  is the diagnosis-specific post-adaptation reward + task-aware meta backbone.
