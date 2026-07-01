# CARE-IFD

RL-guided incremental fault diagnosis for wind turbines on the **CARE** benchmark.
Implementation of the build spec in [`PIPELINE.md`](PIPELINE.md).

Everything runs through **`main.py`**; the rest of the folder is modular code it
orchestrates.

## Layout (one flat folder)

| file            | role |
|-----------------|------|
| `config.yaml`   | all hyperparameters + dataset path |
| `labels.py`     | label definition + variable-U task partition (the swappable layer, §3.3) |
| `dataloader.py` | CARE parsing, zero-masked windowing, task/curriculum construction, synthetic generator |
| `backbone.py`   | CNN-LSTM + growable task-partitioned head with eq-(6) masking (§5) |
| `meta.py`       | Reptile meta-training (eq 3,4) + task-aware inference (eq 5,6) (§6) |
| `memory.py`     | replay buffer + curation strategies: random / herding / reservoir / gss / **rl** (§7) |
| `care_score.py` | official CARE scorer — Coverage, Earliness, Reliability; Accuracy as a **gate** (§8) |
| `metrics.py`    | avg incremental accuracy, avg forgetting, harmonic mean (§9) |
| `selftest.py`   | fast synthetic + tiny-real assertions covering the §10 DoDs |
| `main.py`       | CLI orchestrator: `selftest \| baselines \| train \| eval` |

## Quick start

```bash
pip install -r requirements.txt

python main.py selftest                 # cheap invariants; no data/GPU needed
python main.py baselines                # baseline ladder on a small subset -> scoreboard
python main.py train --method proposed --all-seeds
python main.py eval  --method proposed  # + native CARE score
```

## Dataset

Point `data.root` in `config.yaml` at the CARE download (default `../Datasets/Care`),
which must contain `Wind Farm A/`, `Wind Farm B/`, `Wind Farm C/`, each with a
`datasets/*.csv` folder, `event_info.csv`, and `feature_description.csv`
(semicolon-separated CSVs).

**If the dataset is absent**, the data-dependent commands fall back to a synthetic
curriculum so the whole pipeline still runs — those numbers are for validation
only, not benchmark results (the fallback prints a loud warning).

## Notes / deviations from the paper

See §0 of `PIPELINE.md`. In short: variable-width task heads (CARE has no per-farm
fault taxonomy off Farm A), `farm_x_faultgroup` task granularity for a longer RL
curriculum, BC warm-start + bandit fallback for the RL curation policy, and the
CARE score uses `omega = (1,1,1,0)` with Accuracy as a pass/fail gate.
