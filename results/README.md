# results/ — experiment records

Every run of `main.py {train,eval,baselines}` writes a **self-contained record**
here so results can be reviewed later (including in a fresh chat session) without
re-running anything.

## Where to start
1. **`INDEX.md`** — master table, one row per run (method, seed, headline metrics,
   CARE). Read this first; each row links to that run's `insights.md`.
2. **`index.jsonl`** — the same index, one JSON object per line (machine-readable).

## Layout
```
results/
  INDEX.md                     # human overview (append-only table)
  index.jsonl                  # machine overview (one JSON line per run)
  runs/
    <timestamp>__<command>__<method>__seed<seed>/
        summary.json           # complete record (see fields below)
        insights.md            # human-readable narrative of this one run
        config.snapshot.yaml   # exact config that produced the run
    <timestamp>__<command>__<label>__AGGREGATE.json   # mean±std across seeds/methods
```
`timestamp` = `YYYYMMDD-HHMMSS`. A `--all-seeds` train or a `baselines` ladder
writes one run folder per (method, seed) plus one `AGGREGATE.json`.

## `summary.json` fields
| field | meaning |
|---|---|
| `method`, `seed`, `command`, `timestamp`, `run_id` | run identity |
| `synthetic` | `true` if the synthetic fallback was used (NOT benchmark numbers) |
| `device`, `env` | cpu/cuda + python/torch/gpu/platform/dataset_root |
| `n_tasks`, `total_classes` | curriculum size and final head width |
| `avg_incremental_accuracy` | mean over stages of accuracy on tasks seen so far (§9) |
| `avg_forgetting` | mean over tasks of (best past acc − final acc) (§9) |
| `harmonic_mean_base_novel` | harmonic mean of base-task vs novel-task final accuracy |
| `final_avg_accuracy` | mean final accuracy across all tasks |
| `care` | native CARE score + sub-scores (Coverage/Earliness/Reliability, accuracy gate); real data only |
| `tasks` | per-task curriculum metadata: key, farm, U, classes, support/query/pool sizes |
| `per_task` | per-task after-learning acc, final acc, forgetting |
| `accuracy_matrix` | `[stage][task]` = accuracy on `task` after training through `stage` — the full learning curve |

## `AGGREGATE.json`
`per_method[method]` holds `inc_acc_mean/std`, `forgetting_mean/std`,
`harmonic_mean/std`, `care_mean`, and the contributing `run_ids`.

> This folder is gitignored. Delete `runs/` (and `INDEX.md` / `index.jsonl`) to
> reset the history; new runs recreate the index.
