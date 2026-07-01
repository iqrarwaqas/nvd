"""CARE parsing, windowing and task/curriculum construction (PIPELINE.md §3, §4).

The rest of the pipeline consumes three plain data structures produced here:

    WindowSet   - a batch of windows  X[N, C, T], per-timestep valid mask[N, T],
                  global class labels y[N], task id[N], and the source farm.
    Task        - one curriculum step: a K-shot support set, a query set, and a
                  labelled candidate pool for the replay memory.
    Curriculum  - the ordered list of Tasks + the shared LabelSpace.

Two builders exist:

  * :func:`build_care_curriculum` -- the real loader. Streams CARE CSVs, windows
    them with zero-masking, standardizes per-channel/per-farm on the *training*
    year only, and partitions into tasks at the configured granularity.
  * :func:`make_synthetic_curriculum` -- a fast, dependency-light generator that
    produces the same structures for selftests and for running the pipeline
    end-to-end without the dataset present.

CARE layout (confirmed against a local copy):

    <root>/Wind Farm A/datasets/<event_id>.csv     (sep=';')
                       /event_info.csv               (event_id,event_label,
                                                       event_start_id,event_end_id,
                                                       [event_description])
                       /feature_description.csv
    columns: time_stamp, asset_id, id, train_test, status_type_id, <sensor>_<stat>
    train_test in {"train","prediction"}; events located by the `id` column.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from labels import (
    DEFAULT_NORMAL_STATUS_IDS,
    FARM_IDS,
    META_COLS,
    NORMAL,
    LabelSpace,
    fault_group_of,
    window_label,
)

# Canonical cross-farm signals used when data.channel_mode == "shared".
# Per-farm column mapping derived from each farm's feature_description.csv
# (matched by physical meaning). Missing columns fail loudly at load time.
SHARED_SIGNALS = ["active_power", "wind_speed", "reactive_power",
                  "rotor_speed", "ambient_temp", "gearbox_oil_temp"]

FARM_SIGNAL_MAP: dict[str, dict[str, str]] = {
    "A": {"active_power": "power_30_avg", "wind_speed": "wind_speed_3_avg",
          "reactive_power": "sensor_31_avg", "rotor_speed": "sensor_52_avg",
          "ambient_temp": "sensor_0_avg", "gearbox_oil_temp": "sensor_12_avg"},
    "B": {"active_power": "power_62_avg", "wind_speed": "wind_speed_61_avg",
          "reactive_power": "reactive_power_11_avg", "rotor_speed": "sensor_25_avg",
          "ambient_temp": "sensor_8_avg", "gearbox_oil_temp": "sensor_39_avg"},
    "C": {"active_power": "power_6_avg", "wind_speed": "wind_speed_235_avg",
          "reactive_power": "reactive_power_122_avg", "rotor_speed": "sensor_144_avg",
          "ambient_temp": "sensor_7_avg", "gearbox_oil_temp": "sensor_186_avg"},
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
@dataclass
class WindowSet:
    """A batch of time windows in channel-first layout for Conv1d.

    X    : float32 [N, C, T]      standardized sensor windows
    mask : float32 [N, T]         1 = valid timestep, 0 = masked (missing/zero-run)
    y    : int64   [N]            global class index (LabelSpace); -1 = unlabelled
    task : int64   [N]            task id each window belongs to
    farm : list[str] length N
    """

    X: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    task: np.ndarray
    farm: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_timesteps(self) -> int:
        return int(self.X.shape[2])

    def subset(self, idx) -> "WindowSet":
        idx = np.asarray(idx, dtype=int)
        return WindowSet(
            X=self.X[idx], mask=self.mask[idx], y=self.y[idx],
            task=self.task[idx], farm=[self.farm[i] for i in idx],
        )

    @staticmethod
    def concat(parts: list["WindowSet"]) -> "WindowSet":
        parts = [p for p in parts if p is not None and len(p) > 0]
        if not parts:
            raise ValueError("cannot concat empty WindowSet list")
        return WindowSet(
            X=np.concatenate([p.X for p in parts], axis=0),
            mask=np.concatenate([p.mask for p in parts], axis=0),
            y=np.concatenate([p.y for p in parts], axis=0),
            task=np.concatenate([p.task for p in parts], axis=0),
            farm=[f for p in parts for f in p.farm],
        )


@dataclass
class Task:
    """One curriculum step (PIPELINE.md §4)."""

    key: str                 # e.g. "A/gearbox", "B/anomaly"
    tid: int                 # task id (position in the curriculum)
    farm: str
    local_classes: list[str]  # fault-group names owned by this task, width U_t
    support: WindowSet        # K-shot labelled windows (few-shot novelty)
    query: WindowSet          # evaluation windows
    pool: WindowSet           # labelled candidates eligible for the replay memory
    # Raw per-dataset evaluation frames for the native CARE score (filled by the
    # real loader; empty for synthetic data).
    care_frames: list = field(default_factory=list)

    @property
    def U(self) -> int:
        return len(self.local_classes)


@dataclass
class Curriculum:
    tasks: list[Task]
    label_space: LabelSpace
    n_channels: int
    n_timesteps: int

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)


@dataclass
class CareFrame:
    """Prediction-frame arrays for one dataset, consumed by care_score.py."""

    dataset_id: str
    farm: str
    scores_index: np.ndarray   # window -> center row position (for reconstructing per-row preds)
    status_ids: np.ndarray     # status_type_id per prediction row
    is_normal_only: bool
    event_window: tuple[int, int] | None   # (start_pos, end_pos) in prediction-row index
    windows: WindowSet         # windows over the prediction frame (unlabelled: y=-1)


# ===========================================================================
# Real CARE loader
# ===========================================================================
def _farm_dir(root: str, farm_id: str) -> str:
    d = os.path.join(root, f"Wind Farm {farm_id}")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Expected farm directory {d!r}. CARE layout requires "
            f"'Wind Farm A/B/C' folders under root={root!r}.")
    return d


def _feature_columns(df) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _channel_columns(farm_id: str, feats: list[str], channel_mode: str) -> list[str]:
    """Resolve which columns feed the backbone for this farm."""
    if channel_mode == "shared":
        mapping = FARM_SIGNAL_MAP[farm_id]
        cols = []
        for sig in SHARED_SIGNALS:
            col = mapping[sig]
            if col not in feats:
                raise ValueError(
                    f"Farm {farm_id}: shared signal {sig!r} -> {col!r} not in data. "
                    f"Resolve FARM_SIGNAL_MAP before proceeding (§3).")
            cols.append(col)
        return cols
    # farm_max: use every sensor '*_avg' channel (min/max/std dropped to bound C).
    return [c for c in feats if c.endswith("_avg")] or feats


def _zero_run_mask(x: np.ndarray, min_run: int) -> np.ndarray:
    """Per-row validity mask: rows inside an all-zero run >= ``min_run`` -> invalid.

    Farms B, C fill missing values with 0. Long consecutive zero runs are missing
    data and must be masked (never standardized or windowed as real readings).
    """
    all_zero = np.all(x == 0.0, axis=1)
    valid = np.ones(x.shape[0], dtype=bool)
    i, n = 0, x.shape[0]
    while i < n:
        if not all_zero[i]:
            i += 1
            continue
        j = i
        while j < n and all_zero[j]:
            j += 1
        if (j - i) >= min_run:
            valid[i:j] = False
        i = j
    return valid


def _make_windows(x: np.ndarray, row_valid: np.ndarray, status: np.ndarray,
                  window: int, stride: int):
    """Slice standardized rows into windows, dropping any window that overlaps a
    masked zero-run (§3.3 DoD: no window spans a masked zero-run).

    Returns (X[n, C, T], mask[n, T], center_rows[n], status_per_window[n, T]).
    """
    n_rows, n_ch = x.shape
    Xs, masks, centers, statuses = [], [], [], []
    for start in range(0, max(0, n_rows - window + 1), stride):
        sl = slice(start, start + window)
        if not row_valid[sl].all():
            continue  # window overlaps missing data -> drop entirely
        Xs.append(x[sl].T)                      # [C, T]
        masks.append(np.ones(window, dtype=np.float32))
        centers.append(start + window // 2)
        statuses.append(status[sl])
    if not Xs:
        empty = np.zeros((0, n_ch, window), np.float32)
        return empty, np.zeros((0, window), np.float32), np.zeros((0,), int), np.zeros((0, window), int)
    return (np.asarray(Xs, np.float32), np.asarray(masks, np.float32),
            np.asarray(centers, int), np.asarray(statuses, int))


def _read_event_info(fdir: str, sep: str):
    import pandas as pd
    path = os.path.join(fdir, "event_info.csv")
    if not os.path.exists(path):
        return {}
    ev = pd.read_csv(path, sep=sep)
    out = {}
    # description column name varies across releases; take the first that exists.
    desc_col = next((c for c in ("event_description", "description",
                                 "fault_description", "comment") if c in ev.columns), None)
    for _, row in ev.iterrows():
        eid = str(row.get("event_id"))
        out[eid] = {
            "label": str(row.get("event_label", NORMAL)).strip().lower(),
            "start_id": row.get("event_start_id"),
            "end_id": row.get("event_end_id"),
            "description": row.get(desc_col) if desc_col else None,
        }
    return out


def _event_window_positions_from_ids(ids, start_id, end_id):
    """Translate event start/end `id` values into positional row indices.

    Events are located by the `id` column (timestamps have gaps), per PIPELINE.md
    §3.1. Returns None for normal-only datasets (no event id range).
    """
    import pandas as pd
    if start_id is None or end_id is None or pd.isna(start_id) or pd.isna(end_id):
        return None
    id_to_pos = {int(v): i for i, v in enumerate(ids)}
    s = id_to_pos.get(int(start_id))
    e = id_to_pos.get(int(end_id))
    if s is None or e is None:
        # event window lies outside this frame's id range -> no positive window here
        return None
    return (min(s, e), max(s, e))


def build_care_curriculum(cfg: dict, seed: int = 0) -> Curriculum:
    """Build the real curriculum from CARE CSVs (PIPELINE.md §3 & §4).

    Streams each farm, standardizes on the training year only, windows with
    zero-masking, assigns labels via :mod:`labels`, and groups windows into tasks
    at ``task.granularity``. Requires pandas + the dataset on disk.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    data, taskcfg = cfg["data"], cfg["task"]
    root = data["root"]
    sep = data.get("sep", ";")
    window, stride = int(data["window"]), int(data["stride"])
    normal_ids = tuple(data.get("normal_status_ids", DEFAULT_NORMAL_STATUS_IDS))
    channel_mode = data.get("channel_mode", "shared")
    zero_run_min = int(data.get("zero_run_min", 6))
    K = int(taskcfg["shots_K"])
    binary = taskcfg.get("granularity") == "farm"  # farm granularity collapses fault types

    label_space = LabelSpace(binary=(taskcfg.get("label_mode") == "binary"))
    task_windows: dict[str, dict] = {}   # task_key -> accumulated arrays
    care_frames: dict[str, list] = {}
    n_channels_ref = None

    max_ds = data.get("max_datasets_per_farm")     # None = all; else cap for quick runs

    for farm_id in data.get("farms", FARM_IDS):
        fdir = _farm_dir(root, farm_id)
        events = _read_event_info(fdir, sep)
        csvs = sorted(glob.glob(os.path.join(fdir, "datasets", "*.csv")))
        if not csvs:
            raise FileNotFoundError(f"No dataset CSVs under {fdir}/datasets")
        if max_ds:
            csvs = csvs[:int(max_ds)]

        # Resolve channel columns from the header alone, then read ONLY those
        # (+ the meta columns we need). Farm C has ~957 columns but shared mode
        # needs ~6, so `usecols` cuts the parse cost dramatically.
        header = pd.read_csv(csvs[0], sep=sep, nrows=0)
        chan_cols = _channel_columns(farm_id, _feature_columns(header), channel_mode)
        need_meta = [c for c in ("id", "train_test", "status_type_id") if c in header.columns]
        usecols = need_meta + chan_cols
        if n_channels_ref is None:
            n_channels_ref = len(chan_cols)

        # --- SINGLE streaming pass: read each CSV once, buffer the (small) needed
        #     columns, and accumulate the standardizer stats on normal TRAIN rows.
        buffered = []                                # per (dataset, split) raw slices
        sum_, sumsq_, cnt_ = None, None, 0
        for path in csvs:
            eid = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path, sep=sep, usecols=usecols)
            meta = events.get(eid, {"label": NORMAL, "start_id": None,
                                    "end_id": None, "description": None})
            fgroup = fault_group_of(meta["description"]) if farm_id == "A" else NORMAL
            for split in ("train", "prediction"):
                part = df[df["train_test"] == split]
                if part.empty:
                    continue
                raw = np.nan_to_num(part[chan_cols].to_numpy(np.float64), nan=0.0)
                status = part["status_type_id"].to_numpy(int)
                ids = part["id"].to_numpy() if "id" in part.columns else np.arange(len(part))
                buffered.append((eid, split, meta, fgroup, raw, status, ids))
                if split == "train":                 # stats: normal-status train rows only
                    keep = np.isin(status, normal_ids)
                    if keep.any():
                        xr = raw[keep]
                        sum_ = xr.sum(0) if sum_ is None else sum_ + xr.sum(0)
                        sumsq_ = (xr * xr).sum(0) if sumsq_ is None else sumsq_ + (xr * xr).sum(0)
                        cnt_ += xr.shape[0]
        if cnt_ == 0:
            raise ValueError(f"Farm {farm_id}: no normal training rows to standardize")
        mean = sum_ / cnt_
        std = np.sqrt(np.maximum(sumsq_ / cnt_ - mean ** 2, 1e-8))

        # --- window the buffered slices and label them ---
        # A window is a FAULT window only if it falls inside the event's id-range
        # (located via the `id` column, §3.1); everything else is normal. Only
        # ANOMALY datasets create/feed a diagnosis task -- normal-only datasets
        # contribute solely to standardization and to the CARE detection frames
        # (no degenerate "normal" tasks).
        for eid, split, meta, fgroup, raw, status, ids in buffered:
            valid = (_zero_run_mask(raw, zero_run_min)
                     if data.get("zero_mask", True) else np.ones(len(raw), bool))
            x = (raw - mean) / std
            X, m, centers, win_status = _make_windows(x, valid, status, window, stride)
            if len(X) == 0:
                continue

            is_anom = str(meta["label"]).strip().lower() == "anomaly"
            ew = _event_window_positions_from_ids(ids, meta["start_id"], meta["end_id"]) \
                if is_anom else None

            if split == "prediction":
                care_frames.setdefault(farm_id, []).append(CareFrame(
                    dataset_id=eid, farm=farm_id, scores_index=centers,
                    status_ids=status, is_normal_only=(not is_anom),
                    event_window=ew,
                    windows=WindowSet(X, m, np.full(len(X), -1, np.int64),
                                      np.zeros(len(X), np.int64), [farm_id] * len(X))))

            if not is_anom:
                continue                              # normal-only dataset -> no task
            # anomaly dataset: window centers inside the event id-range are faults
            local_cls = fgroup if farm_id == "A" else "anomaly"
            task_key = f"{farm_id}/{local_cls}"
            names = [local_cls if (ew is not None and ew[0] <= c <= ew[1]) else NORMAL
                     for c in centers]
            _accumulate(task_windows, task_key, farm_id, names, X, m)

    # --- finalize: register blocks, order the curriculum, build K-shot splits ---
    return _finalize_curriculum(task_windows, care_frames, label_space,
                                n_channels_ref, window, K, taskcfg, rng)


def _accumulate(store, task_key, farm, class_names, X, m):
    """Append windows + their per-window class names to a task's staging area.

    ``class_names`` holds ``NORMAL`` or this task's fault-class name per window;
    names are resolved to global column indices in :func:`_finalize_curriculum`
    once every task's block is registered.
    """
    d = store.setdefault(task_key, {"farm": farm, "classes": set(),
                                    "X": [], "m": [], "raw_y": []})
    d["classes"].update(class_names)
    d["X"].append(X)
    d["m"].append(m)
    d["raw_y"].append(list(class_names))


def _finalize_curriculum(task_windows, care_frames, label_space, n_ch, window,
                         K, taskcfg, rng):
    # Order tasks: base task (most populous) first, then the rest (§4).
    keys = list(task_windows.keys())
    sizes = {k: sum(len(x) for x in task_windows[k]["X"]) for k in keys}
    keys.sort(key=lambda k: (-sizes[k], k))  # base = largest
    if taskcfg.get("order") == "resampled":
        base, rest = keys[0], keys[1:]
        rng.shuffle(rest)
        keys = [base] + rest

    tasks: list[Task] = []
    for tid, key in enumerate(keys):
        d = task_windows[key]
        fault_classes = sorted(c for c in d["classes"] if c != NORMAL)
        if not fault_classes:            # a normal-only task still needs one channel
            fault_classes = ["anomaly"]
        start, width = label_space.register_task(key, fault_classes)

        X = np.concatenate(d["X"], 0)
        m = np.concatenate(d["m"], 0)
        raw_y = [c for chunk in d["raw_y"] for c in chunk]
        y = np.array([label_space.normal_index if c == NORMAL
                      else label_space.global_index(key, c if c in fault_classes else fault_classes[0])
                      for c in raw_y], dtype=np.int64)

        support, query, pool = _kshot_split(WindowSet(X, m, y, np.full(len(X), tid, np.int64),
                                                      [d["farm"]] * len(X)),
                                            K, label_space.normal_index, rng)
        tasks.append(Task(key=key, tid=tid, farm=d["farm"],
                          local_classes=fault_classes, support=support, query=query,
                          pool=pool, care_frames=care_frames.get(d["farm"], [])))
    return Curriculum(tasks, label_space, n_ch, window)


def _kshot_split(ws: WindowSet, K: int, normal_index: int, rng):
    """Split a task's windows into K-shot support / query / memory pool.

    Support = K labelled windows per fault class (the few-shot novelty budget).
    The remainder is halved into query (evaluation) and pool (memory candidates).
    """
    idx_by_class: dict[int, list[int]] = {}
    for i, yi in enumerate(ws.y):
        idx_by_class.setdefault(int(yi), []).append(i)

    support_idx, rest_idx = [], []
    for cls, idxs in idx_by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = min(K, len(idxs)) if cls != normal_index else min(K, len(idxs))
        support_idx += idxs[:k]
        rest_idx += idxs[k:]
    rng.shuffle(rest_idx)
    cut = len(rest_idx) // 2
    query_idx, pool_idx = rest_idx[:cut], rest_idx[cut:]
    empty = ws.subset([])
    return (ws.subset(support_idx),
            ws.subset(query_idx) if query_idx else ws.subset(support_idx),
            ws.subset(pool_idx) if pool_idx else empty)


# ===========================================================================
# Synthetic curriculum (fast; no dataset or pandas required)
# ===========================================================================
def make_synthetic_curriculum(n_tasks: int = 6, n_channels: int = 6, window: int = 32,
                              per_class: int = 60, K: int = 5, seed: int = 0,
                              binary: bool = False) -> Curriculum:
    """Generate a curriculum with clearly separable per-class signals.

    Each fault class is a distinct sinusoid + offset over the channels, so a small
    model can learn it -- exactly what M2's DoD (task-aware adaptation beats
    no-adapt) needs. Task 0 is the widest (base task); later tasks add 1-2 classes.
    """
    rng = np.random.default_rng(seed)
    ls = LabelSpace(binary=binary)
    tasks: list[Task] = []
    t = np.linspace(0, 2 * np.pi, window)

    for tid in range(n_tasks):
        farm = ["A", "B", "C"][tid % 3]
        U = 2 if (tid == 0 or farm == "A") else 1     # base + Farm-A tasks are multi-class
        local_classes = [f"f{tid}_{j}" for j in range(U)]
        ls.register_task(f"T{tid}", local_classes)

        Xs, ys = [], []
        # normal class
        for _ in range(per_class):
            Xs.append(_synth_window(rng, n_channels, t, freq=1.0, amp=0.3, offset=0.0))
            ys.append(ls.normal_index)
        # fault classes
        for j, _ in enumerate(local_classes):
            g = ls.global_index(f"T{tid}", local_classes[j])
            for _ in range(per_class):
                Xs.append(_synth_window(rng, n_channels, t,
                                        freq=2.0 + j + tid, amp=1.0 + 0.5 * j,
                                        offset=1.0 + j))
                ys.append(g)
        X = np.stack(Xs).astype(np.float32)           # [N, C, T]
        y = np.asarray(ys, np.int64)
        m = np.ones((len(X), window), np.float32)
        ws = WindowSet(X, m, y, np.full(len(X), tid, np.int64), [farm] * len(X))
        support, query, pool = _kshot_split(ws, K, ls.normal_index, rng)
        tasks.append(Task(f"T{tid}", tid, farm, local_classes, support, query, pool))

    return Curriculum(tasks, ls, n_channels, window)


def _synth_window(rng, n_channels, t, freq, amp, offset):
    phase = rng.uniform(0, np.pi, size=(n_channels, 1))
    base = amp * np.sin(freq * t[None, :] + phase) + offset
    return (base + rng.normal(0, 0.1, size=base.shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# Batching helper (shared by meta.py / memory.py)
# ---------------------------------------------------------------------------
def iterate_batches(ws: WindowSet, batch_size: int, rng, shuffle: bool = True):
    n = len(ws)
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    for s in range(0, n, batch_size):
        yield ws.subset(order[s:s + batch_size])
