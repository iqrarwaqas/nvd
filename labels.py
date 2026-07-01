"""Label definition + variable-U task partition (PIPELINE.md §3.3).

This is the *swappable layer* where the paper's "task owns U fault channels" is
realised honestly against CARE's label reality:

  * Farm A (EDP, onshore) has real component fault types in its logbook -> multiple
    fault channels (U_A = number of distinct fault groups present).
  * Farms B, C (offshore, anonymised) expose a single `anomaly` label -> U = 1 each.

The global class space is the union:

    {normal} u block(task_1) u block(task_2) u ... u block(task_S)

where `block(t)` is a contiguous range of `U_t` columns. Column 0 is always the
shared `normal` class. A `binary` mode collapses every fault to a single `anomaly`
class (used for the detection ablation and the native CARE score).

Nothing here hardcodes class counts: `U_t` is discovered from the data at build
time (see :func:`fault_group_of`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

FARM_IDS = ["A", "B", "C"]

# Meta (non-sensor) columns present in every CARE dataset CSV.
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]

# Default normal-operation status ids (CARE README: 0=Normal, 2=Idling).
DEFAULT_NORMAL_STATUS_IDS = (0, 2)

NORMAL = "normal"
ANOMALY = "anomaly"

# ---------------------------------------------------------------------------
# Farm-A fault-group taxonomy.
#
# CARE's Farm A logbook stores free-text component descriptions. We bucket them
# into a small fixed set of component groups by keyword. The *present* groups are
# discovered at build time; groups with no events simply never enter the label
# space (so `sum(U_t)` always matches the discovered inventory -- see selftest).
# ---------------------------------------------------------------------------
FAULT_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "generator":   ("generator", "gen ", "stator", "rotor winding"),
    "gearbox":     ("gearbox", "gear ", "gear box"),
    "transformer": ("transformer", "trafo"),
    "hydraulic":   ("hydraulic", "hydraulics", "pitch hydraulic"),
    "bearing":     ("bearing", "main bearing"),
    "converter":   ("converter", "inverter", "igbt"),
    "yaw":         ("yaw",),
    "pitch":       ("pitch",),
    "cooling":     ("cooling", "cooler", "fan", "radiator"),
}
FAULT_GROUP_OTHER = "other"


def fault_group_of(description: str | None) -> str:
    """Map a free-text fault description to a canonical component group.

    Unknown / empty descriptions fall back to :data:`FAULT_GROUP_OTHER` so no
    anomaly event is ever silently dropped.
    """
    if not description:
        return FAULT_GROUP_OTHER
    text = str(description).strip().lower()
    if not text:
        return FAULT_GROUP_OTHER
    for group, keys in FAULT_GROUP_KEYWORDS.items():
        if any(k in text for k in keys):
            return group
    return FAULT_GROUP_OTHER


# ---------------------------------------------------------------------------
# Label space (grown incrementally as tasks arrive).
# ---------------------------------------------------------------------------
@dataclass
class LabelSpace:
    """Maps task blocks to contiguous global class indices.

    Index 0 is reserved for the shared ``normal`` class. Each registered task
    ``key`` owns a contiguous block ``[start, start+width)`` of fault columns.
    """

    binary: bool = False
    normal_index: int = 0
    # task_key -> (block_start, width, [local class names])
    blocks: dict[str, tuple[int, int, list[str]]] = field(default_factory=dict)
    _next: int = 1  # next free global column (0 is normal)

    def register_task(self, task_key: str, local_classes: list[str]) -> tuple[int, int]:
        """Reserve a block of columns for ``task_key``; returns (start, width)."""
        if task_key in self.blocks:
            start, width, _ = self.blocks[task_key]
            return start, width
        width = 1 if self.binary else max(1, len(local_classes))
        names = [ANOMALY] if self.binary else list(local_classes)
        start = self._next
        self.blocks[task_key] = (start, width, names)
        self._next += width
        return start, width

    @property
    def total_classes(self) -> int:
        """Number of active global columns = 1 (normal) + sum of block widths."""
        return self._next

    def global_index(self, task_key: str, local_class: str) -> int:
        """Global column index for a (task, local fault-class) pair."""
        start, width, names = self.blocks[task_key]
        if self.binary:
            return start
        return start + names.index(local_class)

    def task_columns(self, task_key: str) -> list[int]:
        """Columns visible for task ``t`` under eq-(6) masking: {normal} u block(t)."""
        start, width, _ = self.blocks[task_key]
        return [self.normal_index] + list(range(start, start + width))

    def block(self, task_key: str) -> tuple[int, int]:
        start, width, _ = self.blocks[task_key]
        return start, width

    def class_name(self, global_index: int) -> str:
        if global_index == self.normal_index:
            return NORMAL
        for key, (start, width, names) in self.blocks.items():
            if start <= global_index < start + width:
                return f"{key}:{names[global_index - start]}"
        raise IndexError(f"global index {global_index} not in label space")


def window_label(status_ids, event_label: str, fault_group: str,
                 normal_status_ids=DEFAULT_NORMAL_STATUS_IDS) -> str:
    """Return the label for one window (PIPELINE.md §3.3).

    A window is ``normal`` unless it lies inside an anomaly event *and* carries a
    non-normal status; in that case it takes the event's fault group (Farm A) or
    the generic ``anomaly`` group (farms B, C).

    Parameters
    ----------
    status_ids : iterable of int
        ``status_type_id`` values covering the window's timesteps.
    event_label : str
        Dataset-level event label ("anomaly" or "normal").
    fault_group : str
        Component group for the event (from :func:`fault_group_of`); ignored when
        ``event_label`` is normal.
    """
    normal = set(normal_status_ids)
    all_normal_status = all(int(s) in normal for s in status_ids)
    if str(event_label).strip().lower() != ANOMALY or all_normal_status:
        return NORMAL
    return fault_group
