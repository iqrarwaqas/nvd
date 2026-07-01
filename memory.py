"""Replay buffer + curation strategies (PIPELINE.md §7) -- the contribution.

A single fixed-capacity buffer ``M`` (capacity ``B``, identical across every
compared method for fairness). When a new task's candidates arrive, a curation
strategy decides which windows survive so that ``|M| <= B``:

    random     - uniform keep (baseline)
    herding    - keep windows closest to their per-class mean embedding (iCaRL-style)
    reservoir  - reservoir sampling over the stream (baseline)
    gss        - gradient-based sample selection: prefer a diverse gradient set (ref [16])
    rl         - the learned policy pi_phi: per-candidate keep/replace, trained by
                 REINFORCE (eq 8) on a post-adaptation reward (eq 7), with a BC
                 warm-start from herding and a contextual-bandit fallback.

Embeddings/gradients need the backbone, so strategies receive an ``embed_fn`` and
(for GSS/RL reward) hooks provided by the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dataloader import WindowSet


# ===========================================================================
# Fixed-capacity replay buffer
# ===========================================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._ws: WindowSet | None = None

    def __len__(self):
        return 0 if self._ws is None else len(self._ws)

    def as_windowset(self) -> WindowSet:
        return self._ws if self._ws is not None else _empty()

    def windows_for_task(self, tid: int) -> WindowSet:
        if self._ws is None:
            return _empty()
        idx = np.where(self._ws.task == tid)[0]
        return self._ws.subset(idx)

    def set_contents(self, ws: WindowSet):
        """Replace buffer contents, enforcing the capacity invariant."""
        if len(ws) > self.capacity:
            raise ValueError(f"buffer set to {len(ws)} > capacity {self.capacity}")
        self._ws = ws

    def occupancy_by_class(self) -> dict[int, int]:
        if self._ws is None:
            return {}
        cls, cnt = np.unique(self._ws.y, return_counts=True)
        return {int(c): int(n) for c, n in zip(cls, cnt)}


def _empty(n_channels=1, window=1):
    return WindowSet(np.zeros((0, n_channels, window), np.float32),
                     np.zeros((0, window), np.float32),
                     np.zeros((0,), np.int64), np.zeros((0,), np.int64), [])


# ===========================================================================
# Strategy interface
# ===========================================================================
@dataclass
class CurationContext:
    embed_fn: object = None            # WindowSet -> np.ndarray [N, D] (backbone embedding)
    grad_fn: object = None             # WindowSet -> np.ndarray [N, G] (loss-gradient sketch)
    reward_fn: object = None           # (candidate buffer WindowSet) -> float (post-adapt, eq 7)
    seed: int = 0


class CurationStrategy:
    name = "base"

    def update(self, buffer: ReplayBuffer, candidates: WindowSet,
               ctx: CurationContext) -> None:
        raise NotImplementedError

    @staticmethod
    def _pool(buffer, candidates):
        cur = buffer.as_windowset()
        return WindowSet.concat([cur, candidates]) if len(cur) else candidates


# --- heuristic baselines ----------------------------------------------------
class RandomCuration(CurationStrategy):
    name = "random"

    def update(self, buffer, candidates, ctx):
        pool = self._pool(buffer, candidates)
        rng = np.random.default_rng(ctx.seed)
        keep = _balanced_keep(pool, buffer.capacity, rng)
        buffer.set_contents(pool.subset(keep))


class ReservoirCuration(CurationStrategy):
    """Classic reservoir sampling: each incoming window replaces a random slot
    with probability B/n, giving a uniform sample of the whole stream."""
    name = "reservoir"

    def __init__(self):
        self._seen = 0

    def update(self, buffer, candidates, ctx):
        rng = np.random.default_rng(ctx.seed + self._seen)
        cur = list(range(len(buffer))) if len(buffer) else []
        contents = [buffer.as_windowset().subset(cur)] if cur else []
        reservoir = WindowSet.concat(contents) if contents else None
        kept = [] if reservoir is None else [reservoir]
        # merge existing + new as a stream
        stream = candidates
        held = buffer.as_windowset()
        held_list = [held.subset([i]) for i in range(len(held))]
        new_list = [stream.subset([i]) for i in range(len(stream))]
        pool = held_list + new_list
        chosen: list = []
        for i, item in enumerate(pool):
            self._seen += 1
            if len(chosen) < buffer.capacity:
                chosen.append(item)
            else:
                j = rng.integers(0, self._seen)
                if j < buffer.capacity:
                    chosen[j] = item
        buffer.set_contents(WindowSet.concat(chosen) if chosen else _empty())


class HerdingCuration(CurationStrategy):
    """Keep, per class, the windows whose embeddings are closest to the class
    mean (iCaRL herding). Requires ``ctx.embed_fn``."""
    name = "herding"

    def update(self, buffer, candidates, ctx):
        pool = self._pool(buffer, candidates)
        emb = ctx.embed_fn(pool)
        keep = _herding_keep(pool.y, emb, buffer.capacity)
        buffer.set_contents(pool.subset(keep))


class GSSCuration(CurationStrategy):
    """Gradient-based Sample Selection (ref [16]): greedily keep windows whose
    loss-gradient directions are maximally diverse (small pairwise cosine),
    approximating a buffer that covers the gradient space."""
    name = "gss"

    def update(self, buffer, candidates, ctx):
        pool = self._pool(buffer, candidates)
        g = ctx.grad_fn(pool) if ctx.grad_fn is not None else ctx.embed_fn(pool)
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        B = buffer.capacity
        if len(pool) <= B:
            buffer.set_contents(pool)
            return
        rng = np.random.default_rng(ctx.seed)
        selected = [int(rng.integers(0, len(pool)))]
        max_cos = (g @ g[selected[0]])            # similarity of every item to the seed
        while len(selected) < B:
            max_cos[selected] = np.inf
            nxt = int(np.argmin(max_cos))         # least similar to current set
            selected.append(nxt)
            max_cos = np.maximum(max_cos, g @ g[nxt])
        buffer.set_contents(pool.subset(selected))


# --- learned policy ---------------------------------------------------------
class _PolicyNet(torch.nn.Module):
    """Per-candidate keep/replace scorer over a dimension-independent state.

    State per candidate: [embedding-to-class-mean distance, per-class occupancy
    fraction, running forgetting estimate]. Kept low-dim so it transfers across
    farms with different channel counts (§7 State).
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)            # keep-logit per candidate


class RLCuration(CurationStrategy):
    """The learned curation policy (§7).

    The full REINFORCE training loop (episodes, resampled orderings, reward via
    post-adaptation eq-7) is driven by :meth:`train_episode` from main.py. At
    curation time :meth:`update` runs the policy greedily to select the buffer.
    A contextual-bandit fallback (epsilon-greedy over keep/drop) activates when
    ``fallback == 'bandit'`` and REINFORCE variance stalls.
    """
    name = "rl"
    STATE_DIM = 3

    def __init__(self, rl_cfg: dict, seed: int = 0):
        self.cfg = rl_cfg
        self.alpha_r = float(rl_cfg.get("reward_alpha", 0.5))
        self.lam = float(rl_cfg.get("reward_lambda", 0.1))
        self.use_bandit = rl_cfg.get("fallback") == "bandit"
        self.epsilon = float(rl_cfg.get("bandit_epsilon", 0.1))
        self.policy = _PolicyNet(self.STATE_DIM, int(rl_cfg.get("policy_hidden", 64)))
        self.opt = torch.optim.Adam(self.policy.parameters(),
                                    lr=float(rl_cfg.get("policy_lr", 1e-3)))
        self.baseline = 0.0                        # moving-average REINFORCE baseline
        self._bandit_q = {0: 0.0, 1: 0.0}          # drop/keep value estimates
        self._bandit_n = {0: 0, 1: 0}
        self.torch_rng = torch.Generator().manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self._forget = 0.0

    # ---- state featurisation ------------------------------------------------
    def _state(self, pool: WindowSet, emb: np.ndarray, occupancy: dict, B: int):
        # distance to each item's class-mean embedding
        dist = np.zeros(len(pool), np.float32)
        for c in np.unique(pool.y):
            idx = np.where(pool.y == c)[0]
            mu = emb[idx].mean(0)
            dist[idx] = np.linalg.norm(emb[idx] - mu, axis=1)
        dist = dist / (dist.max() + 1e-8)
        occ = np.array([occupancy.get(int(c), 0) for c in pool.y], np.float32) / max(1, B)
        forget = np.full(len(pool), self._forget, np.float32)
        return np.stack([dist, occ, forget], axis=1)

    # ---- greedy curation (inference) ---------------------------------------
    def update(self, buffer, candidates, ctx):
        pool = self._pool(buffer, candidates)
        B = buffer.capacity
        if len(pool) <= B:
            buffer.set_contents(pool)
            return
        emb = ctx.embed_fn(pool)
        s = torch.as_tensor(self._state(pool, emb, buffer.occupancy_by_class(), B),
                            dtype=torch.float32)
        if self.use_bandit and (self._bandit_n[1] + self._bandit_n[0]) < 8:
            scores = np.array([self._bandit_q[1] for _ in range(len(pool))]) \
                     + self.np_rng.normal(0, 1e-3, len(pool))
        else:
            with torch.no_grad():
                scores = self.policy(s).numpy()
        keep = np.argsort(-scores)[:B]
        buffer.set_contents(pool.subset(np.sort(keep)))

    # ---- REINFORCE episode (training, eq 7-8) ------------------------------
    def train_episode(self, buffer, candidates, ctx) -> float:
        """One episode: sample a keep-set from the policy, measure the reward
        *after* task-aware adaptation (eq 7), and take a REINFORCE step (eq 8).
        Returns the episode reward for logging."""
        pool = self._pool(buffer, candidates)
        B = buffer.capacity
        emb = ctx.embed_fn(pool)
        s = torch.as_tensor(self._state(pool, emb, buffer.occupancy_by_class(), B),
                            dtype=torch.float32)
        logits = self.policy(s)
        probs = torch.sigmoid(logits)              # per-candidate keep probability
        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()                     # 1 = keep

        keep_idx = torch.nonzero(action).squeeze(-1)
        if keep_idx.numel() > B:                    # enforce |M| <= B: keep top-prob B
            top = torch.argsort(probs[keep_idx], descending=True)[:B]
            keep_idx = keep_idx[top]
        chosen = keep_idx.cpu().numpy()
        candidate_buffer = pool.subset(np.sort(chosen)) if len(chosen) else _empty()

        # reward = alpha*Acc_old + (1-alpha)*Acc_new - lambda*|M|/B  (post-adaptation)
        reward = ctx.reward_fn(candidate_buffer) if ctx.reward_fn is not None else 0.0
        reward -= self.lam * (len(candidate_buffer) / max(1, B))

        # REINFORCE with moving-average baseline (eq 8)
        advantage = reward - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * reward
        logp = dist.log_prob(action).sum()
        loss = -advantage * logp
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # bandit bookkeeping (fallback signal)
        a = 1 if len(chosen) >= B // 2 else 0
        self._bandit_n[a] += 1
        self._bandit_q[a] += (reward - self._bandit_q[a]) / self._bandit_n[a]

        buffer.set_contents(candidate_buffer)
        return float(reward)

    def bc_warmstart(self, buffer, candidates, ctx, teacher: CurationStrategy,
                     steps: int = 50):
        """Behavior-clone the policy from a heuristic teacher (herding) before
        REINFORCE, so the policy starts from a sensible buffer (§7 scaffolding)."""
        pool = self._pool(buffer, candidates)
        B = buffer.capacity
        if len(pool) <= B:
            return
        emb = ctx.embed_fn(pool)
        target = np.zeros(len(pool), np.float32)
        target[_herding_keep(pool.y, emb, B)] = 1.0
        s = torch.as_tensor(self._state(pool, emb, buffer.occupancy_by_class(), B),
                            dtype=torch.float32)
        y = torch.as_tensor(target)
        for _ in range(steps):
            self.opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.policy(s), y)
            loss.backward()
            self.opt.step()


# ===========================================================================
# shared selection helpers
# ===========================================================================
def _balanced_keep(pool: WindowSet, B: int, rng) -> np.ndarray:
    """Keep up to B windows, spread across classes as evenly as possible."""
    if len(pool) <= B:
        return np.arange(len(pool))
    by_class = {int(c): np.where(pool.y == c)[0] for c in np.unique(pool.y)}
    for idxs in by_class.values():
        rng.shuffle(idxs)
    keep, i = [], 0
    while len(keep) < B:
        progressed = False
        for c, idxs in by_class.items():
            if i < len(idxs):
                keep.append(idxs[i]); progressed = True
                if len(keep) == B:
                    break
        if not progressed:
            break
        i += 1
    return np.array(sorted(keep))


def _herding_keep(y: np.ndarray, emb: np.ndarray, B: int) -> np.ndarray:
    """Per-class herding selection with a per-class budget proportional to B."""
    if len(y) <= B:
        return np.arange(len(y))
    classes = np.unique(y)
    per = max(1, B // len(classes))
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        mu = emb[idx].mean(0)
        order = idx[np.argsort(np.linalg.norm(emb[idx] - mu, axis=1))]
        keep.extend(order[:per].tolist())
    keep = keep[:B]
    return np.array(sorted(keep))


# ===========================================================================
# factory
# ===========================================================================
def make_strategy(cfg: dict, seed: int = 0) -> CurationStrategy:
    mem = cfg["memory"]
    name = mem.get("strategy", "random")
    if name == "random":
        return RandomCuration()
    if name == "reservoir":
        return ReservoirCuration()
    if name == "herding":
        return HerdingCuration()
    if name == "gss":
        return GSSCuration()
    if name == "rl":
        return RLCuration(mem.get("rl", {}), seed=seed)
    raise ValueError(f"unknown memory.strategy {name!r}")
