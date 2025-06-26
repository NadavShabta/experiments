"""
Benchmark *one* very large identical-order instance with a hard 60-second
wall-clock limit.

Why this script?
----------------
For huge inputs it can be tedious to tweak instance sizes by hand until
an algorithm hits the 60 s mark.  Instead we:

1.  **Fix a single problem size** – here `m = BIG_ITEMS` chores for
    `n = N_AGENTS` agents – and time each algorithm once.
2.  **Abort after 60 s** (hard wall clock, not CPU) by running the
    algorithm in a separate *process* that we kill if it overstays.
3.  Record
       • the actual runtime if it finished within ≤ 60 s;
       • the cap (“≥ 60 s”) otherwise.
4.  Produce a tiny bar-chart for a quick visual impression.

Definitions
-----------
* **Agents (n)** – decision -makers who each receive a bundle of chores.
  Here we keep `N_AGENTS = 20`.
* **Items / chores (m)** – indivisible tasks to be divided.
  We test one “huge” instance with `BIG_ITEMS` items.
* **Identical-order costs (IDO)** – every agent ranks the items in the
  *same* descending order of dis-utility, but absolute costs differ
  slightly across agents.

  Implementation:
  ``base``               – a single descending vector of random
                           integers 10 … 100  (length m)\
  ``noise ∈ [−NOISE,+NOISE]`` – zero-mean multiplicative jitter per entry\
  ``costs[a, i]``        – IDO cost matrix (shape n × m)
* **Thresholds τₐ** – an upper bound on the *total* cost an agent a may
  carry.  Given a fairness coefficient `ALPHA` we set
  `τₐ = ALPHA · (Σ_i costs[a,i] / n)`.
  With `ALPHA = 1.0` each agent must not exceed its *proportional*
  share of the total cost it perceives.

The algorithms under test
-------------------------
* **`hffd`**      – original implementation from *Huang & Segal-Halevi (2024)*
* **`hffd_fast`** – our cache-friendly vectorised re-write.

Both live inside *fairpyx* (original) or *improved_hffd.py* (fast).

"""
from __future__ import annotations
from pathlib import Path
import concurrent.futures, time, numpy as np, matplotlib.pyplot as plt
from fairpyx import Instance, divide
from fairpyx.algorithms.hffd import hffd
from improved_hffd import hffd_fast               # ← your optimised version

# ───────────── parameters you may tweak ────────────────────────────────
TIME_LIMIT  = 60            # hard wall-clock timeout (s)
BIG_ITEMS   = 11_000       # chores in the single huge instance
N_AGENTS    = 20
ALPHA, NOISE = 1.0, .30
RNG         = np.random.default_rng(0)
ALGS        = {"hffd": hffd, "hffd_fast": hffd_fast}
# ───────────────────────────────────────────────────────────────────────

def huge_ido(items: int) -> tuple[Instance, dict[int,float]]:
    """
    Build a **huge identical-descending-order (IDO) instance** and per–agent
    thresholds  τₐ.

    Parameters
    ----------
    items : int
        Number of chores (columns).

    Returns
    -------
    (Instance, dict[int, float])
        • `Instance` — the valuation matrix  *costs*  where
          - each row  = one agent (there are `N_AGENTS` rows)
          - each column = one chore, ordered identically for all agents
          Every entry is a *dis-utility* (higher = worse).

        • `thr` — a dict mapping every agent `a` to its personal threshold

                 τₐ = α · ( ∑ᵢ costₐᵢ ) / n

            where
            • α (`ALPHA`) is a global scaling factor (α = 1 ⇒ “average share”)
            • n  (`N_AGENTS`) is the total number of agents

            In words: each agent’s threshold is its **own total cost**, scaled
            down by the number of agents (so it is roughly a “fair share”),
            then optionally stretched up/down by α.

    How the data are generated
    --------------------------
    1. `base`
       One common descending vector of chore costs (largest → smallest).

    2. `costs`
       For every agent we multiply the common `base` by **(1 ± NOISE)** with a
       fresh uniform noise in ±`NOISE`% for each entry.
       This preserves the identical order of columns but makes each row
       individual.

    3. `thr`
       For each agent `a`:
       • take the row–sum `costs[a].sum()` (that agent’s total burden if they
         did everything),
       • divide by `N_AGENTS` to get a proportional share,
       • multiply by `ALPHA` (α).

    The function finally returns the ready-to-use `Instance`
    plus the per-agent threshold dictionary.
    """
    base = RNG.integers(10, 101, size=items)
    base.sort(); base = base[::-1]
    costs = base * (1 + (RNG.random((N_AGENTS, items))*2 - 1)*NOISE)
    inst  = Instance(valuations=costs)
    thr   = {a: ALPHA*costs[a].sum()/N_AGENTS for a in range(N_AGENTS)}
    return inst, thr

def run_once(algo_name: str, n_items: int) -> float|None:
    """Return runtime (s); None if algorithm crashed."""
    inst, thr = huge_ido(n_items)
    t0 = time.perf_counter()
    try:
        divide(ALGS[algo_name], inst, thresholds=thr)
    except Exception:
        return None
    return time.perf_counter() - t0

# ───────────── run each algorithm with hard timeout ───────────────────
Path("../compare/results").mkdir(exist_ok=True)
runtimes : dict[str,float|None] = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
    for name in ALGS:
        fut = pool.submit(run_once, name, BIG_ITEMS)
        try:
            rt = fut.result(timeout=TIME_LIMIT)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            rt = TIME_LIMIT                       # treat as “took ≥ 60 s”
        runtimes[name] = rt
        print(f"{name:10} → "
              f"{'timeout' if rt==TIME_LIMIT else f'{rt:7.2f} s'}")

# ───────────── plot quick bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(5,3))
bars = ax.bar(runtimes.keys(),
              [TIME_LIMIT if v is None else v for v in runtimes.values()],
              color=['tab:blue','tab:orange'])
ax.set_ylabel("runtime (s)")
ax.set_title(f"Single huge instance (m={BIG_ITEMS}, n={N_AGENTS})")
ax.set_ylim(0, TIME_LIMIT*1.1)
ax.axhline(TIME_LIMIT, color='red', ls='--', lw=.8)
ax.bar_label(bars, fmt=lambda v: "≥60 s" if v>=TIME_LIMIT else f"{v:.1f}s",
             padding=3)
fig.tight_layout()
fig.savefig("results/result_improve/huge_instance_timeout2.png")
print("✓  results/result_improve/huge_instance_timeout2.png")
