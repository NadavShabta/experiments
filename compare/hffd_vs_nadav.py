"""Compare baseline HFFD with Nadav's optimised HFFD.

Outputs three artefacts in *compare/results/hffd_vs_nadav/*:

1. **runtime.png** – bigger line plot (not bars) of algorithm wall‑times.
2. **diagnostic.png** – heat‑map + bar comparisons (unchanged layout).
3. **CSV files** –
   * `cost_matrix.csv`   — IDO cost matrix (agents × items)
   * `thresholds.csv`    — per‑agent τₐ
   * `alloc_hffd.csv`    — allocation lists for baseline HFFD
   * `alloc_fast.csv`    — allocation lists for Nadav's fast HFFD

All paths printed at the end are absolute‑or‑relative depending on the
current working directory, so you always know where to look.
"""
from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairpyx import Instance, divide
from fairpyx.algorithms.hffd import hffd
from algos import hffd_nadav  # ← Nadav's optimised version

# ───────────── tweakable parameters ────────────────────────────────────
TIME_LIMIT = 60          # seconds before treating as timeout
BIG_ITEMS = 5_000        # chores in the *runtime* benchmark
N_AGENTS = 30            # number of agents (rows)
ALPHA = 0.45             # baseline slack factor for thresholds
NOISE = 0.25             # multiplicative noise for per‑agent costs
RNG = np.random.default_rng(0)
ALGS = {"hffd": hffd, "hffd_nadav": hffd_nadav}
# ───────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------------------
# Instance + threshold generator
# ----------------------------------------------------------------------

def huge_ido(items: int) -> Tuple[Instance, Dict[int, int]]:
    """Return an *IDO* cost matrix (int) and varied integer thresholds."""
    base = RNG.integers(10, 101, size=items)
    base.sort(); base = base[::-1].astype(int)  # common ranking

    scales = RNG.uniform(1 - NOISE, 1 + NOISE, size=(N_AGENTS, 1))
    costs = (scales * base).round().astype(int)

    slack = RNG.uniform(ALPHA * 0.8, ALPHA * 1.2, size=N_AGENTS)
    thr = {a: int(slack[a] * costs[a].sum() / N_AGENTS) for a in range(N_AGENTS)}
    return Instance(valuations=costs), thr


# ----------------------------------------------------------------------
# Runtime comparison helpers
# ----------------------------------------------------------------------

def run_once(algo_name: str, n_items: int) -> float | None:
    """Run *algo_name* on a fresh instance; return wall‑time (s) or None on crash."""
    inst, thr = huge_ido(n_items)
    t0 = time.perf_counter()
    try:
        divide(ALGS[algo_name], inst, thresholds=thr)
    except Exception:
        return None
    return time.perf_counter() - t0


# ----------------------------------------------------------------------
# 1) Measure runtimes under hard timeout
# ----------------------------------------------------------------------
result_dir = Path("compare/results/hffd_vs_nadav")
result_dir.mkdir(parents=True, exist_ok=True)

runtimes: Dict[str, float | None] = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
    for name in ALGS:
        fut = pool.submit(run_once, name, BIG_ITEMS)
        try:
            rt = fut.result(timeout=TIME_LIMIT)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            rt = TIME_LIMIT
        runtimes[name] = rt
        status = "timeout" if rt == TIME_LIMIT else f"{rt:6.2f} s"
        print(f"{name:10} → {status}")

# ───────── line plot of runtimes ─────────────────────────────────────--
fig_rt, ax_rt = plt.subplots(figsize=(8, 4))
labels = list(runtimes.keys())
values = [TIME_LIMIT if v is None else v for v in runtimes.values()]
ax_rt.plot(labels, values, marker="o", lw=2)
ax_rt.set_ylabel("runtime (s)")
ax_rt.set_title(f"Runtime on huge IDO instance (m={BIG_ITEMS}, n={N_AGENTS})")
ax_rt.axhline(TIME_LIMIT, color="red", ls="--", lw=0.8)
for x, y in zip(labels, values):
    ax_rt.text(x, y + 1, "≥60" if y >= TIME_LIMIT else f"{y:.1f}", ha="center")
fig_rt.tight_layout()
rt_png = result_dir / "runtime.png"
fig_rt.savefig(rt_png, dpi=120, bbox_inches="tight")
plt.close(fig_rt)

try:
    rel_rt = rt_png.relative_to(Path.cwd())
except ValueError:
    rel_rt = rt_png
print(f"✓  saved runtime plot → {rel_rt}")

# ----------------------------------------------------------------------
# 1b) Runtime vs problem size (new experiment)
# ----------------------------------------------------------------------

sizes = list(range(200, BIG_ITEMS + 1, 200))  # e.g., 200, 400, ..., 5000
results = {name: [] for name in ALGS}

print("\nMeasuring runtime vs problem size...")

for n_items in sizes:
    print(f"  Problem size: {N_AGENTS} agents x {n_items} items", flush=True)
    for name in ALGS:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(run_once, name, n_items)
            try:
                rt = fut.result(timeout=TIME_LIMIT)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                rt = TIME_LIMIT
        results[name].append(rt)
        print(f"    {name:10} → {'timeout' if rt == TIME_LIMIT else f'{rt:6.2f} s'}", flush=True)

# Plot runtime vs problem size
fig_sz, ax_sz = plt.subplots(figsize=(8, 5))
for name, times in results.items():
    ax_sz.plot([N_AGENTS * n for n in sizes], times, marker="o", label=name)
ax_sz.set_xlabel("agents × items")
ax_sz.set_ylabel("runtime (s)")
ax_sz.set_title(f"Runtime vs problem size (n_agents={N_AGENTS})")
ax_sz.axhline(TIME_LIMIT, color="red", ls="--", lw=0.8, label="timeout")
ax_sz.legend()
fig_sz.tight_layout()
rt_size_png = result_dir / "runtime_vs_size.png"
fig_sz.savefig(rt_size_png, dpi=120, bbox_inches="tight")
plt.close(fig_sz)

try:
    rel_sz = rt_size_png.relative_to(Path.cwd())
except ValueError:
    rel_sz = rt_size_png
print(f"✓  saved runtime-vs-size plot → {rel_sz}")

# ----------------------------------------------------------------------
# 2) Build diagnostic figure with costs + allocations
# ----------------------------------------------------------------------
DIAG_ITEMS = min(BIG_ITEMS, 2_000)  # readability for heat‑map
inst_diag, thr_diag = huge_ido(DIAG_ITEMS)

# Save cost matrix & thresholds as CSV
pd.DataFrame(inst_diag._valuations).to_csv(result_dir / "cost_matrix.csv", index_label="agent")
pd.Series(thr_diag).sort_index().to_csv(result_dir / "thresholds.csv", header=["tau"], index_label="agent")

# helper to run algorithm, keep allocation & totals --------------------

def allocation_and_totals(algo_fn):
    inst_copy = Instance(valuations=inst_diag._valuations.copy())
    allocation = divide(algo_fn, inst_copy, thresholds=thr_diag)
    totals = {
        a: sum(inst_copy.agent_item_value(a, i) for i in items)
        for a, items in allocation.items()
    }
    return allocation, totals

print("Running algorithms for diagnostic figure …")

alloc_baseline, tot_baseline = allocation_and_totals(hffd)
alloc_fast, tot_fast = allocation_and_totals(hffd_nadav)

# Save allocations to CSV (agent → semicolon‑separated item list) -------
for tag, alloc in [("alloc_hffd.csv", alloc_baseline), ("alloc_fast.csv", alloc_fast)]:
    pd.Series({a: ";".join(map(str, items)) for a, items in alloc.items()}) \
        .sort_index().to_csv(result_dir / tag, header=["items"], index_label="agent")

# ----------------------------------------------------------------------
# Create composite diagnostic figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

# (a) heat‑map -----------------------------------------------------------
ax0 = fig.add_subplot(gs[0, :])
img = ax0.imshow(inst_diag._valuations, aspect="auto")
ax0.set_title("IDO cost matrix (agents × items)")
ax0.set_xlabel("item index")
ax0.set_ylabel("agent index")
fig.colorbar(img, ax=ax0, orientation="vertical", fraction=0.025, pad=0.01)

agents = np.arange(N_AGENTS)
thr_vals = [thr_diag[a] for a in agents]

# (b) baseline HFFD bar comparison --------------------------------------
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(agents - 0.2, thr_vals, width=0.4, label="τₐ (threshold)")
ax1.bar(agents + 0.2, [tot_baseline.get(a, 0) for a in agents], width=0.4, label="HFFD cost")
ax1.set_title("Baseline HFFD")
ax1.set_xlabel("agent")
ax1.set_ylabel("cost")
ax1.legend()

# (c) fast HFFD bar comparison ------------------------------------------
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(agents - 0.2, thr_vals, width=0.4, label="τₐ (threshold)")
ax2.bar(agents + 0.2, [tot_fast.get(a, 0) for a in agents], width=0.4, label="Fast cost")
ax2.set_title("Nadav's fast HFFD")
ax2.set_xlabel("agent")
ax2.set_ylabel("cost")
ax2.legend()

fig.tight_layout()

diag_png = result_dir / "diagnostic.png"
fig.savefig(diag_png, dpi=150)
plt.close(fig)

try:
    rel_diag = diag_png.relative_to(Path.cwd())
except ValueError:
    rel_diag = diag_png
print(f"✓  saved diagnostic figure → {rel_diag}")

# Show interactively when backend permits ------------------------------
if plt.get_backend().lower() not in {"agg", "cairo", "pdf", "pgf"}:
    plt.show()
