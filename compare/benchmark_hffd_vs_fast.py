"""
Benchmark *only* runtime of the original `hffd` vs the optimised `hffd_fast`
on increasingly large IDO instances, and plot the result.

Outputs:  results/hffd_speedup.png
"""

from pathlib import Path
import time, logging, numpy as np
import matplotlib.pyplot as plt
from fairpyx import Instance, divide
from fairpyx.algorithms.hffd import hffd             # original
from improve.improved_hffd import hffd_fast                 # your sped-up version


logging.getLogger("fairpyx").setLevel(logging.CRITICAL)   # silence info

# --------------------------------------------------------------------------
# helper – generate an Identical-Descending-Order (IDO) instance
# --------------------------------------------------------------------------
def ido_instance(num_agents: int, items_per_agent: int, rng):
    m = num_agents * items_per_agent
    base = rng.integers(10, 101, size=m)
    base.sort(); base = base[::-1]
    costs = np.tile(base, (num_agents, 1))
    inst = Instance(valuations=costs)
    # proportional thresholds (sum(c_i)/n)
    thr = {a: costs[a].sum()/num_agents for a in range(num_agents)}
    return inst, thr

# --------------------------------------------------------------------------
# experiment grid
# --------------------------------------------------------------------------
AGENTS_LIST        = [10, 20, 40, 80]
ITEMS_PER_AGENT    = 6
REPEATS_PER_SIZE   = 3
rng = np.random.default_rng(0)

results = []   # (instance_size, algo_name, runtime)

for n in AGENTS_LIST:
    for _rep in range(REPEATS_PER_SIZE):
        inst, thr = ido_instance(n, ITEMS_PER_AGENT, rng)
        for name, algo in [("hffd", hffd), ("hffd_fast", hffd_fast)]:
            t0 = time.perf_counter()
            divide(algo, inst, thresholds=thr)
            dt = time.perf_counter() - t0
            results.append((n*ITEMS_PER_AGENT, name, dt))

# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
Path("results").mkdir(exist_ok=True)
plt.figure(figsize=(6,4))
for algo in ["hffd", "hffd_fast"]:
    xs  = [sz for sz,a,_ in results if a==algo]
    ys  = [t  for _,a,t in results if a==algo]
    means = {}
    for sz, t in zip(xs, ys):
        means.setdefault(sz, []).append(t)
    sz_sorted = sorted(means)
    mean_t    = [np.mean(means[sz]) for sz in sz_sorted]
    plt.plot(sz_sorted, mean_t, marker="o", label=algo)

plt.xlabel("instance size  (#items)")
plt.ylabel("runtime  (s)")
plt.title("HFFD runtime vs optimised HFFD_fast")
plt.legend()
plt.tight_layout()
out = Path("results/hffd_speedup.png")
plt.savefig(out, dpi=150)
print(f"✓ figure written → {out}")