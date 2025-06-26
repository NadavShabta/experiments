#!/usr/bin/env python3
"""
Benchmark the original HFFD implementation against Nadav’s variant
(hffd_nadav) on identical-descending-order (IDO) chore-allocation
instances.

Changes vs the previous version
-------------------------------
• Much larger parameter grid   (up to 80 agents, 12 items/agent, extra
  noise ratios, extra α values).
• Stricter wall-clock limit  (30 s per run).
• Extra statistics written to CSV:
      num_items, instance_size, mean_tau, max_tau
• Still benchmarks ONLY the two algorithms of interest.
"""

from pathlib import Path
import logging
import time
import numpy as np
import experiments_csv
from fairpyx import Instance, divide
from fairpyx.algorithms.hffd import hffd          # ← original
from algos import hffd_nadav                      # ← Nadav’s variant

# ----------------------------------------------------------------------
# 1.  Instance generator – costs with IDENTICAL ORDER
# ----------------------------------------------------------------------
def random_ido_instance(
    num_of_agents: int,
    num_of_items: int,
    value_noise_ratio: float,
    alpha: float,
    rng: np.random.Generator,
):
    """
    Returns  (instance, thresholds, stats_dict).

    * Start with a single descending vector of base-costs c[0..m-1].
    * Each agent gets  c ± noise·c  (ranking is identical).
    * Threshold  τ_a = alpha · (sum(costs_a) / n).
    """
    # 1. common descending base costs
    base = rng.integers(10, 101, size=num_of_items)
    base.sort(); base = base[::-1]                # largest → smallest

    # 2. per-agent noisy copy (keeps order)
    noise  = (rng.random((num_of_agents, num_of_items))*2 - 1) * value_noise_ratio
    costs  = base * (1 + noise)

    inst = Instance(valuations=costs)             # additive dis-utilities

    # 3. per-agent thresholds τ_a
    thresholds = {
        a: alpha * costs[a].sum() / num_of_agents
        for a in range(num_of_agents)
    }
    stats = {
        "mean_tau": np.mean(list(thresholds.values())),
        "max_tau" : np.max (list(thresholds.values())),
    }
    return inst, thresholds, stats


# ----------------------------------------------------------------------
# 2.  One experiment run = one (instance, algorithm, …)
# ----------------------------------------------------------------------
def run_single(
    num_of_agents: int,
    items_per_agent: int,
    value_noise_ratio: float,
    alpha: float,
    algorithm,
    seed: int,
):
    rng  = np.random.default_rng(seed)
    num_items = num_of_agents * items_per_agent
    inst, tau, tau_stats = random_ido_instance(
        num_of_agents      = num_of_agents,
        num_of_items       = num_items,
        value_noise_ratio  = value_noise_ratio,
        alpha              = alpha,
        rng                = rng,
    )

    t0 = time.perf_counter()
    try:
        allocation = divide(algorithm, inst, thresholds=tau)
        runtime = time.perf_counter() - t0

        # overload > 0  means someone exceeded τ_a
        max_overload = max(
            inst.agent_bundle_value(a, items) - tau[a]
            for a, items in allocation.items()
        )
        success = int(max_overload <= 0)
    except Exception:
        runtime, success, max_overload = time.perf_counter() - t0, 0, float("inf")

    return {
        "success"       : success,
        "max_overload"  : max_overload,
        "num_items"     : num_items,
        "instance_size" : num_items * num_of_agents,
        "mean_tau"      : tau_stats["mean_tau"],
        "max_tau"       : tau_stats["max_tau"],
        "runtime"       : runtime,          # experiments_csv stores it too; kept for convenience
    }


# ----------------------------------------------------------------------
# 3.  Experiment configuration
# ----------------------------------------------------------------------
algorithms_to_check = [hffd, hffd_nadav]

input_ranges = {
    "num_of_agents"     : [10, 20, 40, 80],
    "items_per_agent"   : [3, 6, 12],        # total items = agents × this
    "value_noise_ratio" : [0.0, 0.1, 0.3, 0.6],
    "alpha"             : [0.6, 0.8, 1.0, 1.2],
    "algorithm"         : algorithms_to_check,
    "seed"              : range(3),          # 3 random instances per row
}

# ----------------------------------------------------------------------
# 4.  Run (with automatic skip & time-limit)
# ----------------------------------------------------------------------
Path("results").mkdir(exist_ok=True)
ex = experiments_csv.Experiment("results", "hffd_comparison.csv",
                                backup_folder="results/backups")
experiments_csv.logger.setLevel(logging.INFO)

TIME_LIMIT = 30       # seconds per run
ex.run_with_time_limit(run_single, input_ranges, time_limit=TIME_LIMIT)

print("\n✓  Finished!  See  results/hffd_comparison.csv")

def main() -> None:
    Path("results").mkdir(exist_ok=True)
    ex = experiments_csv.Experiment(
        "results", "hffd_comparison.csv", backup_folder="results/backups"
    )
    experiments_csv.logger.setLevel(logging.INFO)

    TIME_LIMIT = 60      # seconds per run
    ex.run_with_time_limit(run_single, input_ranges, time_limit=TIME_LIMIT)

    print("\n✓  Finished!  See  results/hffd_comparison.csv")


# ----------------------------------------------------------------------
# 5.  Only execute when the file is run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
