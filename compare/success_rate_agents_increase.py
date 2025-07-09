import fairpyx
import numpy as np
import time
import logging
import io
import matplotlib.pyplot as plt
from typing import Any, Mapping, Sequence
# Import the 'divide' function from fairpyx.adaptors
from fairpyx.adaptors import divide

# Suppress fairpyx info logs for cleaner output during testing
logging.getLogger("fairpyx").setLevel(logging.WARNING)


# --- HFFD Algorithm Implementations ---

# Original hffd algorithm
# =========================================================================
def hffd(
        builder,
        *,
        thresholds: Mapping[Any, float] | Sequence[float],
        universal_order: Sequence[Any] | None = None,
) -> None:
    inst = builder.instance
    agents = list(builder.remaining_agents())
    items_0 = list(builder.remaining_items())

    if not isinstance(thresholds, Mapping):
        if len(thresholds) != len(agents):
            raise ValueError("threshold list length must equal #agents")
        thresholds = dict(zip(agents, thresholds))
    if set(thresholds) != set(agents):
        raise ValueError("thresholds must specify every agent exactly once")

    tau = {a: float(thresholds[a]) for a in agents}

    order = list(universal_order) if universal_order is not None else items_0
    if set(order) != set(items_0):
        raise ValueError("universal_order must match remaining items exactly")

    cost = {(a, i): inst.agent_item_value(a, i) for a in agents for i in order}

    remaining_items = set(order)
    agents_left = agents.copy()
    bundle_no = 0

    while agents_left and remaining_items:
        bundle: list[Any] = []

        for i in order:
            if i not in remaining_items:
                continue
            if any(sum(cost[(a, j)] for j in bundle) + cost[(a, i)] <= tau[a]
                   for a in agents_left):
                bundle.append(i)

        if not bundle:
            break

        chosen = next(a for a in agents_left
                      if sum(cost[(a, j)] for j in bundle) <= tau[a])

        bundle_cost = sum(cost[(chosen, j)] for j in bundle)
        bundle_no += 1
        logging.info("Bundle #%d → agent %s : %s  (cost %.0f)",
                     bundle_no, chosen, bundle, bundle_cost)

        for j in bundle:
            builder.give(chosen, j)
            remaining_items.remove(j)

        tau[chosen] -= bundle_cost
        agents_left.remove(chosen)

    if remaining_items:
        logging.info("Unallocated chores: %s", sorted(remaining_items))
    else:
        logging.info("All chores allocated.")


# Optimized hffd_fast algorithm
# =========================================================================
def hffd_fast(
        builder: "BuilderLike",  # "BuilderLike" is a forward reference, which is fine
        *,
        thresholds: Mapping[Any, float] | Sequence[float],
        universal_order: Sequence[Any] | None = None,
) -> set:
    inst = builder.instance
    agents = list(builder.remaining_agents())
    items_0 = list(builder.remaining_items())

    if not isinstance(thresholds, Mapping):
        thresholds = dict(zip(agents, thresholds))
    tau = np.array([float(thresholds[a]) for a in agents], dtype=np.float64)

    order = list(universal_order) if universal_order is not None else items_0
    if set(order) != set(items_0):
        raise ValueError("universal_order must equal remaining items")
    item_pos = {i: k for k, i in enumerate(order)}

    # Construct the costs matrix explicitly using agent_item_value
    # This handles cases where inst._valuations might be a dict and not directly convertible to a NumPy array.
    costs = np.array(
        [[inst.agent_item_value(a, i) for i in order] for a in agents],
        dtype=np.float64,
    )

    n_agents = len(agents)
    n_items = len(order)
    remaining_mask = np.ones(n_items, dtype=bool)
    agent_mask = np.ones(n_agents, dtype=bool)
    tau_left = tau.copy()
    bundle_no = 0
    EPS = 1e-9

    while agent_mask.any() and remaining_mask.any():
        bundle_mask = np.zeros(n_items, dtype=bool)
        for idx in np.where(remaining_mask)[0]:
            feasible = (costs[agent_mask, idx] <= tau_left[agent_mask] + EPS)
            if feasible.any():
                bundle_mask[idx] = True
        if not bundle_mask.any():
            break
        bundle_idxs = np.where(bundle_mask)[0]

        bundle_costs = costs[:, bundle_idxs].sum(axis=1)
        feasible_agents = np.where(agent_mask & (bundle_costs <= tau_left + EPS))[0]
        if feasible_agents.size == 0:
            for shrink in range(len(bundle_idxs) - 1, -1, -1):
                bundle_idxs_shrunk = bundle_idxs[:shrink]
                if len(bundle_idxs_shrunk) == 0:
                    break
                bundle_costs_shrunk = costs[:, bundle_idxs_shrunk].sum(axis=1)
                feasible_agents = np.where(agent_mask & (bundle_costs_shrunk <= tau_left + EPS))[0]
                if feasible_agents.size > 0:
                    bundle_idxs = bundle_idxs_shrunk
                    bundle_costs = bundle_costs_shrunk
                    break
            if feasible_agents.size == 0 or len(bundle_idxs) == 0:
                break
        chosen_agent = feasible_agents[0]
        chosen_agent_id = agents[chosen_agent]
        bundle_items = [order[i] for i in bundle_idxs]
        bundle_cost = bundle_costs[chosen_agent]
        bundle_no += 1
        logging.info("Bundle #%d → agent %s : %s  (cost %.0f)",
                     bundle_no, chosen_agent_id, bundle_items, bundle_cost)
        for i in bundle_idxs:
            builder.give(chosen_agent_id, order[i])
            remaining_mask[i] = False
        tau_left[chosen_agent] -= bundle_cost
        agent_mask[chosen_agent] = False

    if remaining_mask.any():
        logging.info("Unallocated chores: %s", [order[i] for i in np.where(remaining_mask)[0]])
        return set(order[i] for i in np.where(remaining_mask)[0])
    else:
        logging.info("All chores allocated.")
        return set()


# Nadav's hffd_nadav algorithm
# =========================================================================
def _as_list(obj) -> list[Any]:
    return list(obj() if callable(obj) else obj)


def hffd_nadav(
        builder,
        *,
        thresholds: Mapping[Any, float],
        universal_order: Sequence[Any] | None = None,
) -> set[Any]:
    agents = _as_list(getattr(builder, "agents", builder.instance.agents))
    items = _as_list(getattr(builder, "remaining_items", builder.remaining_items))

    if not isinstance(thresholds, Mapping):
        thresholds = dict(zip(agents, thresholds))
    if len(thresholds) != len(agents):
        raise ValueError("thresholds size must match number of agents")

    tau = {a: float(thresholds[a]) for a in agents}

    inst = builder.instance
    cost = {(a, i): inst.agent_item_value(a, i) for a in agents for i in items}

    order = list(universal_order) if universal_order is not None else items
    if set(order) != set(items):
        raise ValueError("universal_order must be a permutation of items")

    for i in order:
        for a in agents:
            if cost[(a, i)] <= tau[a]:
                builder.give(a, i)
                tau[a] -= cost[(a, i)]
                break
    return set(builder.remaining_items())


# --- User-provided Helper Functions for Input Generation ---
# =========================================================================

def check_ido(agents):
    """
    Check if the input satisfies Identical-Order Preference (IDO).
    For each pair of items (i,j), if any agent considers i more costly than j,
    then all agents must consider i at least as costly as j.
    """
    num_items = len(agents[0])
    for i in range(num_items):
        for j in range(i + 1, num_items):
            # Check if any agent considers i more costly than j
            any_i_more_than_j = any(agent[i] > agent[j] for agent in agents)

            # If any agent considers i more costly than j, check that all agents
            # consider i at least as costly as j
            if any_i_more_than_j:
                if not all(agent[i] >= agent[j] for agent in agents):
                    return False, (i, j)

    return True, None


def validate_input(agents_data, thresholds_data):
    """
    Validate the input parameters for HFFD algorithm
    """
    try:
        # Parse agents' valuations
        agents = []
        for agent_data in agents_data:
            values = [float(x.strip()) for x in agent_data.split(',')]
            if not all(v >= 0 for v in values):
                return False, "All values must be non-negative"
            agents.append(values)

        # Check all agents have same number of items
        if not all(len(a) == len(agents[0]) for a in agents):
            return False, "All agents must have the same number of items"

        # Check IDO property
        is_ido, conflict = check_ido(agents)
        if not is_ido:
            i, j = conflict
            return False, f"Input violates IDO property: Agents disagree on relative ordering of items {i} and {j}"

        # Parse thresholds
        thresholds = [float(t.strip()) for t in thresholds_data.split(',')]
        if len(thresholds) != len(agents):
            return False, "Number of thresholds must match number of agents"
        if not all(t > 0 for t in thresholds):
            return False, "All thresholds must be positive"

        return True, (agents, thresholds)
    except ValueError:
        return False, "Please enter valid numbers"


def generate_random_input(num_agents: int | None = None,
                          num_chores: int | None = None,
                          low_min: int = 1,
                          high_min: int = 6,
                          low_gap: int = 1,
                          high_gap: int = 10,
                          max_attempts: int = 30):
    """
    מחזיר קלט רנדומלי שמקיים IDO (Identical-Order Preference).

    • num_agents / num_chores – אם None יוגרלו (2-5, 5-10).
    • low_min..high_min         – טווח ערך החפץ הקטן ביותר.
    • low_gap..high_gap         – טווח הפערים בין חפצים סמוכים (חיובי ⇒ שומר דירוג).
    • max_attempts              – מספר ניסיונות לייצר וקטורים ייחודיים ו-IDO.

    מחזיר dict:
        {
          'agents'    : ["v1,v2,..."],  # לכל סוכן
          'thresholds': "t1,t2,..."
        }
    """
    if num_agents is None:
        num_agents = np.random.randint(2, 6)  # 2-5
    if num_chores is None:
        num_chores = np.random.randint(5, 11)  # 5-10

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        valuations = []

        # ─── 1) בונים וקטור *נפרד* לכל סוכן ───
        for _ in range(num_agents):
            # ערך החפץ "הכי זול"
            min_val = np.random.randint(low_min, high_min)

            # פערים חיוביים אקראיים בין חפצים סמוכים
            # high_gap + 1 to make high_gap inclusive in randint
            gaps = np.random.randint(low_gap, high_gap + 1, size=num_chores - 1)

            v = np.empty(num_chores, dtype=int)
            v[-1] = min_val
            # הולכים מהחפץ האחרון (הזול) לראשון (היקר)
            for i in range(num_chores - 2, -1, -1):
                v[i] = v[i + 1] + gaps[i]

            valuations.append(v)

        valuations = np.vstack(valuations)

        # ─── 2) ודא ייחודיות בין הסוכנים ───
        if len({tuple(row) for row in valuations}) < num_agents:
            continue

        # ─── 3) בדיקת IDO (כל הסוכנים שומרים אותו סדר) ───
        is_ido, _ = check_ido(valuations)
        if not is_ido:
            continue

        # ─── 4) חישוב ספים ───
        # Thresholds based on sum of valuations for each agent, divided by num_agents
        # This is the base threshold before applying threshold_factor
        thresholds = np.ceil(valuations.sum(axis=1) / num_agents).astype(int)

        return {
            'agents': [','.join(map(str, row)) for row in valuations],
            'thresholds': ','.join(map(str, thresholds))
        }

    # If max_attempts reached without generating valid input, raise an error
    raise RuntimeError(f"Could not generate IDO input after {max_attempts} attempts.")


# --- Test Framework ---
# =========================================================================

def run_test(algorithm_func, n_agents, n_items, threshold_factor, random_seed=None):
    """
    Runs a single test for an algorithm using generate_random_input
    and returns runtime and success.
    """
    # Use the provided random seed for reproducibility across runs,
    # especially important for generate_random_input
    np.random.seed(random_seed)

    # Generate IDO-compliant input
    try:
        # Use user's generate_random_input
        generated_input = generate_random_input(num_agents=n_agents, num_chores=n_items)
    except RuntimeError as e:
        print(f"Skipping test for {n_agents} agents, {n_items} items due to: {e}")
        return 0.0, False  # Return 0 runtime and False success for skipped tests

    # Parse agents data from string format
    agents_data_str = generated_input['agents']
    agent_valuations = {}
    agent_names_list = [f"agent_{i}" for i in range(n_agents)]
    item_names_list = [f"item_{i}" for i in range(n_items)]

    for i, agent_data_csv in enumerate(agents_data_str):
        values = [float(x.strip()) for x in agent_data_csv.split(',')]
        agent_valuations[agent_names_list[i]] = {item_names_list[j]: values[j] for j in range(n_items)}

    inst = fairpyx.Instance(valuations=agent_valuations)
    original_item_count = len(inst.items)  # Get original item count for success check

    # Parse base thresholds from string format and apply threshold_factor
    base_thresholds_str = generated_input['thresholds'].split(',')
    base_thresholds_list = [float(t.strip()) for t in base_thresholds_str]

    thresholds = {
        agent_names_list[i]: int(np.round(base_thresholds_list[i] * threshold_factor))
        for i, _ in enumerate(agent_names_list)
    }

    # Capture logging output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)  # Only capture INFO for success/fail messages
    algorithm_logger = logging.getLogger("hffd_logger")
    if not any(isinstance(h, type(handler)) for h in algorithm_logger.handlers):
        algorithm_logger.addHandler(handler)
    algorithm_logger.propagate = False  # Prevent logs from going to console

    start_time = time.perf_counter()

    # Use fairpyx.adaptors.divide to run the algorithm
    allocation = divide(algorithm_func, instance=inst, thresholds=thresholds)

    end_time = time.perf_counter()
    runtime = end_time - start_time

    # Determine success based on the allocation returned by divide
    allocated_item_count = sum(len(items) for items in allocation.values())
    success = (allocated_item_count == original_item_count)

    algorithm_logger.removeHandler(handler)  # Clean up handler
    algorithm_logger.propagate = True  # Restore propagation

    return runtime, success


# Define algorithm functions to test
algorithms = {
    "hffd": hffd,
    "hffd_fast": hffd_fast,
    "hffd_nadav": hffd_nadav,
}

# Parameters for comparison tests
min_items = 10
max_items = 1000  # Increased max items
step_items = 50  # Adjusted step for more data points over larger range
fixed_agents = 5

min_agents = 2
max_agents = 100  # Increased max agents
step_agents = 10  # Adjusted step for more data points over larger range
fixed_items = 100

num_runs_per_case = 5  # Number of times to run each scenario for averaging
threshold_factors = [0.5, 1.0, 1.5, 2.0]  # Different threshold multipliers

# Data storage
# Nested dictionary: algo_name -> threshold_factor -> {"runtimes": [], "success_rates": []}
results_items_varying = {algo_name: {tf: {"runtimes": [], "success_rates": []} for tf in threshold_factors} for
                         algo_name in algorithms}
results_agents_varying = {algo_name: {tf: {"runtimes": [], "success_rates": []} for tf in threshold_factors} for
                          algo_name in algorithms}

item_counts = list(range(min_items, max_items + 1, step_items))
agent_counts = list(range(min_agents, max_agents + 1, step_agents))

print("Running tests for varying number of items (fixed agents)...")
for n_items in item_counts:
    for threshold_factor in threshold_factors:  # New loop for threshold factors
        for algo_name, algo_func in algorithms.items():
            total_runtime = 0
            successful_runs = 0
            for run_idx in range(num_runs_per_case):
                # Pass threshold_factor and a unique random_seed for each run
                runtime, success = run_test(algo_func, fixed_agents, n_items, threshold_factor, random_seed=run_idx)
                total_runtime += runtime
                if success:
                    successful_runs += 1
            # Store results nested by threshold_factor
            results_items_varying[algo_name][threshold_factor]["runtimes"].append(total_runtime / num_runs_per_case)
            results_items_varying[algo_name][threshold_factor]["success_rates"].append(
                successful_runs / num_runs_per_case)
    print(f"  Finished {n_items} items for all threshold factors.")

print("\nRunning tests for varying number of agents (fixed items)...")
for n_agents in agent_counts:
    for threshold_factor in threshold_factors:  # New loop for threshold factors
        for algo_name, algo_func in algorithms.items():
            total_runtime = 0
            successful_runs = 0
            for run_idx in range(num_runs_per_case):
                # Pass threshold_factor and a unique random_seed for each run
                runtime, success = run_test(algo_func, n_agents, fixed_items, threshold_factor, random_seed=run_idx)
                total_runtime += runtime
                if success:
                    successful_runs += 1
            # Store results nested by threshold_factor
            results_agents_varying[algo_name][threshold_factor]["runtimes"].append(total_runtime / num_runs_per_case)
            results_agents_varying[algo_name][threshold_factor]["success_rates"].append(
                successful_runs / num_runs_per_case)
    print(f"  Finished {n_agents} agents for all threshold factors.")

# --- Plotting ---
# =========================================================================

# Define a color palette for better visual distinction
colors = {
    "hffd": "red",
    "hffd_fast": "blue",
    "hffd_nadav": "green",
}

# Define markers for each algorithm
markers = {
    "hffd": "o",
    "hffd_fast": "s",
    "hffd_nadav": "^",
}

# Plot 1: Runtime Comparison (remains the same, as runtime is mainly size-dependent)
plt.figure(figsize=(16, 7))

# Subplot 1a: Runtime vs. Number of Items (Fixed Agents)
plt.subplot(1, 2, 1)
# For runtime, we can average across threshold factors for a cleaner view.
for algo_name, data in algorithms.items():
    # Ensure there's data for all threshold factors for averaging
    if all(len(results_items_varying[algo_name][tf]["runtimes"]) == len(item_counts) for tf in threshold_factors):
        avg_runtimes = [np.mean([results_items_varying[algo_name][tf]["runtimes"][i] for tf in threshold_factors])
                        for i in range(len(item_counts))]
        plt.plot(item_counts, avg_runtimes,
                 label=algo_name,
                 color=colors[algo_name],
                 marker=markers[algo_name],
                 linestyle='-',
                 linewidth=2)
    else:
        print(
            f"Warning: Not enough data points for {algo_name} items varying to average runtimes across all threshold factors.")
plt.xlabel("Number of Items", fontsize=12)
plt.ylabel("Average Runtime (seconds)", fontsize=12)
plt.title(f"Algorithm Runtime vs. Number of Items (Agents = {fixed_agents})", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()

# Subplot 1b: Runtime vs. Number of Agents (Fixed Items)
plt.subplot(1, 2, 2)
for algo_name, data in algorithms.items():
    # Ensure there's data for all threshold factors for averaging
    if all(len(results_agents_varying[algo_name][tf]["runtimes"]) == len(agent_counts) for tf in threshold_factors):
        avg_runtimes = [np.mean([results_agents_varying[algo_name][tf]["runtimes"][i] for tf in threshold_factors])
                        for i in range(len(agent_counts))]
        plt.plot(agent_counts, avg_runtimes,
                 label=algo_name,
                 color=colors[algo_name],
                 marker=markers[algo_name],
                 linestyle='-',
                 linewidth=2)
    else:
        print(
            f"Warning: Not enough data points for {algo_name} agents varying to average runtimes across all threshold factors.")
plt.xlabel("Number of Agents", fontsize=12)
plt.ylabel("Average Runtime (seconds)", fontsize=12)
plt.title(f"Algorithm Runtime vs. Number of Agents (Items = {fixed_items})", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.suptitle("HFFD Algorithm Runtime Comparison (Averaged over Thresholds)", fontsize=16, y=1.02)
plt.savefig("hffd_runtime_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Success Rate Comparison - Varying Items, by Threshold Factor
num_threshold_factors = len(threshold_factors)
fig_rows = int(np.ceil(num_threshold_factors / 2))  # Arrange in 2 columns
fig_cols = 2

plt.figure(figsize=(8 * fig_cols, 6 * fig_rows))  # Adjust figure size dynamically
plt.suptitle(f"Algorithm Success Rate vs. Number of Items (Agents = {fixed_agents}) by Threshold Factor", fontsize=16,
             y=1.02)

for i, tf in enumerate(threshold_factors):
    plt.subplot(fig_rows, fig_cols, i + 1)
    for algo_name, data in algorithms.items():
        plt.plot(item_counts, results_items_varying[algo_name][tf]["success_rates"],
                 label=algo_name,
                 color=colors[algo_name],
                 marker=markers[algo_name],
                 linestyle='-',
                 linewidth=2)
    plt.xlabel("Number of Items", fontsize=10)
    plt.ylabel("Success Rate (Proportion)", fontsize=10)
    plt.title(f"Threshold Factor: {tf}", fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to make space for suptitle
plt.savefig("hffd_success_rate_items_by_threshold.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Success Rate Comparison - Varying Agents, by Threshold Factor
plt.figure(figsize=(8 * fig_cols, 6 * fig_rows))  # Adjust figure size dynamically
plt.suptitle(f"Algorithm Success Rate vs. Number of Agents (Items = {fixed_items}) by Threshold Factor", fontsize=16,
             y=1.02)

for i, tf in enumerate(threshold_factors):
    plt.subplot(fig_rows, fig_cols, i + 1)
    for algo_name, data in algorithms.items():
        plt.plot(agent_counts, results_agents_varying[algo_name][tf]["success_rates"],
                 label=algo_name,
                 color=colors[algo_name],
                 marker=markers[algo_name],
                 linestyle='-',
                 linewidth=2)
    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Success Rate (Proportion)", fontsize=10)
    plt.title(f"Threshold Factor: {tf}", fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to make space for suptitle
plt.savefig("hffd_success_rate_agents_by_threshold.png", dpi=300, bbox_inches='tight')
plt.close()

# NEW PLOT: Plot 4: Average Success Rate vs. Number of Agents (Fixed Items)
plt.figure(figsize=(10, 7))
for algo_name, data in algorithms.items():
    # Ensure there's data for all threshold factors for averaging
    if all(len(results_agents_varying[algo_name][tf]["success_rates"]) == len(agent_counts) for tf in
           threshold_factors):
        avg_success_rates = [
            np.mean([results_agents_varying[algo_name][tf]["success_rates"][i] for tf in threshold_factors])
            for i in range(len(agent_counts))]
        plt.plot(agent_counts, avg_success_rates,
                 label=algo_name,
                 color=colors[algo_name],
                 marker=markers[algo_name],
                 linestyle='-',
                 linewidth=2)
    else:
        print(
            f"Warning: Not enough data points for {algo_name} agents varying to average success rates across all threshold factors.")
plt.xlabel("Number of Agents", fontsize=12)
plt.ylabel("Average Success Rate (Proportion)", fontsize=12)
plt.title(f"Average Algorithm Success Rate vs. Number of Agents (Items = {fixed_items})", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-0.05, 1.05)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig("hffd_avg_success_rate_agents.png", dpi=300, bbox_inches='tight')
plt.close()
