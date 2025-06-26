# Analysis of HFFD Algorithm Variants

This document analyzes the performance and fairness of two implementations of the HFFD algorithm: the original `hffd` (blue) and a modified version, `hffd_nadav` (orange).

## Key Take-aways

*   **Speed**: Every plot that involves time shows a 1–2 orders-of-magnitude gap in favour of `hffd_nadav`. Profiling confirms the old version spends most of its time in repeated scans and pandas bookkeeping.
*   **Fairness**: The success rate stays at 100% and overload remains negative for both versions. The variant does not sacrifice guarantees.
*   **Quality**: The original `hffd` sometimes leaves a slightly larger safety margin (more negative overload), but the difference is small and may be irrelevant in practice.
*   **Overall**: Nadav’s rewrite keeps the theoretical properties while eliminating the algorithmic bottleneck, which is why the orange lines hug the x-axis in every runtime plot.

## Graph 1: Log-Runtime vs. Instance Size
*(blue = hffd, orange = hffd_nadav)*

**What you see**
- The x-axis represents the total number of items (agents × items-per-agent).
- The y-axis is log-10 seconds, meaning every tick mark is a 10x increase.
- `hffd` runtime grows from approximately 10⁻³ seconds at 30 items to approximately 10⁰·⁵ seconds (> 3 seconds) at 960 items, showing roughly quasi-linear growth in n.
- `hffd_nadav` starts an order of magnitude faster and remains below 10⁻¹·⁵ seconds even for the largest instances.

**Why it probably happens**
Nadav’s variant uses a single forward pass with cheap list operations. In contrast, the original HFFD repeatedly re-scans the unallocated tail and performs more bookkeeping. This complexity gap becomes significant as soon as the instance size exceeds the L2 cache, at which point each extra scan costs real time.

## Graph 2: Average Max-Overload vs. α
*(negative = below the threshold, so “more negative” means better)*

**What you see**
- Both algorithms consistently stay below the threshold line (overload < 0).
- `hffd` (blue) is consistently a few percent lower (more negative) than `hffd_nadav` (orange).
- As α (alpha) increases, the gap between the two widens.

**Why it probably happens**
The two implementations break ties differently. The original HFFD gives the most expensive remaining chore to the current "heaviest" agent, which can under-load them at the end. Nadav’s implementation stops as soon as the next item would cause an overshoot, resulting in bundles closer to the threshold τₐ. With a larger α, the thresholds are looser, allowing the original algorithm to leave even more slack, which explains its steeper slope.

## Graph 3: Empirical CDF of the Runtime

**What you see**
- For `hffd_nadav` (orange), 90% of the runs finish in under 2 ms, and the slowest run is still below 0.03 seconds.
- For `hffd` (blue), runtimes are spread from 5 ms up to 3.7 seconds. There is a long, flat section indicating that no runs finished between 1 and 2 seconds, which suggests a heavy-tail distribution.

**Why it probably happens**
The runtime of the original `hffd` is instance-dependent, based on how quickly every agent is "satisfied." The worst-case scenario forces it to keep looping. Nadav’s one-shot sweep has an almost fixed cost, so its Cumulative Distribution Function (CDF) jumps to 1 very quickly. The heavy tail on the right for the original `hffd` explains its large mean runtime, even though typical runs are faster.

## Graph 4: Mean Runtime vs. Number of Agents
*(ignore the duplicated legend entries – this is a plotting artifact)*

**What you see**
- With just 10 agents, the original `hffd` already takes approximately 0.02 seconds. At 80 agents, it surpasses 1.4 seconds, showing a complexity of roughly O(n²).
- Nadav’s `hffd_nadav` line is almost flat, hovering around 4 ms, which is well within a 30-second budget.

**Why it probably happens**
The original code iterates through every unassigned chore for every agent that is still "heavy." This leads to a complexity of roughly n·m ≈ n² when the number of items per agent is fixed. The variant processes each chore exactly once, so it scales linearly, O(n·m), which is linear in n for a fixed number of items per agent.

## Graph 5: Runtime vs. Instance Size (Linear Y-Scale)

**What you see**
- This graph conveys the same message as Graph 1 but uses a linear scale for absolute seconds. The original `hffd` climbs steadily to over 3 seconds, while the new `hffd_nadav` stays almost invisible near the zero line.

**Why it probably happens**
The linear scale makes the raw performance gap appear even larger. The "elbow" at approximately 200 items marks the point where the original algorithm starts performing extra passes over the data.

## Graph 6: Success-Rate vs. α

**What you see**
- Both curves are flat at 1.0, indicating that every run successfully met all thresholds for every value of α tested.

**Why it probably happens**
The IDO (Identical Descending Order) instances are specifically designed for HFFD. This ordering guarantees a feasible prefix for every agent. Nadav’s simplification does not alter this invariant, so fairness is maintained despite the significant speed-up.