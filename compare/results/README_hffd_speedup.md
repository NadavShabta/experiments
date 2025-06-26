# How we accelerated HFFD

| Bottleneck in vanilla `hffd` | Change in `hffd_fast` | Why this helps |
|------------------------------|-----------------------|----------------|
| **Repeated bundle-cost sums** – for every candidate agent we re-summed all chores already in the tentative bundle ( `O(|bundle|)`  each time). | **Running prefix totals**: keep `bundle_cost[a]` that we update *incrementally* when a new item is appended. | turns the inner feasibility check from **O(k)** to **O(1)**, where *k* is bundle length. |
| **Python loops over agents × items** for `inst.agent_item_value(a,i)`. | Build one **NumPy matrix** `cost[a, i]` once ⇒ pure C loops thereafter. | vectorised lookup is ~30–50× faster than Python attr-calls. |
| **Dict look-ups inside hot loops** (`thresholds[a]`) | Materialise **local NumPy array** `tau[idx]` in the tight loop. | eliminates hashing; converts comparison to raw `float32` array ops. |
| **No early exit** once a feasible agent is found. | Break immediately after the first agent passes the check. | saves scanning the remaining agents for most bundles. |

### Empirical gain  

On the test grid in `compare_hffd_algorithms.py` (40 agents × 6 items *etc.*) the mean run-time dropped

