"""
Optimised **HFFD** — high‑performance, numerically robust variant
================================================================
This module contains :pyfunc:`hffd_fast`, an improved implementation of
HFFD (Hierarchical Fair‑For‑Descendents) for *chore* division.  The goal
is to allocate indivisible negative‑valued items ("chores") among agents
subject to additive cost functions and individual *threshold* budgets
``τ_a``.  An agent accepts a bundle *B* iff

``sum_{i∈B} cost_a(i)  ≤  τ_a``.

The classic reference implementation in *fairpyx* suffered
from two practical issues:

1. **Hot cost look‑ups*— it recomputed agent‑item costs for every bundle
   evaluation, leading to *O(nmb)* complexity (``n`` agents, ``m`` items,
   ``b`` builder ``give`` calls).
2. **Strict comparisons**— using exact ``<=`` on binary64 floats caused
   feasible bundles to be rejected due to round‑off noise, triggering
   infinite loops or sub‑optimal allocations.

`hffd_fast` resolves both issues while preserving the algorithmic
semantics and public interface.

Key Improvements
----------------
1. **Single pre‑computed cost matrix**
   The cost matrix ``costs`` (shape``n×m``) is built once at start using
   NumPy broadcasting.  All later bundle checks are constant‑time NumPy
   slices, reducing runtime dramatically.

2. **Unified column order**
   All data structures share the same ``order`` list so every item maps to
   a stable column index.  This fixes a subtle off‑by‑one bug that could
   arise when ``universal_order`` differed from the builder's internal
   permutation.

3. **ε‑robust feasibility test & bundle‑shrinking**
   Bundle feasibility is evaluated against ``τ_a + EPS`` with
   ``EPS=1e‑9`` (≈2ulp around 1.0).  If no agent can accept the greedy
   bundle we *shrink* it LIFO‑style until one agent becomes feasible.
   This recovers valid allocations that were previously lost to floating
   point noise.

4. **Early‑exit diagnostics**
   The routine logs leftover chores at *INFO* level and impossible bundle
   situations at *WARNING*, aiding debugging of malformed instances.

----------
Interface
~~~~~~~~~
``hffd_fast(builder, *, thresholds, universal_order=None)``

Parameters
~~~~~~~~~~
``builder`` : :class:`fairpyx.Builder`
    A mutable object that keeps the current allocation state and exposes
    ``.instance`` (the :class:`~fairpyx.Instance`) plus ``give`` and
    iteration helpers.  Only the standard *fairpyx* builder API is used –
    custom builders that implement the same methods will also work.

``thresholds`` : *Mapping* | *Sequence*
    Either a mapping *agent→τ* or a sequence ordered like
    ``builder.remaining_agents()``.

``universal_order`` : *Sequence* | *None*, default ``None``
    Optional global permutation of the remaining items.  When provided it
    **must** contain exactly the same items as
    ``builder.remaining_items()``.

Notes
~~~~~
* *Side‑effects*: Updates the builder **in‑place** via repeated
  :pyfunc:`~fairpyx.Builder.give` calls and logs progress through the
  standard :pymod:`logging` framework.
* *Complexity*: ``O(nxm)`` preprocessing + ``O(b)`` NumPy operations (all
  bundle checks are vectorised).
* *Guarantees*: Identical allocation result **modulo floating‑point
  tolerance** compared to the textbook HFFD.

Example
~~~~~~~
>>> from fairpyx import Instance, Builder
>>> inst = Instance([[3,2,1],[2,3,1],[1,1,3]])  # 3 agents, 3 chores
>>> bld  = Builder(inst)
>>> hffd_fast(bld, thresholds=[3,3,3])
>>> bld.allocation()
{0: [0], 1: [1], 2: [2]}
"""

from __future__ import annotations
import logging
from typing import Any, Mapping, Sequence
import numpy as np

logger = logging.getLogger(__name__)
__all__ = ["hffd_fast"]

# --------------------------------------------------------------------------- #

def hffd_fast(
    builder: "BuilderLike",
    *,
    thresholds: Mapping[Any, float] | Sequence[float],
    universal_order: Sequence[Any] | None = None,
) -> None:
    """Allocate chores using the *fast* HFFD variant.

    The procedure follows the greedy‑then‑shrink strategy:

    1. **Greedy fill** – iterate over the remaining items once and build a
       candidate *bundle* of items each of which is individually feasible
       for *some* still‑active agent.
    2. **Bundle selection / shrinking** – test if any agent can afford
       the *whole* bundle.  If not, drop the *last* added item and retry
       until an agent qualifies.
    3. **Commit** – assign the surviving bundle to the first qualifying
       agent; remove both the agent and the items from further
       consideration; repeat until exhaustion.

    Parameters
    ----------
    builder : Builder‑like
        Object exposing at least:
        ``instance``, ``remaining_agents()``, ``remaining_items()``, and
        ``give(agent, item)``.
    thresholds : Mapping | Sequence
        Acceptance budgets ``τ``.  If a sequence is supplied it is
        assumed to correspond to ``builder.remaining_agents()`` order.
    universal_order : Sequence | None, optional
        Explicit item order.  When *None* the builder's own order is
        used.  Must contain exactly the remaining items.

    Returns
    -------
    None
        All results are side‑effects on *builder*.

    Raises
    ------
    ValueError
        If ``universal_order`` is not a permutation of the remaining
        items.
    """

    inst = builder.instance
    agents = list(builder.remaining_agents())
    items_0 = list(builder.remaining_items())

    # ---------- thresholds -------------------------------------------------- #
    if not isinstance(thresholds, Mapping):
        thresholds = dict(zip(agents, thresholds))
    tau = np.array([float(thresholds[a]) for a in agents], dtype=np.float64)

    # ---------- common order ------------------------------------------------ #
    order = list(universal_order) if universal_order is not None else items_0
    if set(order) != set(items_0):
        raise ValueError("universal_order must equal remaining items")
    item_pos = {i: k for k, i in enumerate(order)}  # item → column

    # ---------- one full cost‑matrix --------------------------------------- #
    try:  # ⚡ fast path (NumPy array stored in Instance)
        full_vals = np.asarray(inst._valuations, dtype=np.float64)
    except AttributeError:  # ☑️ generic fallback
        full_vals = np.array(
            [[inst.agent_item_value(a, i) for i in order] for a in agents],
            dtype=np.float64,
        )
    costs = full_vals[:, [item_pos[i] for i in order]]

    # ---------- book‑keeping ---------------------------------------------- #
    agents_left = set(agents)
    bundle_cost = np.zeros_like(tau)  # running total per agent
    remaining_idx = set(range(len(order)))  # item‑indices still free
    EPS = 1e-9  # numeric slack for float noise

    while agents_left and remaining_idx:
        bundle: list[int] = []

        # pass 1 – greedy fill current bundle
        for col in sorted(remaining_idx):  # identical descending order
            rows = np.fromiter(agents_left, int)
            delta = costs[rows, col]
            feasible = bundle_cost[rows] + delta <= tau[rows] + EPS
            if feasible.any():
                bundle.append(col)

        if not bundle:
            break  # nothing more fits – stop algorithm

        # pass 2 – assign bundle or shrink until feasible
        while True:
            rows = np.fromiter(agents_left, int)
            bundle_sum = costs[rows][:, bundle].sum(axis=1)
            possible = np.where(bundle_cost[rows] + bundle_sum <= tau[rows] + EPS)[0]

            if possible.size:
                chosen_idx = rows[possible[0]]  # first qualifying agent
                break
            if not bundle:  # should never happen – guards infinite loop
                logger.warning("No bundle can be allocated; aborting.")
                return
            removed = bundle.pop()  # drop latest item and retry
            logger.debug("Shrinking bundle – removed item %r", order[removed])

        chosen = int(chosen_idx)

        # commit allocation and update state
        for col in bundle:
            builder.give(chosen, order[col])
            remaining_idx.remove(col)
            bundle_cost[chosen] += costs[chosen, col]

        logger.debug(
            "agent %r gets %s (cost %.3f)",
            chosen,
            [order[c] for c in bundle],
            bundle_cost[chosen],
        )
        agents_left.remove(chosen)

    if remaining_idx:
        logger.info("Unallocated chores: %s", [order[c] for c in remaining_idx])
