# JLI — The Junction-Linked List

**JLI** is a sorted linked list with a fast search index — but unlike a skip list, the index doesn't live inside the nodes. It lives entirely *outside* the list, in a separate three-level structure (junctions → blocks → block-level skip list) that sits on top of a completely ordinary, unmodified singly-linked list.

This repo contains the C11 implementation, the benchmark harness, the raw/aggregated result data, and the paper (submitted to FSTTCS 2026, Track A) that describes and proves it.

> 📄 Paper: [`FSTTCS_SUBMISSION_NO_8.pdf`](./FSTTCS_SUBMISSION_NO_8.pdf)

---

## The idea in one paragraph

A sorted linked list is cheap to splice but costs O(n) to search. The classical fix is a **skip list**: give every node a randomized tower of forward pointers, so search descends the towers in O(log n). That's effective, but it's also a *per-node tax* — every single node pays for an index whether or not it's ever used to route a query. JLI asks: does that index have to live on the node at all? JLI keeps the base list a genuine, plain linked list (nodes only carry `value`, `payload`, `next` — no index field), and moves the entire search apparatus into an external hierarchy that indexes the list in *segments* rather than per element.

Removing the index from the node removes that tax, but creates a new problem: nothing on a plain node can signal that a segment has drifted out of shape under repeated inserts/deletes. JLI solves this with a three-tier deferred maintenance scheme that watches for drift and repairs it at increasing granularity, only when needed.

---

## How it works

### The structure, bottom to top

1. **Base list** — an ordinary sorted singly-linked list of `Node { value, payload, next }`. No index field, no per-node overhead beyond a plain linked list.
2. **Junctions** — the list is partitioned into contiguous **segments** of `S` nodes. Each segment gets one external **junction**: a struct holding the segment's start/end pointers, an interior anchor node, and an array of `K` **shortcut pointers** spaced across the segment. Search finds the right junction, binary-searches its `K` shortcuts, then walks at most `S/K` remaining nodes by hand.
3. **Blocks** — junctions are grouped into **blocks** of up to `M` junctions, stored as arrays (not pointer-chained), so a block's junction can be found by binary search too.
4. **Block-level skip list** — a standard probabilistic skip list, but built *over blocks* rather than over individual nodes. It's rebuilt from scratch on any structural change rather than patched incrementally.

A search descends all three layers — skip list over blocks (O(log B)), binary search within a block (O(log M)), binary search over a junction's shortcuts (O(log K)) — then does a short bounded linear walk (O(S/K)). Net: **O(log n + S/K)** amortized, which collapses to a clean O(log n) if K is scaled with S (in the benchmarks here, K is held fixed, so the honest bound keeps the S/K term explicit).

### Keeping the index honest: three-tier maintenance

Because the index is *outside* the node, mutations can leave it stale in two independent ways:
- **Positional drift** — a junction's anchor wanders off the segment's midpoint.
- **Size drift** — a segment's length drifts away from the target `S`.

JLI handles this with three escalating repair tiers, each scoped to a different region size and checked at a different interval, so cheap repairs happen often and expensive ones only when the cheap ones aren't enough:

| Tier | Scope | Trigger |
|---|---|---|
| **Local rebuild** | One junction's shortcuts | Anchor drifts outside a tolerance band |
| **Sub-optimal rebuild** | A contiguous run of flagged junctions | Flagged-segment ratio crosses a threshold |
| **Global rebuild** | Full reconstruction (`Build` from scratch) | Emergency hard-drift ratio, or sustained sub-optimal rebuilds without convergence |

A separate, always-on mechanism (independent of the three tiers) handles **underflow**: any delete that shrinks a segment to ≤2 nodes triggers an immediate merge/removal, never deferred.

### 41 structural invariants, proved against the implementation

The paper doesn't just assert correctness — it states 41 structural invariants (sort order, partition consistency, shortcut-array shape, maintenance escalation rules, etc.) and gives inductive proofs checked line-by-line against the C implementation. The pass:
- Proved **32** exactly as originally stated.
- Proved **4 more** (shortcut monotonicity, block capacity, skip-list tower size, maintenance-reset attribution) in a corrected or more precise form.
- Found **4** (interval/threshold ordering validation) genuinely **unenforced** in an earlier build — a real bug, reported and since fixed (the validator is now called unconditionally in the constructor).
- Left **1** with a known, presently-inert accounting gap (a payload-byte counter not decremented on delete — never exercised in these payload-free benchmarks).

---

## What's better, and what isn't

JLI is compared against a **textbook probabilistic skip list** (p = 0.5) — deliberately the sharpest baseline available, since a skip list isolates the "index lives on the node" cost with no other structural change (unlike a B-tree, which replaces the list entirely). It is *not* compared against B-trees/B+-trees empirically; Appendix H gives a complexity-only comparison instead (see below).

**Memory — unconditionally better.**
Across all 87 tested configurations, JLI's structural index overhead was **75.0%–80.4%** of the skip list's (mean ≈77.8%), with **zero exceptions**. This holds most cleanly in BUILD (a single fixed configuration, never tuned for search), showing the memory win is structural, not a side effect of the parameter search favoring memory-friendly configs.

**Search — unconditionally better in this study.**
JLI's mean search latency was lower than the skip list's in **all 30 STATIC configurations tested**, never reversing at the sizes and machine tested. Advantage ranges from **1.07×–1.12×** under `miss` (worst case) up to **2.68×–3.90×** under `adversarial` (best case).

**Mutation — a real tax, direction-dependent.**
This is where the trade-off resurfaces:
- **INSERT**: averaged with sign, +7.28% slower than the skip list (median +5.41%). Actually *faster* under `adversarial` (tail-append) and `random`; worst under `zipfian` (up to 1.36×).
- **DELETE**: +4.31% slower on average (median +2.51%). Close to parity or faster on most patterns, but sharply worse under `adversarial` (tail-delete), up to 1.33×.

The mechanism is the same in both directions: sustained mutation concentrated at one boundary (the tail) drives the three-tier maintenance system to fire repeatedly, and each fire pays a fixed O(B) rebuild-floor cost that doesn't amortize away — a workload class where the paper's own amortized-O(log n) claim is proven, not assumed, to fail.

**vs. B+-trees (complexity only, no benchmarks run).**
JLI trades a **probabilistic** bound for a smaller constant factor; a B+-tree offers a **deterministic**, per-operation worst-case guarantee and disk/block alignment that JLI doesn't attempt to compete on. This is a genuine, stated limitation, not glossed over.

---

## How it was tested

- **Baseline**: a textbook probabilistic skip list (p = 0.5), same translation unit, same key pools, generators, and memory/timing accounting code — so the comparison is apples-to-apples.
- **Sections**: `BUILD` (bulk construction, 7 sizes), `STATIC` (search-only, 6 access patterns), `INSERT` and `DELETE` (5 patterns each) — **87 total (section, pattern, n) configurations**, every one reported, none omitted.
- **Sizes**: n ∈ {10K, 50K, 100K, 250K, 500K, 750K, 1M} for BUILD; {50K, 100K, 250K, 500K, 1M} for the rest.
- **Access patterns**: `random`, `sequential`, `hotspot`, `zipfian`, `miss` (STATIC only), and `adversarial` — note `adversarial` means three *different* things per section (guaranteed-miss lookups in STATIC, monotonic tail-append in INSERT, tail-backward delete in DELETE); the paper spells out each precisely rather than relying on the name.
- **Tuning**: STATIC/INSERT/DELETE were each tuned via a bounded hierarchical random search (100–200 trials) *optimizing solely for search-latency ratio* — memory was never a search objective, which is why the memory-invariance result is treated as a found regularity, not an enforced one. BUILD used one fixed, untuned configuration.
- **Repetition**: 30 internal timed runs per process, aggregated into mean/median/percentile stats, averaged across 10 process repetitions (first discarded as cold start).
- **Hardware**: primary run on an AMD Ryzen 7 7735HS (8 cores, 16 GB RAM); STATIC was independently re-run in full on a second machine (Intel i5-1155G7, 4C/8T, 8 GB RAM) as a cross-device check. Memory replicated cleanly (75.1–79.2%); the search-latency advantage held directionally but narrowed — and reversed on a few patterns — at n = 1M on the lower-core-count machine, so the latency win is flagged as more hardware-sensitive than the primary numbers alone suggest.
- **Statistics**: `collect_row.py` aggregates all raw/aggregated CSVs and computes, per configuration, the mean JLI/skip-list ratio, its 95% CI, a Wilcoxon signed-rank test, and a sign test — all included in `all_aggregated.csv`.

---

## Repo layout

```
jli_v8_1.c              # The JLI data structure itself (Build, Search, Insert, Delete, MaintenanceHook)
8.c                      # Benchmark harness — includes jli_v8_1.c, implements the reference skip list,
                         #   pattern generators, and timing/memory accounting
parameter_search.py      # Hierarchical random search over JLI's tuning parameters (S, K, block size,
                         #   maintenance thresholds), per (section, pattern, size)
collect_row.py           # Recursively collects raw_run_*.csv / aggregated.csv output, infers metadata
                         #   from folder structure, and computes ratio/CI/Wilcoxon/sign-test statistics
all_raw_runs.csv         # Every individual timed run, consolidated
all_aggregated.csv       # Per-configuration aggregated results (the 87-row table behind Table 6)
all_raw_runs_search.csv  # Raw runs from the parameter-search phase
all_aggregated_search.csv# Aggregated results from the parameter-search phase
FSTTCS_SUBMISSION_NO_8.pdf # The paper
```

### Building and running

```bash
gcc -O3 -o bench 8.c -lm
./bench   # runs the configured benchmark sections; see parameter_search.py for search-mode invocation
```

`parameter_search.py` drives the compiled harness across the hierarchical search; `collect_row.py` then walks the resulting output tree and produces the two consolidated CSVs above (pass `--input-dir` pointing at the results tree).

---

## Where JLI sits, honestly

This is presented as a narrow, specific result, not a general claim that skip lists are obsolete:

- It's tested against **one baseline** (a probabilistic skip list), chosen because it's the sharpest available test of "does the index have to live on the node." B-trees are excluded from the empirical comparison by design (they abandon the list representation entirely) and only compared analytically (Appendix H).
- The parameter search optimized **one axis** (search latency) and is a bounded 100–200 trial search, not a verified global optimum.
- Everything reported is **single-threaded** and **payload-free**; concurrency is future work, and the memory ratio is expected to move toward parity as real payload size grows.
- The clean O(log n) search bound assumes K scales with S; every benchmarked configuration here holds K fixed, so the honest bound is O(log n + S/K).

Full detail, all 87 rows, and the complete proof appendix are in the paper.
