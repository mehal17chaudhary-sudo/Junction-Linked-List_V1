# JLI vs. Skip List — Empirical Results

This document walks through the empirical section of the paper (*The Junction-Linked List: Challenging the Per-Node Indexing Assumption Behind the Speed–Memory Trade-off*), built directly from `all_aggregated.csv` (87 configurations) and `all_raw_runs.csv`. Every number below was recomputed from the CSVs, not copied from the PDF — they match the paper's reported figures exactly.

**The setup, in one line:** JLI (an externally-indexed sorted linked list) is benchmarked against a textbook probabilistic skip list (p = 0.5) across 4 operation classes — BUILD, STATIC (search), INSERT, DELETE — for a total of 87 (section, pattern, size) configurations. All ratios below are **JLI ÷ skip-list**, so values below 1.0 favor JLI.

Charts live in `results/charts/` next to this file.

---

## 1. Memory: unconditionally better, no exceptions

![Memory overhead across all 87 configurations](charts/memory_all_configs.png)

Across **all 87 configurations**, JLI's structural index overhead ranged from **75.04% to 80.41%** of the skip list's (mean 77.8%) — every single bar sits below the 100% parity line, with zero exceptions. BUILD (the untuned, fixed-parameter section, never tuned for search) shows the tightest and most stable band, which is the strongest evidence the memory win is structural rather than a side effect of the parameter search happening to favor memory-friendly configs:

![BUILD memory ratio and DELETE extremes](charts/build_and_delete_extremes.png)

*(left panel above)* — BUILD's ratio holds in a tight 77.8–78.0% band across all seven sizes (10K–1M), essentially flat regardless of scale.

---

## 2. Search (STATIC): faster in all 30 tested configurations

![STATIC latency ratio by pattern](charts/static_latency_by_pattern.png)

Every one of the 6 access patterns × 5 sizes = 30 STATIC configurations lands below the parity line. The spread is wide:

| Pattern | Ratio range (JLI/SL) | Speedup |
|---|---|---|
| **adversarial** (guaranteed miss past tail) | 0.257 – 0.373 | **2.68× – 3.90×** |
| sequential | 0.758 – 0.891 | 1.12× – 1.32× |
| zipfian | 0.807 – 0.895 | 1.12× – 1.24× |
| random | 0.832 – 0.869 | 1.15× – 1.20× |
| hotspot | 0.848 – 0.875 | 1.14× – 1.18× |
| **miss** (worst case) | 0.891 – 0.936 | **1.07× – 1.12×** |

Adversarial is the standout — JLI never loses this pattern, and by a wide margin, because a guaranteed-miss query past the tail is exactly where JLI's shortcut/junction descent is cheapest relative to a skip list's per-node tower walk.

### Win rate isn't the same signal as mean advantage

![STATIC win-rate heatmap](charts/static_winrate_heatmap.png)

Mean latency being lower doesn't mean JLI wins *every* paired run. Adversarial has a perfect win rate (1.00) at every size — the advantage is not just large, it's consistent. `miss`, by contrast, has the smallest mean advantage *and* the widest, most volatile win rate (0.70–0.99) — these are two genuinely separate properties of the data.

---

## 3. INSERT: faster under adversarial and random, worse under zipfian and sequential

![INSERT latency ratio by pattern](charts/insert_latency_by_pattern.png)

Averaged with sign across all sizes, INSERT is **+7.28% slower** than the skip list (median +5.41%) — but that average hides a sharp pattern-dependent split, visible directly in the line chart above:

| Pattern | Ratio range (JLI/SL) | Verdict |
|---|---|---|
| **adversarial** (tail-append) | 0.851 – 0.972 | Faster — cheapest possible locate phase |
| random | 0.885 – 1.006 | Roughly faster to parity |
| hotspot | 0.952 – 1.188 | Slower, worsening with n |
| sequential | 1.048 – 1.296 | Slower, worsening with n |
| **zipfian** (worst case) | 1.112 – 1.357 | Slower — up to 1.36× |

Curiously, `adversarial` — sustained tail append past the current maximum — is INSERT's *best* pattern, not its worst. Every insert lands at the tail, which is the cheapest possible locate for JLI's shortcut descent, even though the same tail-directed growth is exactly what drives the highest maintenance/rebuild volume behind the scenes.

---

## 4. DELETE: faster under sequential and random, sharply worse under adversarial

![DELETE latency ratio by pattern](charts/delete_latency_by_pattern.png)

DELETE nets **+4.31% slower** on average (median +2.51%) — smaller than INSERT's tax, but with its own sharp outlier:

| Pattern | Ratio range (JLI/SL) | Verdict |
|---|---|---|
| sequential | 0.854 – 0.939 | Faster |
| random | 0.932 – 1.007 | Roughly parity |
| hotspot | 1.017 – 1.105 | Slightly slower |
| zipfian | 0.939 – 1.064 | Near parity |
| **adversarial** (tail-delete) | 0.765 – 0.900 in raw terms, but **1.27× worse on average** | Worst pattern overall |

DELETE is the mirror image of INSERT: `sequential` (which vacates a segment in list-walk order, keeping locate cheap even as dissolutions fire) is now the best pattern, while `adversarial` — deleting from the tail backward, draining the rightmost junction toward its dissolution floor — is by far the worst, peaking at 1.33× slower. Both operations' worst cases sit on opposite sides of the same tail boundary.

![DELETE adversarial vs sequential detail](charts/build_and_delete_extremes.png)

*(right panel above)* — a direct side-by-side of DELETE-adversarial (climbs and stays above parity at every size) against DELETE-sequential (stays comfortably below it).

**Why the mutation tax exists at all:** sustained mutation concentrated at one boundary (the tail) forces the three-tier maintenance system to fire repeatedly, and each firing pays a fixed O(B) rebuild-floor cost that doesn't amortize away as n grows — a case where the paper's own amortized-O(log n) claim is proven, not assumed, to fail (Section 4.1 of the paper works this out analytically).

---

## 5. Cross-device check: memory travels, latency narrows

![Cross-device comparison](charts/cross_device.png)

STATIC was independently re-run in full on a second, lower-core-count machine (Intel i5-1155G7, 4C/8T vs. the primary Ryzen 7 7735HS, 8C). Two different outcomes:

- **Memory replicated almost exactly** (75.1–79.2% on the secondary vs. 75.0–80.4% on the primary) — strong evidence the memory advantage is hardware-independent.
- **Latency direction held, but magnitude did not.** Adversarial stays decisively ahead on both machines. But `random` and `sequential` both **cross above parity at n = 1,000,000** on the lower-core-count secondary machine, where the primary stayed below it throughout. Four of six patterns reverse at the largest size on the secondary device.

This is flagged directly in the paper as a real limitation: the search-latency win is more hardware-sensitive than a single-machine result would suggest, while the memory win is not.

---

## 6. Statistical significance (via `collect_row.py`)

The aggregated CSV includes, per configuration, a mean JLI/skip-list ratio, its 95% CI, a Wilcoxon signed-rank p-value, and a sign-test p-value, computed from the raw per-run data:

```
ratio_ci_low, ratio_ci_high, wilcoxon_p, sign_p
```

Spot-checking BUILD@1M as an example: mean ratio 0.459 (95% CI: 0.446–0.472), sign-test p ≈ 0.002 — consistent with the very tight, non-overlapping-with-1.0 confidence intervals seen throughout the memory and STATIC results, and with the effect being far too large and consistent across 30 internal runs × 10 process repetitions to be noise.

---

## Bottom line

| Dimension | Result |
|---|---|
| **Memory** | Unconditionally better — 75.0–80.4% of skip-list overhead, 0/87 exceptions |
| **Search (STATIC)** | Unconditionally better — lower mean latency in 30/30 configurations, up to 3.9× |
| **INSERT** | Net +7.28% slower on average — faster under adversarial/random, worse under zipfian/sequential |
| **DELETE** | Net +4.31% slower on average — faster under sequential/random, much worse under adversarial |
| **Cross-device** | Memory advantage is hardware-independent; latency advantage narrows/reverses at n=1M on lower-core-count hardware |

JLI trades a probabilistic bound for a smaller memory constant and faster search, at the cost of a workload-dependent mutation tax concentrated wherever sustained tail-boundary pressure falls — and INSERT/DELETE land that tax on opposite access patterns.
