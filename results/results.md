# Results

This document walks through the empirical section of the paper (FSTTCS submission no. 8),
built directly from `all_aggregated.csv` (the 87-configuration main benchmark) and
`all_raw_runs.csv`. Every figure below was recomputed from those CSVs, not copied from
the PDF — they match the paper's reported numbers exactly.

**Setup, in one line:** JLI is benchmarked against a textbook probabilistic skip list
(p = 0.5) across four operation classes — `BUILD`, `STATIC` (search), `INSERT`, `DELETE`
— for 87 total `(section, pattern, n)` configurations. All ratios below are
**JLI ÷ skip-list**, so values below 1.0 favor JLI. Charts are in `charts/`.

## Contents

- [Memory](#memory)
- [Search (STATIC)](#search-static)
- [INSERT](#insert)
- [DELETE](#delete)
- [Cross-device check](#cross-device-check)
- [Statistical significance](#statistical-significance)
- [Summary](#summary)

---

## Memory

![Memory overhead across all 87 configurations](charts/memory_all_configs.png)

Across all 87 configurations, JLI's structural index overhead ranged from **75.04% to
80.41%** of the skip list's (mean 77.8%), with zero exceptions. `BUILD` — the one
section run at a single fixed, untuned configuration — shows the tightest band (77.8–78.0%
across all seven sizes), which is the cleanest evidence the memory win is structural
rather than an artifact of the parameter search favoring memory-friendly configs:

![BUILD memory ratio and DELETE-adversarial vs sequential](charts/build_and_delete_extremes.png)

*(left panel)* — BUILD's ratio, 10K through 1M.

---

## Search (STATIC)

![STATIC latency ratio by pattern](charts/static_latency_by_pattern.png)

JLI's mean search latency was lower than the skip list's in all 30 STATIC configurations
(6 patterns × 5 sizes) — no reversals.

| Pattern | Ratio range (JLI/SL) | Speedup |
|---|---|---|
| `adversarial` (guaranteed miss past tail) | 0.257 – 0.373 | **2.68× – 3.90×** |
| `sequential` | 0.758 – 0.891 | 1.12× – 1.32× |
| `zipfian` | 0.807 – 0.895 | 1.12× – 1.24× |
| `random` | 0.832 – 0.869 | 1.15× – 1.20× |
| `hotspot` | 0.848 – 0.875 | 1.14× – 1.18× |
| `miss` (worst case) | 0.891 – 0.936 | **1.07× – 1.12×** |

`adversarial` is the standout: a guaranteed-miss query past the tail is exactly where
JLI's shortcut/junction descent is cheapest relative to a skip list's tower walk.

**Win rate vs. mean advantage.** These are separate signals — mean latency being lower
doesn't mean JLI wins every paired run:

![STATIC win-rate heatmap](charts/static_winrate_heatmap.png)

`adversarial` has a perfect win rate (1.00) at every size. `miss` has both the smallest
mean advantage *and* the widest, most volatile win rate (0.70–0.99).

---

## INSERT

![INSERT latency ratio by pattern](charts/insert_latency_by_pattern.png)

Averaged with sign across all sizes, INSERT is **+7.28% slower** than the skip list
(median +5.41%). That average hides a sharp pattern split:

| Pattern | Ratio range (JLI/SL) | Verdict |
|---|---|---|
| `adversarial` (tail-append) | 0.851 – 0.972 | Faster — cheapest possible locate phase |
| `random` | 0.885 – 1.006 | Roughly parity or faster |
| `hotspot` | 0.952 – 1.188 | Slower, worsening with n |
| `sequential` | 1.048 – 1.296 | Slower, worsening with n |
| `zipfian` (worst case) | 1.112 – 1.357 | Slower, up to 1.36× |

`adversarial` — sustained tail append past the current maximum — is INSERT's *best*
pattern, not its worst: every insert lands at the tail, the cheapest possible locate for
JLI's descent, even though that same tail-directed growth drives the highest maintenance
rebuild volume behind the scenes (see the paper's Section 5 ablation).

---

## DELETE

![DELETE latency ratio by pattern](charts/delete_latency_by_pattern.png)

DELETE nets **+4.31% slower** on average (median +2.51%) — a smaller tax than INSERT's,
but with its own sharp outlier:

| Pattern | Ratio range (JLI/SL) | Verdict |
|---|---|---|
| `sequential` | 0.854 – 0.939 | Faster |
| `random` | 0.932 – 1.007 | Roughly parity |
| `zipfian` | 0.939 – 1.064 | Near parity |
| `hotspot` | 1.017 – 1.105 | Slightly slower |
| `adversarial` (tail-delete) | up to **1.27× worse on average** | Worst pattern overall |

DELETE mirrors INSERT: `sequential` (vacating a segment in list-walk order, keeping
locate cheap even as dissolutions fire) is the best pattern here, while `adversarial`
(deleting from the tail backward, draining the rightmost junction toward its dissolution
floor) is the worst — peaking at 1.33× slower. INSERT's and DELETE's worst cases sit on
opposite sides of the same tail boundary.

![DELETE adversarial vs sequential detail](charts/build_and_delete_extremes.png)

*(right panel)* — DELETE-adversarial climbs and stays above parity at every size;
DELETE-sequential stays comfortably below it.

**Why the tax exists:** sustained mutation concentrated at one boundary forces the
three-tier maintenance system to fire repeatedly, and each firing pays a fixed O(B)
rebuild-floor cost that doesn't amortize away as n grows — the paper's Section 4.1 proves
this analytically rather than just observing it empirically.

---

## Cross-device check

![Cross-device comparison](charts/cross_device.png)

STATIC was independently re-run in full on a second, lower-core-count machine (see
`machine_2/` in this repo), replaying the primary bench's already-tuned parameters rather
than re-searching them.

- **Memory replicated almost exactly** — 75.1–79.2% on the secondary machine vs.
  75.0–80.4% on the primary. Strong evidence the memory advantage is hardware-independent.
- **Latency direction held, magnitude did not.** `adversarial` stays decisively ahead on
  both machines. But `random` and `sequential` both cross above parity at n = 1,000,000 on
  the secondary machine, where the primary stayed below parity throughout — four of six
  patterns reverse at the largest size on the lower-core-count device.

The paper flags this directly: the search-latency win is more hardware-sensitive than a
single-machine result would suggest; the memory win is not.

---

## Statistical significance

`all_aggregated.csv` includes, per configuration, the mean JLI/skip-list ratio, its 95%
CI, a Wilcoxon signed-rank p-value, and a sign-test p-value — computed by `collect_row.py`
from the raw per-run data in `all_raw_runs.csv`:

```
ratio_ci_low, ratio_ci_high, wilcoxon_p, sign_p
```

Spot-check, `BUILD` at n = 1M: mean ratio 0.459 (95% CI 0.446–0.472), sign-test
p ≈ 0.002 — consistent with the tight confidence intervals seen throughout the memory and
STATIC results, and with an effect too large and consistent across 30 internal runs × 10
process repetitions to be noise.

---

## Summary

| Dimension | Result |
|---|---|
| Memory | Unconditionally better — 75.0–80.4% of skip-list overhead, 0/87 exceptions |
| Search (STATIC) | Unconditionally better — lower mean latency in 30/30 configurations, up to 3.9× |
| INSERT | +7.28% slower on average — faster under `adversarial`/`random`, worse under `zipfian`/`sequential` |
| DELETE | +4.31% slower on average — faster under `sequential`/`random`, much worse under `adversarial` |
| Cross-device | Memory advantage is hardware-independent; latency advantage narrows/reverses at n = 1M on lower-core-count hardware |

JLI trades a probabilistic search bound for a smaller memory constant and faster search,
at the cost of a workload-dependent mutation tax concentrated wherever sustained
tail-boundary pressure falls — and INSERT/DELETE land that tax on opposite access patterns.
