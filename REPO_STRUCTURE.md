# Junction-Linked List (JLI) — FSTTCS Submission No. 8

This repository contains the reference implementation, benchmark harness, parameter
search, and result-aggregation scripts for the Junction-Linked List (JLI), evaluated
against a textbook probabilistic skip list. It accompanies FSTTCS submission no. 8.

## Repository contents

| File / folder | Purpose |
|---|---|
| `jli_v8_1.c` | The JLI data structure (v8.1) **and** the reference probabilistic skip list used as its baseline. This is a header-style `.c` file included directly by the benchmark (see below) — it is not compiled or run on its own. |
| `8.c` | The benchmark harness. Includes `jli_v8_1.c` and runs the four benchmark sections (`BUILD`, `STATIC`, `INSERT`, `DELETE`) across configurable sizes, access patterns, and structural/tuning parameters, emitting CSV output. |
| `parameter_search.py` | Hierarchical random-search driver that tunes JLI's structural and maintenance parameters (segment size, shortcut budget, block size, drift thresholds, maintenance intervals, etc.) against `8.c`, optimizing solely for search-latency ratio. |
| `run_insert.py` | Runs `8.c` with a **fixed, pre-determined** set of parameters over a fixed set of (section, pattern, n) configurations — i.e. it replays an already-chosen configuration rather than searching for one. Used both for standard final-evaluation runs and for the long-run maintenance ablation (see below). |
| `collect_row.py` | Recursively walks a results directory tree, collects every `raw_run_*.csv` and `aggregated.csv` file it finds, infers `(section, pattern, n)` metadata from the folder names, and writes two consolidated CSVs: `all_raw_runs.csv` and `all_aggregated.csv`. For each configuration it also computes the JLI/skip-list mean ratio, a 95% CI, and two non-parametric significance tests (Wilcoxon signed-rank, sign test) directly from the raw per-run data, adding these as extra columns on the aggregated output. |
| `maintaince/`, `no_maintaince/` | Raw/aggregated results from the long-run maintenance ablation (Section 5 / Appendix H of the paper): `nops = 40n` INSERT runs at `n = 50,000`, with the three-tier deferred maintenance system enabled (`maintaince/`) vs. disabled by setting `local_interval`/`sub_interval` above the run length (`no_maintaince/`). *(Note: both folder names are missing an "n" — `maintenance`/`no_maintenance` — as currently committed; see note above.)* |
| `all_aggregated.csv`, `all_raw_runs.csv`, `all_ratio.csv` | Pre-generated consolidated result files, as produced by `collect_row.py` (plus a ratio-only view), corresponding to the main 87-configuration benchmark reported in the paper. |
| `machine_2/` | Second-machine replication of the main benchmark, using the primary bench's already-tuned parameters (not an independent parameter search). See the dedicated note near the end of this README. |

## Build

```bash
gcc -O2 -o bench 8.c -lm
```

This produces a `bench` (or `bench.exe` on Windows) executable. `jli_v8_1.c` does not
need to be compiled separately — `8.c` pulls it in via `#include "jli_v8_1.c"`.

## Usage

### 1. Run the benchmark directly

```bash
./bench --section STATIC --pattern adversarial --n 500000 --runs 30 --nops 5000 --seed 42
```

Key flags (see `./bench --help` for the full list):

- `--section BUILD|STATIC|INSERT|DELETE|ALL`
- `--pattern <name>` — restrict to one access pattern (`random`, `sequential`, `hotspot`, `zipfian`, `miss`, `adversarial`, as applicable to the section)
- `--n N` — run only a specific problem size (otherwise sweeps `8.c`'s hardcoded sizes)
- `--runs N` / `--nops N` — internal repetition count / operations per timed run
- `--build-params`, `--static-params`, `--insert-params`, `--delete-params` — override `segment_size,shortcuts_per_junction,max_skip_level` (3 comma-separated integers) per section
- `--build-full-params`, `--static-full-params`, `--insert-full-params`, `--delete-full-params` — override the full 12/13-value maintenance/tuning parameter tuple per section
- `--out FILE.csv`, `--seed S`, `--warmup N`

### 2. Search for tuned parameters

```bash
python3 parameter_search.py --section INSERT --pattern zipfian --n-ref 500000 \
    --target-sizes 50000 100000 250000 1000000 --trials 200 --refine
```

Runs a hierarchical random search (reference-size search, then transfer + local
refinement at each target size) against the compiled `bench` executable, optimizing
purely for search-latency ratio (memory is never a search objective — see the paper's
own discussion of this scope in Sections 6/9).

### 3. Re-run a fixed, already-chosen configuration

```bash
python3 run_insert.py --section INSERT --patterns adversarial zipfian \
    --sizes 50000 100000 250000 500000 1000000 --reps 10 --runs 30 \
    --params-csv <tuned_params>.csv --output-dir final_eval_INSERT
```

This is the script used both for standard final-evaluation runs (feeding in parameters
already found by `parameter_search.py`) and for the long-run maintenance ablation runs
that produced `maintaince/` and `no_maintaince/` (by fixing `local_interval`/`sub_interval`
above the run length to disable maintenance in the latter case).

### 4. Consolidate results

```bash
python3 collect_row.py --input-dir final_eval_INSERT \
    --output-raw all_raw_runs.csv --output-agg all_aggregated.csv
```

Recursively finds every `raw_run_*.csv` / `aggregated.csv` under `--input-dir`, infers
`(section, pattern, n)` from folder-naming conventions (`Results_<SECTION>_<pattern>_<timestamp>/<n>k/`
or `final_eval_<SECTION>_<timestamp>/<pattern>_<n>k/`), and writes the two consolidated
CSVs plus per-configuration ratio/CI/significance-test columns. Requires `scipy` for the
CI and hypothesis-test columns (falls back to a normal approximation for the CI and
leaves the p-value columns as NaN if `scipy` is unavailable).

## Reproducing the paper's headline results

The tuned parameters this paper reports are already baked into the CSVs committed in
this repo, so reproduction does **not** require re-running `parameter_search.py` from
scratch — it means replaying those already-found parameters through `run_insert.py`.

1. Build `bench` as above.
2. **Main 87-configuration result:** pass `all_aggregated.csv` directly as
   `run_insert.py`'s `--params-csv`. Each row already carries the tuned
   `segment_size` / `shortcuts_per_junction` / `block_size` / maintenance-tuning values
   found for that exact `(section, pattern, n)`, so `run_insert.py` uses them as-is with
   no re-optimization and reproduces that configuration's raw/aggregated results:

   ```bash
   python3 run_insert.py --section ALL --params-csv all_aggregated.csv \
       --reps 10 --runs 30 --output-dir reproduce_main
   ```

   (`--sizes`/`--patterns` don't need to be passed — with `--params-csv` given,
   `run_insert.py` defaults to exactly the `(section, pattern, n)` rows present in that
   CSV.)

3. **Long-run maintenance ablation:** do the same with the two ablation CSVs, once
   against the `maintaince/` results CSV and once against `no_maintaince/`'s, to replay
   the `nops = 40n`, `n = 50000` INSERT runs with maintenance enabled vs. disabled:

   ```bash
   python3 run_insert.py --section INSERT --params-csv maintaince/aggregated.csv \
       --reps 10 --runs 30 --output-dir reproduce_maintenance
   python3 run_insert.py --section INSERT --params-csv no_maintaince/aggregated.csv \
       --reps 10 --runs 30 --output-dir reproduce_no_maintenance
   ```

   (Adjust the CSV filename above to whatever the actual aggregated file is named inside
   each folder.)

4. Run `collect_row.py` over each resulting output tree if you want the same consolidated
   `all_raw_runs.csv` / `all_aggregated.csv` format (with ratio/CI/significance columns)
   as this repo ships, rather than just `run_insert.py`'s own per-run output.

## Requirements

- GCC (tested with GCC 15.2.0) with C11 support; `-lm` for `math.h` functions.
- Python 3 with `scipy` installed (optional but recommended — needed for the Wilcoxon/
  sign-test p-value columns and the t-distribution-based CI in `collect_row.py`).

## Note on `machine_2/` (second-machine results)

`machine_2/all_aggregated_search.csv` and `machine_2/all_raw_runs_search.csv` were
generated on a **second machine**, distinct from the primary benchmarking machine
described under Requirements/Experimental Setup above. These are not a fresh,
independently-tuned run: they were produced by feeding the **primary bench's
already-tuned parameters** into `run_insert.py` on the second machine, under the same
repetition protocol/rigor as the primary bench (30 internal timed runs per process
repetition, aggregated across 10 process repetitions with the first discarded as cold
start). In other words, this is a cross-machine replication of the primary bench's
configurations, used to sanity-check that the reported results aren't an artifact of a
single machine, not a separate parameter search.
