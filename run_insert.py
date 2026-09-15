#!/usr/bin/env python3
"""
final_eval.py - Run JLI vs skip-list benchmarks for any section
                 (BUILD, STATIC, INSERT, DELETE), aggregate per
                 (section, pattern, n) group across repeated process
                 calls, then combine everything into one final file.

Generalized from the original BUILD-only script. Key facts about
8.c baked in here (verified against source, not assumed):

  - BUILD only has pattern "sorted" and IGNORES --n: a single process
    call sweeps ALL of 8.c's hardcoded sizes
    [10000, 50000, 100000, 250000, 500000, 750000, 1000000] in one
    CSV append, unless --n is passed to pin it to exactly one size.
  - STATIC patterns: random, sequential, hotspot, zipfian, miss, adversarial
  - INSERT patterns: random, sequential, hotspot, zipfian, adversarial
  - DELETE patterns: random, sequential, hotspot, zipfian, adversarial
  - --build-params / --static-params / --insert-params / --delete-params
    take "segment_size,shortcuts_per_junction,max_skip_level" (3 values).
  - --build-full-params / --static-full-params / --insert-full-params /
    --delete-full-params take 13 comma values in this exact order:
        level_probability, local_interval, sub_interval, t_j,
        soft_pct, hard_pct, flagged_ratio_limit, min_seg_len_pct,
        max_suboptimal_segments, min_suboptimal_events_before_global,
        emergency_hard_segment_ratio, stop_crash_local_sub, block_size
    (level_probability is always silently forced to 0.5 inside 8.c
    regardless of what's passed - it's still required positionally.)
  - max_skip_level is always passed as 0 -> 8.c auto-derives it from
    n, segment_size, block_size.
  - Raw per-call CSV header (RAW_HEADER below) is written by 8.c itself
    via csv_header(); this script never writes it manually, just reads it.

Output CSV follows the exact column order of the aggregated header
(AGG_HEADER below).
"""

import subprocess
import csv
import os
import sys
import statistics
import argparse
from datetime import datetime

# Path to the compiled benchmark - adjust if needed
BENCH_EXE = "./bench" if os.name != "nt" else "./bench.exe"

# Sections 8.c understands, and the patterns valid for each.
SECTION_PATTERNS = {
    "BUILD":  ["sorted"],
    "STATIC": ["random", "sequential", "hotspot", "zipfian", "miss", "adversarial"],
    "INSERT": ["random", "sequential", "hotspot", "zipfian", "adversarial"],
    "DELETE": ["random", "sequential", "hotspot", "zipfian", "adversarial"],
}

# 8.c's hardcoded sweep sizes (used internally whenever --n is NOT pinned).
DEFAULT_SIZES = [10000, 50000, 100000, 250000, 500000, 750000, 1000000]

# Fixed core parameters - max_skip_level is 0 (auto)
CORE_PARAMS = {
    "segment_size": 128,
    "shortcuts_per_junction": 4,
    "max_skip_level": 0,  # 0 -> auto-calculate from n, segment_size, block_size
}

# Tuning parameters - matching 8.c's DEFAULT_TUNING, 13 values in parser order
TUNING_PARAMS = {
    "level_probability": 0.5,  # always 0.5; C parser overrides this anyway
    "local_interval": 1000,
    "sub_interval": 2500,
    "t_j": 0.30,
    "soft_pct": 0.25,
    "hard_pct": 0.50,
    "flagged_ratio_limit": 0.30,
    "min_seg_len_pct": 0.30,
    "max_suboptimal_segments": 0.25,
    "min_suboptimal_events_before_global": 5,
    "emergency_hard_segment_ratio": 0.95,
    "stop_crash_local_sub": 0.08,
    "block_size": 4,
}

# Raw per-call CSV header, exactly as written by 8.c's csv_header().
RAW_HEADER = [
    "section", "pattern", "n", "nops", "param_set", "label",
    "segment_size", "shortcuts_per_junction", "max_skip_level", "level_probability",
    "local_interval", "sub_interval", "t_j", "soft_pct", "hard_pct",
    "flagged_ratio_limit", "min_seg_len_pct", "max_suboptimal_segments",
    "min_suboptimal_events_before_global", "emergency_hard_segment_ratio",
    "stop_crash_local_sub", "block_size",
    "jli_avg_us", "jli_min_us", "jli_max_us", "jli_p50_us", "jli_p95_us", "jli_p99_us",
    "jli_stddev_us", "jli_jitter_us", "jli_ci_lo_us", "jli_ci_hi_us",
    "sl_avg_us", "sl_min_us", "sl_max_us", "sl_p50_us", "sl_p95_us", "sl_p99_us",
    "sl_stddev_us", "sl_jitter_us", "sl_ci_lo_us", "sl_ci_hi_us",
    "ratio", "win_rate", "jli_mem_bytes", "sl_mem_bytes",
    "jli_steps_avg", "jli_steps_p99", "jli_dissolve_rate", "jli_rebuild_rate",
    "jli_local_scan", "jli_sub_scan", "jli_local_rebuild", "jli_sub_rebuild",
    "jli_global_rebuild", "jli_rebuild_node_touches", "jli_effective_cost",
    "jli_efficiency", "mem_ratio", "repetition", "seed", "base_seed",
    "repetitions_count", "section2", "pattern2", "n2", "nops2", "eval_id",
]

# Static parameter columns that must be constant within one aggregation group.
STATIC_KEYS = [
    "section", "pattern", "n", "nops",
    "segment_size", "shortcuts_per_junction", "max_skip_level",
    "level_probability", "local_interval", "sub_interval",
    "t_j", "soft_pct", "hard_pct", "flagged_ratio_limit",
    "min_seg_len_pct", "max_suboptimal_segments",
    "min_suboptimal_events_before_global", "emergency_hard_segment_ratio",
    "stop_crash_local_sub", "block_size",
]

# Metrics to mean/median across repetitions, in the exact order of the
# provided aggregated header.
METRIC_ORDER = [
    "base_seed", "eval_id",
    "jli_avg_us", "jli_ci_hi_us", "jli_ci_lo_us",
    "jli_dissolve_rate", "jli_effective_cost", "jli_efficiency",
    "jli_global_rebuild", "jli_jitter_us", "jli_local_rebuild",
    "jli_local_scan", "jli_max_us", "jli_mem_bytes", "jli_min_us",
    "jli_p50_us", "jli_p95_us", "jli_p99_us",
    "jli_rebuild_node_touches", "jli_rebuild_rate", "jli_stddev_us",
    "jli_steps_avg", "jli_steps_p99", "jli_sub_rebuild", "jli_sub_scan",
    "label", "mem_ratio", "n2", "nops2", "param_set", "pattern2",
    "ratio", "repetition", "repetitions_count", "section2",
    "sl_avg_us", "sl_ci_hi_us", "sl_ci_lo_us",
    "sl_jitter_us", "sl_max_us", "sl_mem_bytes", "sl_min_us",
    "sl_p50_us", "sl_p95_us", "sl_p99_us", "sl_stddev_us",
    "win_rate",
]

AGG_HEADER = list(STATIC_KEYS) + ["seed"]
for _m in METRIC_ORDER:
    AGG_HEADER.append(f"mean_{_m}")
    AGG_HEADER.append(f"median_{_m}")


# -------------------------------------------------------------------------
def params_flag(section):
    """Return the --<section>-params flag name (lowercase section)."""
    return f"--{section.lower()}-params"


def full_params_flag(section):
    """Return the --<section>-full-params flag name (lowercase section)."""
    return f"--{section.lower()}-full-params"


def core_str(core=None):
    c = core if core is not None else CORE_PARAMS
    return (
        f"{c['segment_size']},"
        f"{c['shortcuts_per_junction']},"
        f"{c.get('max_skip_level', 0)}"
    )


def tuning_str(tuning=None):
    t = tuning if tuning is not None else TUNING_PARAMS
    return (
        f"{t.get('level_probability', 0.5):.6f},"
        f"{t['local_interval']},"
        f"{t['sub_interval']},"
        f"{t['t_j']:.6f},"
        f"{t['soft_pct']:.6f},"
        f"{t['hard_pct']:.6f},"
        f"{t['flagged_ratio_limit']:.6f},"
        f"{t['min_seg_len_pct']:.6f},"
        f"{t['max_suboptimal_segments']:.6f},"
        f"{t['min_suboptimal_events_before_global']},"
        f"{t['emergency_hard_segment_ratio']:.6f},"
        f"{t['stop_crash_local_sub']:.6f},"
        f"{t['block_size']}"
    )


# -------------------------------------------------------------------------
# Load per-row (section, pattern, n) -> params from a CSV like
# all_agg_final_phase3tuned.csv. Missing columns fall back to
# CORE_PARAMS / TUNING_PARAMS defaults. Used as-is, no re-optimization.
# -------------------------------------------------------------------------
def load_params_csv(path):
    param_map = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = row.get("section", "").strip().upper()
            pattern = row.get("pattern", "").strip()
            try:
                n = int(float(row.get("n", 0)))
            except (ValueError, TypeError):
                continue
            if not section or not pattern or not n:
                continue

            def _get_int(key, default):
                try:
                    return int(float(row[key]))
                except (KeyError, ValueError, TypeError):
                    return default

            def _get_float(key, default):
                try:
                    return float(row[key])
                except (KeyError, ValueError, TypeError):
                    return default

            core = {
                "segment_size": _get_int("segment_size", CORE_PARAMS["segment_size"]),
                "shortcuts_per_junction": _get_int("shortcuts_per_junction", CORE_PARAMS["shortcuts_per_junction"]),
                "max_skip_level": 0,
            }
            tuning = {
                "level_probability": 0.5,
                "local_interval": _get_int("local_interval", TUNING_PARAMS["local_interval"]),
                "sub_interval": _get_int("sub_interval", TUNING_PARAMS["sub_interval"]),
                "t_j": _get_float("t_j", TUNING_PARAMS["t_j"]),
                "soft_pct": _get_float("soft_pct", TUNING_PARAMS["soft_pct"]),
                "hard_pct": _get_float("hard_pct", TUNING_PARAMS["hard_pct"]),
                "flagged_ratio_limit": _get_float("flagged_ratio_limit", TUNING_PARAMS["flagged_ratio_limit"]),
                "min_seg_len_pct": _get_float("min_seg_len_pct", TUNING_PARAMS["min_seg_len_pct"]),
                "max_suboptimal_segments": _get_float("max_suboptimal_segments", TUNING_PARAMS["max_suboptimal_segments"]),
                "min_suboptimal_events_before_global": _get_int("min_suboptimal_events_before_global", TUNING_PARAMS["min_suboptimal_events_before_global"]),
                "emergency_hard_segment_ratio": _get_float("emergency_hard_segment_ratio", TUNING_PARAMS["emergency_hard_segment_ratio"]),
                "stop_crash_local_sub": _get_float("stop_crash_local_sub", TUNING_PARAMS["stop_crash_local_sub"]),
                "block_size": _get_int("block_size", TUNING_PARAMS["block_size"]),
            }
            nops = None
            if "nops" in row and row["nops"] not in ("", None):
                nops = _get_int("nops", None)

            param_map[(section, pattern, n)] = {"core": core, "tuning": tuning, "nops": nops}
    return param_map


# -------------------------------------------------------------------------
def build_cmd(section, pattern, n_keys, seed, out_file, nops=None, runs=30, core=None, tuning=None):
    """Return command line for bench.exe for a given section/pattern/n."""
    cmd = [
        BENCH_EXE,
        "--section", section,
        "--pattern", pattern,
        "--n", str(n_keys),
        "--runs", str(runs),
        "--seed", str(seed),
        "--no-table",
        "--single-csv",
        "--out", out_file,
        params_flag(section), core_str(core),
        full_params_flag(section), tuning_str(tuning),
    ]
    if nops is not None:
        cmd += ["--nops", str(nops)]
    return cmd


# -------------------------------------------------------------------------
def run_bench(section, pattern, n_keys, seed, out_file, nops=None, runs=30, core=None, tuning=None):
    """Run one process call and return CSV rows (or empty list on error)."""
    cmd = build_cmd(section, pattern, n_keys, seed, out_file, nops, runs, core=core, tuning=tuning)
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
        if result.returncode != 0:
            print(f"    Benchmark error (code {result.returncode}): {result.stderr[:200]}")
            return []
        if not os.path.exists(out_file):
            print(f"    Output file {out_file} was not created.")
            return []
        with open(out_file, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except subprocess.TimeoutExpired:
        print("    Timeout after 600s.")
        return []
    except Exception as e:
        print(f"    Unexpected error: {e}")
        return []


# -------------------------------------------------------------------------
def aggregate_reps(csv_files, output_agg, base_seed):
    """
    Collect the last row (summary) of each repetition file, then compute
    mean/median for all metrics. Output CSV follows AGG_HEADER exactly:
    static parameters first, then mean/median pairs per metric.
    """
    rows = []
    for f in csv_files:
        if not os.path.exists(f):
            continue
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
            if data:
                rows.append(data[-1])  # last row is the aggregated run
    if not rows:
        print("  No valid rows to aggregate.")
        return

    # Static values taken from the first row (must be identical across reps).
    static_vals = {k: rows[0].get(k, "") for k in STATIC_KEYS}
    static_vals["seed"] = base_seed

    agg_row = {}
    for metric in METRIC_ORDER:
        vals = []
        for row in rows:
            val = row.get(metric, "")
            if val == "":
                continue
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(val)
        if not vals:
            agg_row[f"mean_{metric}"] = float("nan")
            agg_row[f"median_{metric}"] = float("nan")
        elif all(isinstance(v, (int, float)) for v in vals):
            agg_row[f"mean_{metric}"] = statistics.mean(vals)
            agg_row[f"median_{metric}"] = statistics.median(vals)
        else:
            first = next((v for v in vals if v != ""), "")
            agg_row[f"mean_{metric}"] = first
            agg_row[f"median_{metric}"] = first

    out_row = {**static_vals}
    for metric in METRIC_ORDER:
        out_row[f"mean_{metric}"] = agg_row[f"mean_{metric}"]
        out_row[f"median_{metric}"] = agg_row[f"median_{metric}"]

    with open(output_agg, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_HEADER)
        writer.writeheader()
        writer.writerow(out_row)
    print(f"  Aggregated results saved to {output_agg}")


# -------------------------------------------------------------------------
def evaluate_group(section, pattern, n, nops, seed_base, output_dir,
                    repetitions=10, internal_runs=30, discard_first=True,
                    core=None, tuning=None):
    """
    Run `repetitions` process calls for one (section, pattern, n) group,
    discard the first rep if requested (cold-start), then aggregate.
    """
    size_label = f"{n // 1000}k" if n % 1000 == 0 else str(n)
    sub_dir = os.path.join(output_dir, section, pattern, size_label)
    os.makedirs(sub_dir, exist_ok=True)

    total_reps = repetitions + 1 if discard_first else repetitions
    rep_files = []
    for i in range(total_reps):
        seed = seed_base + i
        out_file = os.path.join(sub_dir, f"raw_run_{i}.csv")
        run_bench(section, pattern, n, seed, out_file, nops=nops, runs=internal_runs,
                  core=core, tuning=tuning)
        rep_files.append(out_file)

    if discard_first:
        discarded = rep_files[0]
        if os.path.exists(discarded):
            try:
                with open(discarded) as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        discarded_avg = rows[-1].get("jli_avg_us", "n/a")
                        print(f"  Discarding rep 0 (cold start), jli_avg_us={discarded_avg}")
                    os.remove(discarded)
            except Exception:
                pass
        rep_files = rep_files[1:]

    aggregate_rep_file = os.path.join(sub_dir, "aggregated.csv")
    aggregate_reps(rep_files, aggregate_rep_file, seed_base)
    return aggregate_rep_file


# -------------------------------------------------------------------------
def combine_all_aggregated(output_dir, agg_files):
    """Merge every group's aggregated.csv into a single combined_agg.csv."""
    combined_rows = []
    for agg_file in agg_files:
        if not os.path.exists(agg_file):
            print(f"  Warning: missing {agg_file}, skipping.")
            continue
        with open(agg_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                combined_rows.append(row)

    if not combined_rows:
        print("No aggregated files found; nothing to combine.")
        return

    out_path = os.path.join(output_dir, "combined_agg.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_HEADER)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"\nCombined aggregated file written to {out_path}")


# -------------------------------------------------------------------------
def resolve_sizes(section, sizes_arg):
    """
    BUILD ignores --n internally if you don't pin it (it sweeps all
    DEFAULT_SIZES in one call), but for this script we always pin --n
    explicitly so every (section, pattern, n) group gets its own clean
    set of repeated process calls and its own aggregation file.
    """
    if sizes_arg:
        return sizes_arg
    return DEFAULT_SIZES


def resolve_patterns(section, patterns_arg):
    valid = SECTION_PATTERNS[section]
    if not patterns_arg:
        return valid
    bad = [p for p in patterns_arg if p not in valid]
    if bad:
        print(f"ERROR: pattern(s) {bad} not valid for section {section}. "
              f"Valid patterns: {valid}")
        sys.exit(1)
    return patterns_arg


# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JLI vs skip-list final evaluation")
    parser.add_argument(
        "--section", default="ALL",
        choices=["ALL", "BUILD", "STATIC", "INSERT", "DELETE"],
        help="Which benchmark section(s) to run (default ALL)",
    )
    parser.add_argument("--patterns", nargs="*", default=None,
                         help="Restrict to these patterns (default: all valid for section)")
    parser.add_argument("--sizes", nargs="*", type=int, default=None,
                         help="Restrict to these n values (default: 8.c's hardcoded sweep sizes)")
    parser.add_argument("--nops", type=int, default=None,
                         help="Operations per timing run for STATIC/INSERT/DELETE "
                              "(default: n/3.5, matching 8.c's own default behavior)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (incremented per rep)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--reps", type=int, default=10,
                         help="Number of process repetitions (after discarding first)")
    parser.add_argument("--runs", type=int, default=30, help="Internal --runs per process")
    parser.add_argument("--no-discard", action="store_true",
                         help="Do not discard first repetition (skip cold-start handling)")
    parser.add_argument("--params-csv", default=None,
                         help="CSV providing per-(section,pattern,n) segment_size/"
                              "shortcuts_per_junction/block_size/tuning params, used "
                              "AS-IS (no re-optimization/search). Rows not found in "
                              "the CSV fall back to this script's hardcoded "
                              "CORE_PARAMS/TUNING_PARAMS. If given, --sizes defaults "
                              "to the n values present in the CSV for the selected "
                              "section(s) instead of the 8.c sweep.")
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"final_eval_{args.section}_{ts}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(BENCH_EXE):
        print(f"ERROR: {BENCH_EXE} not found. Compile 8.c and place the executable here.")
        sys.exit(1)

    sections = ["BUILD", "STATIC", "INSERT", "DELETE"] if args.section == "ALL" else [args.section]

    param_map = {}
    if args.params_csv:
        if not os.path.isfile(args.params_csv):
            print(f"ERROR: --params-csv not found: {args.params_csv}")
            sys.exit(1)
        param_map = load_params_csv(args.params_csv)
        print(f"Loaded {len(param_map)} (section,pattern,n) param row(s) from {args.params_csv} (used as-is, no optimization)")

    print(f"Evaluation with reps={args.reps}, internal runs={args.runs}, sections={sections}")
    print(f"Output directory: {output_dir}")

    agg_files = []
    for section in sections:
        patterns = resolve_patterns(section, args.patterns)

        if args.sizes:
            sizes = args.sizes
        elif args.params_csv:
            csv_sizes = sorted({n for (sec, pat, n) in param_map if sec == section})
            sizes = csv_sizes if csv_sizes else resolve_sizes(section, None)
        else:
            sizes = resolve_sizes(section, None)

        # nops only matters for non-BUILD sections; BUILD always sets nops = n internally.
        base_nops = None if section == "BUILD" else args.nops

        for pattern in patterns:
            for n in sizes:
                row = param_map.get((section, pattern, n))
                core = row["core"] if row else None
                tuning = row["tuning"] if row else None
                nops = base_nops if base_nops is not None else (row["nops"] if row else None)

                src = "params-csv" if row else "script defaults"
                print(f"\nEvaluating section={section} pattern={pattern} n={n}  (params: {src})")
                agg_file = evaluate_group(
                    section, pattern, n, nops, args.seed, output_dir,
                    repetitions=args.reps,
                    internal_runs=args.runs,
                    discard_first=not args.no_discard,
                    core=core, tuning=tuning,
                )
                agg_files.append(agg_file)

    combine_all_aggregated(output_dir, agg_files)


if __name__ == "__main__":
    main()