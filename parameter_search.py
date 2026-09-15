#!/usr/bin/env python3
"""
hierarchical_search_corrected.py – Hierarchical parameter search for JLI (v8, includes block_size).
Reference 100k, targets 50k & 250k.  Final evaluation with warm‑up and aggregation.
Now captures ALL JLI metrics from the CSV (including local_scan, sub_scan, rebuilds, node_touches).

IMPORTANT: max_skip_level is ALWAYS set to 0 (auto-calculated) – it is NOT a tunable parameter.
"""

import subprocess
import csv
import os
import sys
import random
import argparse
import math
from datetime import datetime
import statistics

# ------------------------------------------------------------
# Benchmark executable
# ------------------------------------------------------------
BENCH_EXE = "./bench" if os.name != "nt" else "./bench.exe"

# ------------------------------------------------------------
# Parameter ranges – includes block_size
# ------------------------------------------------------------
PARAM_RANGES = {
    # NOTE (500k-ref update): segment_size, shortcuts_per_junction, and
    # block_size were saturating at the top of their old ranges during the
    # 50k/100k/250k search (best values kept landing on 4096 / 16 / 16).
    # That means the true optimum for larger n was never actually explored -
    # the search could only report "biggest available value wins" without
    # ever testing anything bigger. Ranges below are widened so the search at
    # n_ref=500000 (and transfer to 1,000,000) has room to find a real optimum
    # instead of hitting a ceiling artifact.
    'segment_size': [256, 512, 1024, 2048, 4096, 8192, 16384],
    'shortcuts_per_junction': [2, 4, 6, 8, 10, 12, 16, 20, 24, 32],
    # level_probability is fixed at 0.5 for both JLI and skip list.
    # It is not a search variable; removing it from the search space
    # ensures a fair, principled comparison.
    # local_interval/sub_interval ranges widened at the top end to give
    # scale_params()'s ratio**0.3 scaling room to move for n up to 1M
    # (at n_ref=500000, ratio=1M/500k=2 -> interval * 2**0.3 ~= 1.23x;
    # from the historical n_ref=100000 baseline, ratio=10 -> 10**0.3 ~= 2x).
    'local_interval': list(range(500, 3001, 100)),
    'sub_interval': list(range(1000, 4001, 200)),
    # The percentage/ratio-style parameters below were empirically
    # scale-invariant across 50k/100k/250k in all_agg_final.csv (their
    # observed min/max bands stayed essentially the same at every n), so
    # they are left unchanged for the 500k/1M search.
    't_j': [0.15, 0.20, 0.25, 0.30, 0.35],
    'soft_pct': [0.15, 0.20, 0.25, 0.30],
    'hard_pct': [0.35, 0.45, 0.55, 0.65],
    'flagged_ratio_limit': [0.22, 0.25, 0.29, 0.32, 0.36],
    'min_seg_len_pct': [0.15, 0.18, 0.20, 0.22, 0.25],
    'max_suboptimal_segments': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    'min_suboptimal_events_before_global': [2, 3, 4, 5, 6],
    'emergency_hard_segment_ratio': [0.75, 0.80, 0.85, 0.90, 0.95],
    'stop_crash_local_sub': [0.04, 0.05, 0.06, 0.07, 0.08],
    'block_size': [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24],
}

DEFAULTS = {
    'level_probability': 0.5,
    'local_interval': 1050,
    'sub_interval': 1700,
    't_j': 0.212,
    'soft_pct': 0.212,
    'hard_pct': 0.462857,
    'flagged_ratio_limit': 0.291429,
    'min_seg_len_pct': 0.199,
    'max_suboptimal_segments': 0.3,
    'min_suboptimal_events_before_global': 4,
    'emergency_hard_segment_ratio': 0.86943,
    'stop_crash_local_sub': 0.051,
    'block_size': 4,
}

# Set of parameters that are integral (max_skip_level is NOT included)
INT_PARAMS = {
    'segment_size', 'shortcuts_per_junction', 'local_interval', 'sub_interval',
    'min_suboptimal_events_before_global', 'block_size'
}

# Floating‑point parameters that must lie in [0, 1]
FLOAT_0_1 = {
    'level_probability', 't_j', 'soft_pct', 'hard_pct',
    'flagged_ratio_limit', 'min_seg_len_pct', 'emergency_hard_segment_ratio',
    'stop_crash_local_sub', 'max_suboptimal_segments'
}

# ------------------------------------------------------------
# Constraint enforcement helper
# ------------------------------------------------------------
def fixup_params(params):
    """Adjust params to satisfy:
         local_interval < sub_interval
         soft_pct < hard_pct
         all float params in [0, 1]
    Also ensures max_skip_level is 0 (auto) – but we don't store it.
    """
    # local_interval < sub_interval
    if 'local_interval' in params and 'sub_interval' in params:
        if params['local_interval'] >= params['sub_interval']:
            params['local_interval'], params['sub_interval'] = params['sub_interval'], params['local_interval']
            if params['local_interval'] >= params['sub_interval']:
                params['sub_interval'] = params['local_interval'] + 1

    # soft_pct < hard_pct
    if 'soft_pct' in params and 'hard_pct' in params:
        if params['soft_pct'] >= params['hard_pct']:
            params['soft_pct'], params['hard_pct'] = params['hard_pct'], params['soft_pct']
            if params['soft_pct'] >= params['hard_pct']:
                params['hard_pct'] = min(1.0, params['soft_pct'] + 0.01)

    # clamp floats to [0,1]
    for key in FLOAT_0_1:
        if key in params:
            v = params[key]
            params[key] = max(0.0, min(1.0, float(v)))

    # ensure integer types for known int params
    for key in INT_PARAMS:
        if key in params:
            params[key] = int(params[key])

    # Force max_skip_level to 0 (auto) – it is not a tunable parameter.
    params['max_skip_level'] = 0

    return params

# ------------------------------------------------------------
# Helper: convert a row from CSV into a parameter dict
# ------------------------------------------------------------
def row_to_params(row, section):
    params = {}
    if 'segment_size' in row and 'shortcuts_per_junction' in row:
        params['segment_size'] = int(float(row['segment_size']))
        params['shortcuts_per_junction'] = int(float(row['shortcuts_per_junction']))
        params['level_probability'] = 0.5  # fixed: p=0.5 for both structures
        params['local_interval'] = int(float(row.get('local_interval', DEFAULTS['local_interval'])))
        params['sub_interval'] = int(float(row.get('sub_interval', DEFAULTS['sub_interval'])))
        params['t_j'] = float(row.get('t_j', DEFAULTS['t_j']))
        params['soft_pct'] = float(row.get('soft_pct', DEFAULTS['soft_pct']))
        params['hard_pct'] = float(row.get('hard_pct', DEFAULTS['hard_pct']))
        params['flagged_ratio_limit'] = float(row.get('flagged_ratio_limit', DEFAULTS['flagged_ratio_limit']))
        params['min_seg_len_pct'] = float(row.get('min_seg_len_pct', DEFAULTS['min_seg_len_pct']))
        params['max_suboptimal_segments'] = float(row.get('max_suboptimal_segments', DEFAULTS['max_suboptimal_segments']))
        params['min_suboptimal_events_before_global'] = int(float(row.get('min_suboptimal_events_before_global', DEFAULTS['min_suboptimal_events_before_global'])))
        params['emergency_hard_segment_ratio'] = float(row.get('emergency_hard_segment_ratio', DEFAULTS['emergency_hard_segment_ratio']))
        params['stop_crash_local_sub'] = float(row.get('stop_crash_local_sub', DEFAULTS['stop_crash_local_sub']))
        params['block_size'] = int(float(row.get('block_size', DEFAULTS['block_size'])))
    elif 'seg' in row and 'sc' in row:
        # Fallback format
        params['segment_size'] = int(float(row['seg']))
        params['shortcuts_per_junction'] = int(float(row['sc']))
        params['level_probability'] = 0.5  # fixed: p=0.5 for both structures
        params['local_interval'] = int(float(row.get('r1', DEFAULTS['local_interval'])))
        params['sub_interval'] = int(float(row.get('r2', DEFAULTS['sub_interval'])))
        params['t_j'] = float(row.get('t_j', DEFAULTS['t_j']))
        params['soft_pct'] = float(row.get('soft_pct', DEFAULTS['soft_pct']))
        params['hard_pct'] = float(row.get('hard_pct', DEFAULTS['hard_pct']))
        params['flagged_ratio_limit'] = float(row.get('flagged_ratio', DEFAULTS['flagged_ratio_limit']))
        params['min_seg_len_pct'] = float(row.get('min_seg_len_pct', DEFAULTS['min_seg_len_pct']))
        params['max_suboptimal_segments'] = float(row.get('max_suboptimal_segments', DEFAULTS['max_suboptimal_segments']))
        params['min_suboptimal_events_before_global'] = int(float(row.get('min_events', DEFAULTS['min_suboptimal_events_before_global'])))
        params['emergency_hard_segment_ratio'] = float(row.get('emerg_hard_ratio', DEFAULTS['emergency_hard_segment_ratio']))
        params['stop_crash_local_sub'] = float(row.get('stop_crash', DEFAULTS['stop_crash_local_sub']))
        params['block_size'] = random.choice(PARAM_RANGES['block_size'])
    else:
        raise ValueError("Unknown CSV format: missing required columns")

    if section == "STATIC":
        for p in ['local_interval', 'sub_interval', 't_j', 'soft_pct', 'hard_pct',
                  'flagged_ratio_limit', 'min_seg_len_pct', 'max_suboptimal_segments',
                  'min_suboptimal_events_before_global', 'emergency_hard_segment_ratio',
                  'stop_crash_local_sub']:
            if p in params:
                params[p] = DEFAULTS[p]

    fixup_params(params)
    return params

# ------------------------------------------------------------
# Load seed rows from CSV files
# ------------------------------------------------------------
def load_seed_rows(section, pattern, target_n, primary_csv="filtered_lowest_tp_speedup_1.csv",
                   fallback_csv="filtered_lowest_tp_speedup.csv"):
    seeds = []
    required_n = target_n

    def try_load(csv_path, is_primary):
        if not os.path.exists(csv_path):
            return
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_val = int(float(row.get('n', 0)))
                if n_val != required_n:
                    continue
                row_section = row.get('section', '').strip().upper()
                if row_section != section:
                    continue
                row_pattern = row.get('pattern', '').strip()
                if row_pattern != pattern:
                    continue
                try:
                    params = row_to_params(row, section)
                    seeds.append(params)
                except Exception as e:
                    print(f"Warning: could not parse row from {csv_path}: {e}")
                    continue
    try_load(primary_csv, True)
    try_load(fallback_csv, False)
    return seeds

# ------------------------------------------------------------
# Random parameter generation
# ------------------------------------------------------------
def random_params(section, seed_pool=None):
    if seed_pool and random.random() < 0.3:
        return random.choice(seed_pool).copy()
    params = {}
    params['segment_size'] = random.choice(PARAM_RANGES['segment_size'])
    params['shortcuts_per_junction'] = random.choice(PARAM_RANGES['shortcuts_per_junction'])
    params['level_probability'] = 0.5  # fixed: p=0.5 for both structures
    params['block_size'] = random.choice(PARAM_RANGES['block_size'])

    if section == "STATIC":
        for p in DEFAULTS:
            if p not in params:
                params[p] = DEFAULTS[p]
    else:
        for p in PARAM_RANGES:
            if p not in params:
                if p in ['local_interval', 'sub_interval']:
                    params[p] = random.choice(PARAM_RANGES[p])
                elif p in ['t_j', 'soft_pct', 'hard_pct', 'flagged_ratio_limit',
                           'min_seg_len_pct', 'emergency_hard_segment_ratio', 'stop_crash_local_sub']:
                    params[p] = random.choice(PARAM_RANGES[p])
                elif p in ['max_suboptimal_segments', 'min_suboptimal_events_before_global']:
                    params[p] = random.choice(PARAM_RANGES[p])
                elif p == 'block_size':
                    params[p] = random.choice(PARAM_RANGES[p])
    fixup_params(params)  # this also sets max_skip_level=0
    return params

# ------------------------------------------------------------
# Build command line for bench (with optional warmup)
# max_skip_level is always 0 (auto)
# ------------------------------------------------------------
def build_cmd(section, pattern, n_keys, nops, seed, params, tmp_file, warmup_ops=0, runs=1):
    # core_str always uses 0 for max_skip_level
    core_str = f"{params['segment_size']},{params['shortcuts_per_junction']},0"
    tuning_str = (
        f"{params['level_probability']:.6f},"
        f"{params['local_interval']},"
        f"{params['sub_interval']},"
        f"{params['t_j']:.6f},"
        f"{params['soft_pct']:.6f},"
        f"{params['hard_pct']:.6f},"
        f"{params['flagged_ratio_limit']:.6f},"
        f"{params['min_seg_len_pct']:.6f},"
        f"{params['max_suboptimal_segments']},"
        f"{params['min_suboptimal_events_before_global']},"
        f"{params['emergency_hard_segment_ratio']:.6f},"
        f"{params['stop_crash_local_sub']:.6f},"
        f"{params['block_size']}"
    )
    cmd = [
        BENCH_EXE,
        "--section", section,
        "--pattern", pattern,
        "--n", str(n_keys),
        "--nops", str(nops),
        "--runs", str(runs),
        "--seed", str(seed),
        "--no-table",
        "--single-csv",
        "--out", tmp_file,
    ]
    if warmup_ops > 0:
        cmd += ["--warmup", str(warmup_ops)]
    if section == "STATIC":
        cmd += ["--static-params", core_str, "--static-full-params", tuning_str]
    elif section == "INSERT":
        cmd += ["--insert-params", core_str, "--insert-full-params", tuning_str]
    elif section == "DELETE":
        cmd += ["--delete-params", core_str, "--delete-full-params", tuning_str]
    else:
        raise ValueError(f"Unknown section: {section}")
    return cmd

# ------------------------------------------------------------
# Evaluate a single parameter set – returns a dict with ALL metrics
# max_skip_level is forced to 0 (auto)
# ------------------------------------------------------------
def evaluate_params(params, section, pattern, n_keys, nops, seed=42, repetitions=3, warmup_ops=0):
    eval_params = params.copy()
    # Ensure max_skip_level is 0 – this is critical.
    eval_params['max_skip_level'] = 0
    fixup_params(eval_params)

    all_metrics = []  # list of dicts, one per repetition

    for rep in range(repetitions):
        tmp = f"_tmp_{os.getpid()}_{rep}.csv"
        cmd = build_cmd(section, pattern, n_keys, nops, seed + rep, eval_params, tmp, warmup_ops=warmup_ops)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
            if result.returncode != 0:
                print(f"  Benchmark error (code {result.returncode}): {result.stderr[:200]}")
            if os.path.exists(tmp):
                with open(tmp, newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Take the last row (the actual result row)
                        row = rows[-1]
                        # Convert all numeric fields to float where possible
                        metrics = {}
                        for key, val in row.items():
                            try:
                                metrics[key] = float(val)
                            except (ValueError, TypeError):
                                metrics[key] = val
                        all_metrics.append(metrics)
                os.remove(tmp)
        except subprocess.TimeoutExpired:
            print(f"  Timeout after 600s for {cmd}")
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass

    if not all_metrics:
        return None

    # Aggregate across repetitions: compute median for each numeric metric
    # Also keep the parameter set and other non‑numeric fields
    aggregated = {}
    # Collect all keys from the first metrics dict (they should be the same)
    keys = all_metrics[0].keys()
    for key in keys:
        vals = []
        for m in all_metrics:
            try:
                v = float(m[key])
                vals.append(v)
            except (ValueError, TypeError):
                # Non‑numeric fields like section, pattern, label – keep as is from first
                if key not in aggregated:
                    aggregated[key] = m[key]
                continue
        if vals:
            aggregated[f"median_{key}"] = statistics.median(vals)
            aggregated[f"mean_{key}"] = statistics.mean(vals)
        else:
            # Already handled above
            pass

    # Also add the original parameters (for easy reference)
    for k, v in eval_params.items():
        aggregated[f"param_{k}"] = v

    return aggregated

# ------------------------------------------------------------
# Scaling rules for transferring to other sizes
# ------------------------------------------------------------
def scale_params(params, target_n, ref_n=100000):
    scaled = params.copy()
    ratio = target_n / ref_n
    if 'local_interval' in scaled:
        scaled['local_interval'] = max(1, int(scaled['local_interval'] * (ratio ** 0.3)))
    if 'sub_interval' in scaled:
        scaled['sub_interval'] = max(1, int(scaled['sub_interval'] * (ratio ** 0.3)))
    fixup_params(scaled)
    return scaled

# ------------------------------------------------------------
# Full benchmark with warm‑up, save raw CSV, return rows
# ------------------------------------------------------------
def run_bench_full(section, pattern, n_keys, nops, seed, params, tmp_file, warmup_ops=5000, runs=10):
    cmd = build_cmd(section, pattern, n_keys, nops, seed, params, tmp_file, warmup_ops=warmup_ops, runs=runs)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        print(f"    Benchmark error (code {result.returncode}): {result.stderr[:200]}")
        return []
    if not os.path.exists(tmp_file):
        return []
    with open(tmp_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

# ------------------------------------------------------------
# Aggregate across runs: compute mean/median for all numeric metrics
# ------------------------------------------------------------
def aggregate_runs(all_runs_metrics, parameter_keys, output_agg_file):
    if not all_runs_metrics:
        return
    run_rows = []
    for rows in all_runs_metrics:
        if rows:
            run_rows.append(rows[-1])  # take the last row from each run
    if not run_rows:
        return

    constant_row = {}
    metric_keys = set()
    for row in run_rows:
        for k, v in row.items():
            if k in parameter_keys or k in {'section', 'pattern', 'n', 'nops', 'runs', 'seed'}:
                constant_row[k] = v
            else:
                metric_keys.add(k)

    agg = {}
    for key in metric_keys:
        vals = []
        for row in run_rows:
            try:
                v = float(row[key])
                vals.append(v)
            except (ValueError, KeyError):
                pass
        if vals:
            agg[key] = {
                'mean': statistics.mean(vals),
                'median': statistics.median(vals),
            }
        else:
            agg[key] = {'mean': float('nan'), 'median': float('nan')}

    with open(output_agg_file, 'w', newline='') as f:
        fieldnames = list(constant_row.keys())
        for key in sorted(metric_keys):
            fieldnames.append(f"mean_{key}")
            fieldnames.append(f"median_{key}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        out_row = constant_row.copy()
        for key in sorted(metric_keys):
            out_row[f"mean_{key}"] = agg[key]['mean']
            out_row[f"median_{key}"] = agg[key]['median']
        writer.writerow(out_row)

# ------------------------------------------------------------
# Final evaluation with warm‑up and aggregation
# ------------------------------------------------------------
def final_evaluate_and_aggregate(params, section, pattern, n_keys, nops, seed, output_dir, size_label,
                                 warmup_ops=5000, repetitions=5, internal_runs=10, discard_first_rep=True):
    """
    Runs `repetitions` separate process invocations of the C benchmark, each doing
    `internal_runs` timed repetitions internally (so jli_stddev_us / jli_jitter_us /
    etc. are computed from real intra-process samples, not degenerate runs=1 rows).

    discard_first_rep: if True (default), the very first process invocation (rep 0)
    is run and saved to disk for inspection, but EXCLUDED from aggregation. This
    process pays one-time OS costs (process creation, cold malloc arena, cold
    icache/dcache, CPU frequency ramp-up) that are not representative of steady-state
    algorithm performance and have been observed to disproportionately inflate
    jli_avg_us / ratio relative to later repetitions. Set to False to include it.
    """
    eval_params = params.copy()
    eval_params['max_skip_level'] = 0   # force auto
    fixup_params(eval_params)
    sub_dir = os.path.join(output_dir, f"final_eval_{size_label}")
    os.makedirs(sub_dir, exist_ok=True)

    # Run one extra repetition up front if we're going to discard the first one,
    # so the aggregated stats still reflect `repetitions` worth of real samples.
    total_to_run = repetitions + 1 if discard_first_rep else repetitions

    all_run_metrics = []
    for rep in range(total_to_run):
        raw_file = os.path.join(sub_dir, f"raw_run_{rep}.csv")
        rows = run_bench_full(section, pattern, n_keys, nops, seed + rep, eval_params, raw_file,
                               warmup_ops=warmup_ops, runs=internal_runs)
        all_run_metrics.append(rows)

    if discard_first_rep and all_run_metrics:
        discarded = all_run_metrics[0]
        if discarded:
            try:
                discarded_avg = discarded[-1].get('jli_avg_us', 'n/a')
            except Exception:
                discarded_avg = 'n/a'
            print(f"  Discarding rep 0 as cold-start warmup process (jli_avg_us={discarded_avg})")
        all_run_metrics = all_run_metrics[1:]

    parameter_keys = set(eval_params.keys())
    parameter_keys.update({'section', 'pattern', 'n', 'nops'})  # max_skip_level not needed
    aggregate_file = os.path.join(sub_dir, "aggregated.csv")
    aggregate_runs(all_run_metrics, parameter_keys, aggregate_file)
    print(f"Final evaluation for {size_label} saved in {sub_dir}")

# ------------------------------------------------------------
# Phase 1: Random search at reference size (100k)
# ------------------------------------------------------------
def phase1_search(section, pattern, n_ref, nops_ref, trials, repetitions, output_dir, seed, seed_params, trial_warmup_ops=5000):
    print(f"\n=== Phase 1: Random search at n={n_ref} (with {len(seed_params)} seeds) ===")
    best_speedup = float('inf')
    best_params = None
    results = []

    # Evaluate seeds
    for idx, seed_par in enumerate(seed_params):
        params = seed_par.copy()
        # No max_skip_level chosen; it's forced to 0 inside evaluate_params

        metrics = evaluate_params(params, section, pattern, n_ref, nops_ref,
                                  seed=seed, repetitions=repetitions,
                                  warmup_ops=trial_warmup_ops)
        if not metrics:
            print(f"Seed {idx+1}/{len(seed_params)}: evaluation failed, skipping")
            continue
        trial_speedup = metrics.get('median_ratio', float('inf'))
        results.append({
            'trial': f"seed_{idx}",
            'variant_label': 'seed',
            **metrics
        })

        print(f"Seed {idx+1}/{len(seed_params)}: speedup = {trial_speedup:.4f}")
        if trial_speedup < best_speedup:
            best_speedup = trial_speedup
            best_params = params.copy()
            print(f"  *** NEW BEST: {best_speedup:.4f} ***")

    # Random trials
    for trial in range(trials):
        params = random_params(section, seed_pool=seed_params)
        # No max_skip_level chosen; it's forced to 0 inside evaluate_params

        metrics = evaluate_params(params, section, pattern, n_ref, nops_ref,
                                  seed=seed + trial, repetitions=repetitions,
                                  warmup_ops=trial_warmup_ops)
        if not metrics:
            print(f"Trial {trial+1}/{trials}: evaluation failed, skipping")
            continue
        trial_speedup = metrics.get('median_ratio', float('inf'))
        results.append({
            'trial': trial,
            'variant_label': 'discovered',
            **metrics
        })

        print(f"Trial {trial+1}/{trials}: speedup = {trial_speedup:.4f}")
        if trial_speedup < best_speedup:
            best_speedup = trial_speedup
            best_params = params.copy()
            print(f"  *** NEW BEST: {best_speedup:.4f} ***")

    os.makedirs(output_dir, exist_ok=True)
    phase1_file = os.path.join(output_dir, f"phase1_{section}_{pattern}_n{n_ref}.csv")
    with open(phase1_file, 'w', newline='') as f:
        # Write all keys from the first result (they should be consistent)
        if results:
            fieldnames = list(results[0].keys())
        else:
            fieldnames = ['trial', 'variant_label', 'speedup'] + list(PARAM_RANGES.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    if best_params is None:
        raise RuntimeError(
            f"Phase 1 ({section}/{pattern}): every seed and trial evaluation failed "
            f"(0/{len(seed_params) + trials} succeeded). Check that {BENCH_EXE} exists, "
            f"is executable, and that build_cmd's arguments match what it expects."
        )

    best_params['speedup'] = best_speedup
    best_file = os.path.join(output_dir, f"best_phase1_{section}_{pattern}_n{n_ref}.csv")
    with open(best_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(best_params.keys()))
        writer.writeheader()
        writer.writerow(best_params)

    print(f"Phase 1 complete. Best speedup = {best_speedup:.4f}")
    return best_params, best_speedup

# ------------------------------------------------------------
# Phase 1b: Refinement (optional)
# ------------------------------------------------------------
def phase1b_refine(section, pattern, n_ref, nops_ref, best_params,
                   trials, repetitions, output_dir, seed, trial_warmup_ops=5000):
    print(f"\n=== Refinement phase for {section} / {pattern} (n={n_ref}) ===")
    print(f"  Starting from best speedup = {best_params['speedup']:.4f}")

    # level_probability is fixed at 0.5 — not a search variable
    varied_keys = ['segment_size', 'shortcuts_per_junction', 'block_size'] \
        if section == "STATIC" else [k for k in PARAM_RANGES.keys() if k != 'level_probability']

    fixed_params = {k: v for k, v in best_params.items()
                    if k not in varied_keys + ['speedup']}

    cur_values = {k: best_params.get(k, DEFAULTS.get(k, 0)) for k in varied_keys}
    choices = {k: PARAM_RANGES[k] for k in varied_keys}

    def narrow_range(base, factor=0.5, choices=None):
        if choices is None:
            return [int(base * (1 - factor)), int(base), int(base * (1 + factor))]
        else:
            near = [c for c in choices if abs(c - base) <= base * factor]
            if not near:
                near = [base]
            return near

    best_refine_speedup = best_params['speedup']
    best_refine_params = best_params.copy()
    results = []

    for trial in range(trials):
        new_params = fixed_params.copy()
        for k in varied_keys:
            base = cur_values[k]
            if k in ['segment_size', 'shortcuts_per_junction', 'local_interval', 'sub_interval',
                     'max_suboptimal_segments', 'min_suboptimal_events_before_global', 'block_size']:
                factor = 0.5
                choices_list = choices[k]
                new_val = random.choice(narrow_range(base, factor, choices_list))
                new_params[k] = int(new_val)
            else:
                if k == 'level_probability':
                    factor = 0.3
                elif k in ['t_j', 'soft_pct', 'hard_pct', 'flagged_ratio_limit', 'min_seg_len_pct']:
                    factor = 0.3
                else:
                    factor = 0.3
                choices_list = choices[k]
                near = narrow_range(base, factor, choices_list)
                new_params[k] = random.choice(near)

        fixup_params(new_params)  # also sets max_skip_level=0

        metrics = evaluate_params(new_params, section, pattern, n_ref, nops_ref,
                                  seed=seed + trial,
                                  repetitions=repetitions, warmup_ops=trial_warmup_ops)
        if not metrics:
            continue
        speedup = metrics.get('median_ratio', float('inf'))
        results.append({
            'trial': trial,
            'variant_label': 'discovered',
            **metrics
        })

        print(f"Refine trial {trial+1}/{trials}: speedup = {speedup:.4f}")
        if speedup < best_refine_speedup:
            best_refine_speedup = speedup
            best_refine_params = new_params.copy()
            print(f"  *** NEW BEST: {best_refine_speedup:.4f} ***")

    refine_file = os.path.join(output_dir, f"refine_{section}_{pattern}_n{n_ref}.csv")
    with open(refine_file, 'w', newline='') as f:
        if results:
            fieldnames = list(results[0].keys())
        else:
            fieldnames = ['trial', 'variant_label', 'speedup'] + varied_keys + list(fixed_params.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    best_refine_params['speedup'] = best_refine_speedup
    best_refine_file = os.path.join(output_dir, f"best_refine_{section}_{pattern}_n{n_ref}.csv")
    with open(best_refine_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(best_refine_params.keys()))
        writer.writeheader()
        writer.writerow(best_refine_params)

    print(f"Refinement complete. Best speedup improved from {best_params['speedup']:.4f} to {best_refine_speedup:.4f}")
    return best_refine_params, best_refine_speedup

# ------------------------------------------------------------
# Phase 2: Transfer to other sizes (50k and 250k)
# ------------------------------------------------------------
def phase2_transfer(section, pattern, best_params, ref_n, target_sizes, repetitions, output_dir, seed,
                    seed_params_250k=None, trial_warmup_ops=5000):
    print(f"\n=== Phase 2: Transfer to {target_sizes} ===")
    transfer_results = []
    for target_n in target_sizes:
        nops = max(1, int(target_n / 3.5))
        scaled = scale_params(best_params, target_n, ref_n)

        candidates = [scaled]
        if target_n == 250000 and seed_params_250k:
            print(f"  Evaluating {len(seed_params_250k)} additional target seed(s) directly for n=250000")
            candidates.extend(seed_params_250k)

        best_speedup = float('inf')
        best_label = None
        best_candidate_params = None

        for i, cand in enumerate(candidates):
            cand = fixup_params(cand.copy())
            label = "scaled" if i == 0 else f"seed_{i-1}"

            metrics = evaluate_params(cand, section, pattern, target_n, nops,
                                      seed=seed, repetitions=repetitions,
                                      warmup_ops=trial_warmup_ops)
            if not metrics:
                print(f"  n={target_n} candidate '{label}': evaluation failed, skipping")
                continue
            cand_speedup = metrics.get('median_ratio', float('inf'))
            transfer_results.append({
                'n': target_n,
                'variant_label': label,
                **metrics
            })

            print(f"  n={target_n} candidate '{label}': speedup = {cand_speedup:.4f}")
            if cand_speedup < best_speedup:
                best_speedup = cand_speedup
                best_label = label
                best_candidate_params = cand.copy()

        print(f"  n={target_n}: best speedup = {best_speedup:.4f} ({best_label})")
        if best_candidate_params is None:
            print(f"  WARNING: all candidates failed for n={target_n}; "
                  f"best_transfer_{section}_{pattern}_n{target_n}.csv will be skipped.")
            continue
        best_transfer = best_candidate_params.copy()
        best_transfer['speedup'] = best_speedup
        best_transfer['n'] = target_n
        transfer_best_file = os.path.join(output_dir, f"best_transfer_{section}_{pattern}_n{target_n}.csv")
        with open(transfer_best_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(best_transfer.keys()))
            writer.writeheader()
            writer.writerow(best_transfer)

    transfer_file = os.path.join(output_dir, f"phase2_transfer_{section}_{pattern}.csv")
    with open(transfer_file, 'w', newline='') as f:
        if transfer_results:
            fieldnames = list(transfer_results[0].keys())
        else:
            fieldnames = ['n', 'variant_label', 'speedup'] + list(PARAM_RANGES.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in transfer_results:
            writer.writerow(r)
    return transfer_results


# ------------------------------------------------------------
# Phase 3: Fine-tune outliers
# ------------------------------------------------------------
def phase3_finetune(section, pattern, target_n, current_speedup, output_dir, seed,
                    repetitions=5, extra_trials=30, trial_warmup_ops=5000):
    print(f"\n=== Phase 3: Fine-tuning n={target_n} (current speedup = {current_speedup:.4f}) ===")
    best_file = os.path.join(output_dir, f"best_transfer_{section}_{pattern}_n{target_n}.csv")
    if not os.path.exists(best_file):
        print("  No best transfer file found, skipping fine-tune.")
        return current_speedup, None

    with open(best_file, 'r') as f:
        reader = csv.DictReader(f)
        best_row = next(reader)

    current_params = {}
    for k, v in best_row.items():
        if k in INT_PARAMS:
            try:
                current_params[k] = int(float(v))
            except:
                current_params[k] = v
        else:
            try:
                current_params[k] = float(v)
            except:
                current_params[k] = v
    fixup_params(current_params)

    current_speedup = float(best_row.get('speedup', current_speedup))

    def narrow_range(base, factor=0.5, choices=None):
        if choices is None:
            return [int(base * (1 - factor)), int(base), int(base * (1 + factor))]
        else:
            near = [c for c in choices if abs(c - base) <= base * factor]
            if not near:
                near = [base]
            return near

    seg_choices = PARAM_RANGES['segment_size']
    sc_choices = PARAM_RANGES['shortcuts_per_junction']
    # lp_choices removed: level_probability is fixed at 0.5
    bs_choices = PARAM_RANGES['block_size']
    local_choices = PARAM_RANGES.get('local_interval', [current_params.get('local_interval', 1000)])
    sub_choices = PARAM_RANGES.get('sub_interval', [current_params.get('sub_interval', 1700)])

    best_found = current_speedup
    best_params = current_params.copy()

    for trial in range(extra_trials):
        params = {}
        params['segment_size'] = random.choice(narrow_range(current_params['segment_size'], 0.5, seg_choices))
        params['shortcuts_per_junction'] = random.choice(narrow_range(current_params['shortcuts_per_junction'], 0.5, sc_choices))
        params['level_probability'] = 0.5  # fixed: p=0.5 for both structures
        params['block_size'] = random.choice(narrow_range(current_params['block_size'], 0.5, bs_choices))
        if section != "STATIC":
            params['local_interval'] = random.choice(narrow_range(current_params['local_interval'], 0.4, local_choices))
            params['sub_interval'] = random.choice(narrow_range(current_params['sub_interval'], 0.4, sub_choices))
            params['t_j'] = random.uniform(max(0.05, current_params['t_j']*0.8), min(0.5, current_params['t_j']*1.2))
            params['soft_pct'] = random.uniform(max(0.05, current_params['soft_pct']*0.8), min(0.5, current_params['soft_pct']*1.2))
            params['hard_pct'] = random.uniform(max(0.1, current_params['hard_pct']*0.8), min(0.9, current_params['hard_pct']*1.2))
            params['flagged_ratio_limit'] = random.uniform(max(0.1, current_params['flagged_ratio_limit']*0.8), min(0.5, current_params['flagged_ratio_limit']*1.2))
            params['min_seg_len_pct'] = random.uniform(max(0.05, current_params['min_seg_len_pct']*0.8), min(0.4, current_params['min_seg_len_pct']*1.2))
            params['max_suboptimal_segments'] = random.uniform(0.1, 0.7)
            params['min_suboptimal_events_before_global'] = random.choice([2,3,4,5,6])
            params['emergency_hard_segment_ratio'] = random.uniform(0.7, 0.98)
            params['stop_crash_local_sub'] = random.uniform(0.03, 0.09)
        for k in DEFAULTS:
            if k not in params:
                params[k] = DEFAULTS[k]
        fixup_params(params)  # sets max_skip_level=0

        nops = max(1, int(target_n / 3.5))

        metrics = evaluate_params(params, section, pattern, target_n, nops,
                                  seed=seed + target_n + trial, repetitions=repetitions,
                                  warmup_ops=trial_warmup_ops)
        if not metrics:
            continue
        speedup = metrics.get('median_ratio', float('inf'))

        if speedup < best_found:
            best_found = speedup
            best_params = params.copy()
            print(f"  Fine-tune trial {trial+1}: new best = {best_found:.4f}")

    best_params['speedup'] = best_found
    final_file = os.path.join(output_dir, f"best_final_{section}_{pattern}_n{target_n}.csv")
    with open(final_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(best_params.keys()))
        writer.writeheader()
        writer.writerow(best_params)
    print(f"Fine-tune complete. Final speedup = {best_found:.4f}")
    return best_found, best_params

# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hierarchical parameter search for JLI (v8, includes block_size)")
    parser.add_argument("--section", required=True, choices=["STATIC","INSERT","DELETE"])
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of Phase 1 random trials (default: 100 for STATIC, 180 for INSERT/DELETE)")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--primary-csv", default="filtered_lowest_tp_speedup_1.csv")
    parser.add_argument("--fallback-csv", default="filtered_lowest_tp_speedup.csv")
    parser.add_argument("--refine", action="store_true", help="Enable refinement phase (INSERT/DELETE only)")
    parser.add_argument("--refine-trials", type=int, default=80, help="Number of random trials in refinement phase")
    parser.add_argument("--refine-repetitions", type=int, default=5, help="Repetitions per refinement trial")
    parser.add_argument("--final-warmup-ops", type=int, default=5000, help="Warm‑up ops for final evaluation")
    parser.add_argument("--final-reps", type=int, default=5, help="Repetitions for final evaluation")
    parser.add_argument("--final-internal-runs", type=int, default=10,
                         help="Internal --runs passed to the C benchmark per final-eval repetition "
                              "(must be >1 for jli_stddev_us/jli_jitter_us/etc. to be non-zero; max 30)")
    parser.add_argument("--no-discard-first-rep", action="store_true",
                         help="Include the first final-eval process repetition in aggregation "
                              "(by default it is run but discarded, since it pays one-time process "
                              "cold-start costs that inflate jli_avg_us/ratio relative to steady state)")
    parser.add_argument("--trial-warmup-ops", type=int, default=500,
                        help="Warm‑up ops for each trial evaluation (default 500)")
    parser.add_argument("--n-ref", type=int, default=500000,
                        help="Reference n for Phase 1 / Phase 1b search (default 500000). "
                             "Phase 2 transfers the resulting best params out to --target-sizes.")
    parser.add_argument("--target-sizes", type=int, nargs="+", default=[50000, 100000, 250000, 1000000],
                        help="n values Phase 2/3 transfer+finetune to, besides n-ref "
                             "(default: 50000 100000 250000 1000000, i.e. the existing "
                             "50k/100k/250k paper points plus the new 1M point)")
    args = parser.parse_args()

    if args.trials is None:
        args.trials = 100 if args.section == "STATIC" else 180

    random.seed(args.seed)

    n_ref = args.n_ref
    nops_ref = max(1, int(n_ref / 3.5))
    ref_label = f"{n_ref // 1000}k" if n_ref < 1000000 else f"{n_ref // 1000000}m"

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"hierarchical_{ref_label}_ref_{args.section}_{args.pattern}_{ts}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting hierarchical search for {args.section} / {args.pattern}")
    print(f"Reference size: n={n_ref}, nops={nops_ref}")
    print(f"Phase 1 trials: {args.trials}, repetitions: {args.repetitions}")
    print(f"Transfer/finetune targets: {args.target_sizes}")
    print(f"Output directory: {output_dir}")

    # Load seeds - try the reference n first (500k), then fall back to
    # whatever exact-n seed rows exist (e.g. previously-searched 100k/250k
    # bests), since a fresh 500k CSV likely doesn't exist yet on first run.
    seed_params_ref = load_seed_rows(args.section, args.pattern, n_ref, args.primary_csv, args.fallback_csv)
    seed_params_250k = load_seed_rows(args.section, args.pattern, 250000, args.primary_csv, args.fallback_csv)
    if not seed_params_ref:
        # No exact seeds at n_ref yet -> reuse 250k seeds (closest prior
        # search point) so Phase 1 isn't starting from nothing.
        seed_params_ref = seed_params_250k
    print(f"Loaded {len(seed_params_ref)} seed(s) for n_ref={n_ref}, {len(seed_params_250k)} seed(s) for 250k (exact n)")

    # Phase 1
    best_params, best_speedup = phase1_search(
        args.section, args.pattern, n_ref, nops_ref,
        args.trials, args.repetitions, output_dir, args.seed, seed_params_ref,
        trial_warmup_ops=args.trial_warmup_ops
    )

    # Optional refinement
    if args.refine and args.section in ("INSERT", "DELETE"):
        best_params, best_speedup = phase1b_refine(
            args.section, args.pattern, n_ref, nops_ref,
            best_params,
            args.refine_trials, args.refine_repetitions, output_dir, args.seed,
            trial_warmup_ops=args.trial_warmup_ops
        )

    # Final evaluation for reference size
    final_evaluate_and_aggregate(best_params, args.section, args.pattern,
                                 n_ref, nops_ref,
                                 args.seed, output_dir, ref_label,
                                 warmup_ops=args.final_warmup_ops,
                                 repetitions=args.final_reps,
                                 internal_runs=args.final_internal_runs,
                                 discard_first_rep=not args.no_discard_first_rep)

    # Phase 2 - transfer the 500k-search-derived best params out to every
    # target size (existing paper points 50k/100k/250k, plus the new 1M point)
    target_sizes = [n for n in args.target_sizes if n != n_ref]
    transfer_results = phase2_transfer(
        args.section, args.pattern, best_params, n_ref, target_sizes,
        args.repetitions, output_dir, args.seed, seed_params_250k=seed_params_250k,
        trial_warmup_ops=args.trial_warmup_ops
    )

    # ------------------------------------------------------------
    # After Phase 2 – load best parameters from the CSV files
    # ------------------------------------------------------------
    def load_best_transfer_params(target_n):
        """Read the best_transfer CSV for target_n and return params dict + speedup."""
        csv_path = os.path.join(output_dir, f"best_transfer_{args.section}_{args.pattern}_n{target_n}.csv")
        if not os.path.exists(csv_path):
            return None, float('inf')
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            try:
                row = next(reader)
            except StopIteration:
                return None, float('inf')

        # Convert fields back to appropriate types
        params = {}
        for k, v in row.items():
            if k == 'speedup':
                continue
            if k in INT_PARAMS or k == 'n':
                try:
                    params[k] = int(float(v))
                except (ValueError, TypeError):
                    params[k] = v
            else:
                try:
                    params[k] = float(v)
                except (ValueError, TypeError):
                    params[k] = v

        speedup = float(row.get('speedup', float('inf')))
        return params, speedup

    def size_label(n):
        return f"{n // 1000}k" if n < 1000000 else f"{n // 1000000}m"

    # Phase 3 - runs generically over every target size (50k/100k/250k/1M by
    # default) instead of only the old hardcoded 250k/50k pair.
    target_params = {}
    target_final_speedup = {}
    for target_n in target_sizes:
        params_t, transfer_speedup_t = load_best_transfer_params(target_n)
        final_speedup_t = transfer_speedup_t

        if not args.skip_phase3 and transfer_speedup_t != float('inf') and transfer_speedup_t > 0.5:
            tuned_t, tuned_params_t = phase3_finetune(args.section, args.pattern, target_n, transfer_speedup_t,
                                                       output_dir, args.seed, args.repetitions, extra_trials=50,
                                                       trial_warmup_ops=args.trial_warmup_ops)
            if tuned_t < transfer_speedup_t:
                final_speedup_t = tuned_t
                params_t = tuned_params_t

        target_params[target_n] = params_t
        target_final_speedup[target_n] = final_speedup_t

        # Final evaluation for this target size
        if params_t:
            nops_t = max(1, int(target_n / 3.5))
            final_evaluate_and_aggregate(params_t, args.section, args.pattern,
                                         target_n, nops_t,
                                         args.seed, output_dir, size_label(target_n),
                                         warmup_ops=args.final_warmup_ops,
                                         repetitions=args.final_reps,
                                         internal_runs=args.final_internal_runs,
                                         discard_first_rep=not args.no_discard_first_rep)

    # Summary files
    for target_n in target_sizes:
        best_params_dict = target_params[target_n]
        final_speedup = target_final_speedup[target_n]
        suffix = size_label(target_n)
        summary_file = os.path.join(output_dir, f"final_best_{suffix}_{args.section}_{args.pattern}.csv")
        if best_params_dict:
            row_to_write = best_params_dict.copy()
            row_to_write['final_speedup'] = final_speedup
            fieldnames = list(row_to_write.keys())
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_to_write)
        else:
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['speedup'])
                writer.writeheader()
                writer.writerow({'speedup': final_speedup})

    print("\n=== Search completed ===")
    print(f"All results saved under: {output_dir}")
    for target_n in target_sizes:
        print(f"Best speedup at {size_label(target_n)}: {target_final_speedup[target_n]:.4f}")

# ============================================================
# FINAL EVALUATION RUNNER
# Run final_evaluate_and_aggregate for every row in all_agg CSV
# using the stored best parameters, then merge all aggregated
# outputs into one combined CSV.
#
# Usage:
#   python parameter_search.py --run-final-eval \
#       --agg-csv all_agg_1.csv \
#       --final-output-dir final_eval_results \
#       --final-warmup-ops 5000 \
#       --final-reps 10 \
#       --final-internal-runs 30
#
#   python parameter_search.py --merge-final-eval \
#       --final-output-dir final_eval_results \
#       --merged-csv all_agg_final.csv
# ============================================================

def run_final_eval_from_csv(agg_csv, output_dir, warmup_ops, reps, internal_runs, seed=42):
    """
    Read every row from agg_csv. Each row is one (section, pattern, n) combo
    with its best params already stored. Run final_evaluate_and_aggregate for
    each combo and save results under output_dir.
    """
    if not os.path.exists(agg_csv):
        print(f"ERROR: CSV not found: {agg_csv}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(agg_csv, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {agg_csv}")
    print(f"Settings: warmup_ops={warmup_ops}  reps={reps}  internal_runs={internal_runs}\n")

    for idx, row in enumerate(rows):
        section = row['section'].strip().upper()
        pattern = row['pattern'].strip()
        n       = int(float(row['n']))
        nops    = int(float(row['nops']))

        # BUILD section has no tunable params — skip it
        if section == "BUILD":
            print(f"[{idx+1}/{len(rows)}] Skipping BUILD {pattern} n={n} (no tunable params)")
            continue

        print(f"[{idx+1}/{len(rows)}] {section} / {pattern} / n={n}  nops={nops}")

        try:
            params = row_to_params(row, section)
        except Exception as e:
            print(f"  WARNING: could not parse params for row {idx}: {e} — skipping")
            continue

        # size label used for subdirectory naming
        if n <= 55000:
            size_label = f"{section}_{pattern}_50k"
        elif n <= 110000:
            size_label = f"{section}_{pattern}_100k"
        else:
            size_label = f"{section}_{pattern}_250k"

        final_evaluate_and_aggregate(
            params,
            section,
            pattern,
            n,
            nops,
            seed,
            output_dir,
            size_label,
            warmup_ops=warmup_ops,
            repetitions=reps,
            internal_runs=internal_runs,
            discard_first_rep=True,
        )
        print()

    print(f"All final evaluations complete. Results in: {output_dir}")


def merge_final_eval_csvs(output_dir, merged_csv):
    """
    Walk output_dir, find every aggregated.csv produced by
    final_evaluate_and_aggregate, and concatenate them into one CSV.
    """
    found = []
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == "aggregated.csv":
                found.append(os.path.join(root, fname))

    if not found:
        print(f"No aggregated.csv files found under {output_dir}")
        return

    found.sort()
    print(f"Found {len(found)} aggregated.csv files — merging into {merged_csv}")

    all_rows = []
    header = None
    for path in found:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue
            if header is None:
                header = reader.fieldnames
            all_rows.extend(rows)

    if not all_rows or header is None:
        print("No data to merge.")
        return

    # Union of all fieldnames in case different runs have slightly different columns
    all_fields = list(header)
    for row in all_rows:
        for k in row:
            if k not in all_fields:
                all_fields.append(k)

    with open(merged_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Merged {len(all_rows)} rows → {merged_csv}")


def main_final_eval():
    parser = argparse.ArgumentParser(
        description="Run final evaluation from aggregated CSV and/or merge results"
    )
    parser.add_argument("--run-final-eval",  action="store_true",
                        help="Run final_evaluate_and_aggregate for each row in --agg-csv")
    parser.add_argument("--merge-final-eval", action="store_true",
                        help="Merge all aggregated.csv files in --final-output-dir into --merged-csv")
    parser.add_argument("--agg-csv",          default="all_agg_1.csv",
                        help="Input aggregated CSV with best params per (section, pattern, n)")
    parser.add_argument("--final-output-dir", default="final_eval_results",
                        help="Directory to write final evaluation outputs")
    parser.add_argument("--final-warmup-ops", type=int, default=5000)
    parser.add_argument("--final-reps",       type=int, default=10)
    parser.add_argument("--final-internal-runs", type=int, default=30)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--merged-csv",       default="all_agg_final.csv",
                        help="Output path for the merged CSV (used with --merge-final-eval)")
    args = parser.parse_args()

    if not args.run_final_eval and not args.merge_final_eval:
        parser.print_help()
        sys.exit(0)

    if args.run_final_eval:
        run_final_eval_from_csv(
            agg_csv=args.agg_csv,
            output_dir=args.final_output_dir,
            warmup_ops=args.final_warmup_ops,
            reps=args.final_reps,
            internal_runs=args.final_internal_runs,
            seed=args.seed,
        )

    if args.merge_final_eval:
        merge_final_eval_csvs(
            output_dir=args.final_output_dir,
            merged_csv=args.merged_csv,
        )

if __name__ == "__main__":
    # Route to the final-eval runner if either of its flags appear in argv,
    # otherwise fall through to the normal hierarchical search main().
    if "--run-final-eval" in sys.argv or "--merge-final-eval" in sys.argv:
        main_final_eval()
    else:
        main()