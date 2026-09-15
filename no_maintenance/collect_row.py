#!/usr/bin/env python3
"""
consolidate_results.py – Recursively collect all raw-run and aggregated CSV files,
infer metadata, and produce two consolidated CSVs. Additionally, for each
configuration, compute the JLI/SL ratio, its 95% CI, and non-parametric
significance tests (Wilcoxon and sign test) from the raw runs, and add these
as columns to the aggregated output.
"""

import os
import argparse
import csv
import re
import math
from pathlib import Path
from collections import defaultdict

# Try to import scipy for statistical tests
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Columns that are redundant duplicates (they appear in raw CSVs and aggregated CSVs)
DROP_COLUMNS = {"section2", "pattern2", "n2", "nops2"}
DROP_PREFIXES = ("mean_section2", "median_section2",
                 "mean_pattern2", "median_pattern2",
                 "mean_n2", "median_n2",
                 "mean_nops2", "median_nops2")

# Names of the new statistical columns we will add
STAT_COLUMNS = [
    "mean_ratio",          # average of jli/sl per run
    "ratio_ci_low",        # lower bound of 95% CI for the ratio
    "ratio_ci_high",       # upper bound
    "wilcoxon_p",          # p-value from Wilcoxon signed-rank test
    "sign_p"               # p-value from sign test (binomial)
]

def parse_size_from_dirname(dirname: str):
    """Extract numeric n from a directory name like '50k', '100k', or 'final_eval_50k'."""
    if dirname.lower().endswith("k"):
        try:
            return int(float(dirname[:-1]) * 1000)
        except ValueError:
            return None
    m = re.search(r"final_eval_(\d+)k", dirname)
    if m:
        try:
            return int(float(m.group(1)) * 1000)
        except ValueError:
            return None
    return None

def infer_metadata_from_path(file_path: Path, base_dir: Path):
    """
    Walk up the directory tree to infer section, pattern, and n.
    Returns (section, pattern, n) where n is an int or "UNKNOWN".
    """
    rel_path = file_path.relative_to(base_dir)
    parts = rel_path.parts

    section = None
    pattern = None
    n = None

    for part in parts:
        if part.startswith("Results_"):
            m = re.search(r"Results_([A-Z]+)_([A-Za-z]+)_\d{8}_\d{6}", part)
            if m:
                section = m.group(1)
                pattern = m.group(2)
            n_candidate = parse_size_from_dirname(part)
            if n_candidate is not None:
                n = n_candidate
        elif part.startswith("final_eval_"):
            m = re.search(r"final_eval_([A-Za-z]+)_\d{8}_\d{6}", part)
            if m:
                section = m.group(1)
            n_candidate = parse_size_from_dirname(part)
            if n_candidate is not None:
                n = n_candidate
        elif part.lower().endswith("k"):
            if "_" in part:
                cand_pattern, size_part = part.rsplit("_", 1)
                if size_part.lower().endswith("k"):
                    try:
                        n_candidate = int(float(size_part[:-1]) * 1000)
                        if n is None:
                            n = n_candidate
                        if pattern is None:
                            pattern = cand_pattern
                    except ValueError:
                        pass
            else:
                n_candidate = parse_size_from_dirname(part)
                if n_candidate is not None and n is None:
                    n = n_candidate

    # Fallback: try parent directory
    if pattern is None:
        parent = file_path.parent.name
        if parent.lower().endswith("k") and "_" in parent:
            cand_pattern, _ = parent.rsplit("_", 1)
            pattern = cand_pattern

    if section is None:
        section = "UNKNOWN"
    if pattern is None:
        pattern = "UNKNOWN"
    if n is None:
        n = "UNKNOWN"

    return section, pattern, n

def find_latency_columns(fieldnames):
    """
    Given a list of column names, try to find the columns that contain
    JLI average latency and Skip‑list average latency.
    Returns (jli_col, sl_col) or (None, None) if not found.
    """
    jli_col = None
    sl_col = None
    # Look for columns that contain "jli" and "avg" and "us"
    for col in fieldnames:
        col_lower = col.lower()
        if "jli" in col_lower and ("avg" in col_lower or "mean" in col_lower) and "us" in col_lower:
            jli_col = col
        if "sl" in col_lower and ("avg" in col_lower or "mean" in col_lower) and "us" in col_lower:
            sl_col = col
    # If not found, try more generic: any column with "jli" and any with "sl"
    if jli_col is None:
        for col in fieldnames:
            if "jli" in col.lower():
                jli_col = col
                break
    if sl_col is None:
        for col in fieldnames:
            if "sl" in col.lower() and col != jli_col:  # avoid picking the same
                sl_col = col
                break
    return jli_col, sl_col

def compute_stats_for_config(raw_rows, jli_col, sl_col):
    """
    Given a list of raw rows (dicts) for one configuration, and the column names
    for JLI and SL latencies, compute the ratio, CI, and p-values.
    Returns a dict with keys: mean_ratio, ratio_ci_low, ratio_ci_high,
                              wilcoxon_p, sign_p
    """
    ratios = []
    diffs = []
    for row in raw_rows:
        try:
            jli = float(row.get(jli_col, float('nan')))
            sl  = float(row.get(sl_col, float('nan')))
        except (ValueError, TypeError):
            continue
        if not math.isnan(jli) and not math.isnan(sl) and sl > 0:
            ratios.append(jli / sl)
            diffs.append(jli - sl)

    if not ratios:
        return {col: float('nan') for col in STAT_COLUMNS}

    mean_ratio = sum(ratios) / len(ratios)

    # 95% CI using t-distribution
    n_ratios = len(ratios)
    if n_ratios > 1:
        std_ratio = math.sqrt(sum((r - mean_ratio)**2 for r in ratios) / (n_ratios - 1))
        if HAS_SCIPY:
            t_crit = stats.t.ppf(0.975, n_ratios - 1)
        else:
            t_crit = 1.96  # normal approximation
        ci_low = mean_ratio - t_crit * (std_ratio / math.sqrt(n_ratios))
        ci_high = mean_ratio + t_crit * (std_ratio / math.sqrt(n_ratios))
    else:
        ci_low = ci_high = float('nan')

    # Wilcoxon signed-rank test on differences
    wilcoxon_p = float('nan')
    if HAS_SCIPY and len(diffs) > 1:
        try:
            _, wilcoxon_p = stats.wilcoxon(diffs)
        except Exception:
            wilcoxon_p = float('nan')

    # Sign test (two-sided binomial)
    positive = sum(1 for d in diffs if d > 0)
    total = len(diffs)
    sign_p = float('nan')
    if total > 0:
     if HAS_SCIPY:
        try:
            # Newer SciPy (>=1.7.0)
            result = stats.binomtest(positive, total, 0.5, alternative='two-sided')
            sign_p = result.pvalue
        except AttributeError:
            # Fallback for older versions (deprecated, but keep for safety)
            try:
                sign_p = stats.binom_test(positive, total, 0.5, alternative='two-sided')
            except AttributeError:
                sign_p = float('nan')
    else:
        sign_p = float('nan')

    return {
        'mean_ratio': mean_ratio,
        'ratio_ci_low': ci_low,
        'ratio_ci_high': ci_high,
        'wilcoxon_p': wilcoxon_p,
        'sign_p': sign_p
    }

def collect_raw_runs(base_dir, output_file, exclude_discarded=True):
    """Find all raw_run_*.csv recursively, concatenate, and return the rows."""
    base = Path(base_dir)
    all_rows = []
    fieldnames = None

    for raw_file in base.rglob("raw_run_*.csv"):
        run_str = raw_file.stem.split("_")[-1]
        try:
            run_num = int(run_str)
        except ValueError:
            run_num = run_str

        is_discarded = (run_num == 0)
        if exclude_discarded and is_discarded:
            continue

        section, pattern, n = infer_metadata_from_path(raw_file, base)

        try:
            with open(raw_file, newline='') as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    raw_fields = [c for c in reader.fieldnames if c not in DROP_COLUMNS]
                    fieldnames = ["run", "discarded"] + raw_fields

                for row in reader:
                    row = {k: v for k, v in row.items() if k not in DROP_COLUMNS}
                    row_out = {
                        "run": run_num,
                        "discarded": is_discarded,
                    }
                    row_out.update(row)
                    # Fill missing metadata from path
                    if "section" not in row_out or row_out["section"] == "":
                        row_out["section"] = section
                    if "pattern" not in row_out or row_out["pattern"] == "":
                        row_out["pattern"] = pattern
                    if "n" not in row_out or row_out["n"] == "":
                        row_out["n"] = n
                    all_rows.append(row_out)
        except Exception as e:
            print(f"Warning: skipping {raw_file}: {e}")

    if not all_rows:
        print("No raw run files found.")
        return None

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} raw run records to {output_file}")
    return all_rows

def collect_aggregated(base_dir, output_file, raw_rows=None):
    """
    Find all aggregated.csv recursively, concatenate, and if raw_rows are provided,
    compute per-configuration stats and add them as columns.
    """
    base = Path(base_dir)
    all_rows = []
    fieldnames = None

    # First pass: collect all aggregated rows
    for agg_file in base.rglob("aggregated.csv"):
        section, pattern, n = infer_metadata_from_path(agg_file, base)
        try:
            with open(agg_file, newline='') as f:
                reader = csv.DictReader(f)
                # Determine final fieldnames (excluding redundant DROP_PREFIXES)
                if fieldnames is None:
                    agg_fields = [c for c in reader.fieldnames if not any(c.startswith(p) for p in DROP_PREFIXES)]
                    # Add new stat columns only if they are not already present
                    existing = set(agg_fields)
                    new_cols = [c for c in STAT_COLUMNS if c not in existing]
                    fieldnames = agg_fields + new_cols

                for row in reader:
                    row = {k: v for k, v in row.items() if not any(k.startswith(p) for p in DROP_PREFIXES)}
                    row_out = dict(row)
                    if "section" not in row_out or row_out["section"] == "":
                        row_out["section"] = section
                    if "pattern" not in row_out or row_out["pattern"] == "":
                        row_out["pattern"] = pattern
                    if "n" not in row_out or row_out["n"] == "":
                        row_out["n"] = n
                    all_rows.append(row_out)
        except Exception as e:
            print(f"Warning: skipping {agg_file}: {e}")

    if not all_rows:
        print("No aggregated files found.")
        return

    # If we have raw rows, compute stats per configuration and merge
    if raw_rows:
        # Find JLI and SL column names from the first raw row
        if raw_rows:
            sample_cols = raw_rows[0].keys()
            jli_col, sl_col = find_latency_columns(sample_cols)
            if jli_col is None or sl_col is None:
                print("Warning: Could not find JLI and SL latency columns in raw data. "
                      "Statistical columns will be left empty.")
                jli_col = sl_col = None

        if jli_col is not None and sl_col is not None:
            # Group raw rows by (section, pattern, n)
            raw_by_config = defaultdict(list)
            for r in raw_rows:
                key = (r.get('section', 'UNKNOWN'), r.get('pattern', 'UNKNOWN'), r.get('n', 'UNKNOWN'))
                raw_by_config[key].append(r)

            # Compute stats for each configuration
            stats_cache = {}
            for key, rows in raw_by_config.items():
                stats_cache[key] = compute_stats_for_config(rows, jli_col, sl_col)

            # Add stats to each aggregated row
            for row in all_rows:
                key = (row.get('section', 'UNKNOWN'), row.get('pattern', 'UNKNOWN'), row.get('n', 'UNKNOWN'))
                stats = stats_cache.get(key, {})
                for col in STAT_COLUMNS:
                    row[col] = stats.get(col, float('nan'))

    # Write final aggregated CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} aggregated records to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively consolidate raw run and aggregated CSV results, "
                    "and add statistical metrics (ratio, CI, p-values)."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root directory containing all result folders")
    parser.add_argument("--output-raw", default="all_raw_runs.csv",
                        help="Output CSV for raw runs")
    parser.add_argument("--output-agg", default="all_aggregated.csv",
                        help="Output CSV for aggregated data")
    parser.add_argument("--include-discarded", action="store_true",
                        help="Include raw_run_0.csv (cold‑start warmup) in --output-raw. "
                             "By default it is excluded.")
    args = parser.parse_args()

    # Collect raw runs (needed for stats)
    raw_rows = collect_raw_runs(args.input_dir, args.output_raw,
                                exclude_discarded=not args.include_discarded)

    # Collect aggregated and merge stats
    collect_aggregated(args.input_dir, args.output_agg, raw_rows)

if __name__ == "__main__":
    main()