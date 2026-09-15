/*
 * JLI Benchmark – v3 with level_probability (FIXED v2)
 * Compile: gcc -O2 -o bench bench.c -lm
 *
 * FIXES v2:
 *  - Insert keys are pattern-aware (random, sequential, hotspot, zipfian, adversarial)
 *  - Delete keys now support all meaningful patterns (random, sequential, hotspot, zipfian, adversarial)
 *  - Miss is search-only. Adversarial insert = append stress, adversarial delete = failed search.
 */

#define _GNU_SOURCE
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <ctype.h>

/* ── Import the JLI data structure + reference skip‑list ── */
#include "jli_v8_1.c"

/* ── MAX SKIP LEVEL FORMULAS (p = 0.5 for both structures) ──────────────
 *
 *  Skip list over n elements:
 *      max_level = ceil(log2(n))
 *
 *  JLI block skip-list over num_blocks blocks, where
 *      num_blocks ≈ n / (segment_size * block_m):
 *      max_level = ceil(log2(num_blocks))   = ceil(log2(n / (S * B)))
 *
 *  Both are clamped to [1, 64].
 * ────────────────────────────────────────────────────────────────────── */
static int bench_sl_max_level(int32_t n) {
    /* ceil(log2(n)) via bit-length of (n-1) */
    if (n <= 1) return 1;
    int lvl = 0;
    int32_t v = n - 1;
    while (v > 0) { v >>= 1; lvl++; }
    if (lvl < 1)  lvl = 1;
    if (lvl > 64) lvl = 64;
    return lvl;
}

static int bench_jli_block_max_level(int32_t n, int32_t segment_size, int32_t block_m) {
    /* ceil(log2(num_blocks)) where num_blocks = ceil(n / (segment_size * block_m)) */
    int32_t S = segment_size > 0 ? segment_size : 1;
    int32_t B = block_m     > 0 ? block_m      : 1;
    int32_t num_blocks = (n + S * B - 1) / (S * B);
    if (num_blocks <= 1) return 1;
    int lvl = 0;
    int32_t v = num_blocks - 1;
    while (v > 0) { v >>= 1; lvl++; }
    if (lvl < 1)  lvl = 1;
    if (lvl > 64) lvl = 64;
    return lvl;
}

// near the top, with other globals
static int warmup_ops = 0;
// Simple xorshift32 generator (period 2^32-1)
static uint32_t xorshift32_state = 1;

static void xorshift32_seed(uint32_t seed) {
    // 1. Seed the query workload generator
    xorshift32_state = seed ? seed : 1;

    // 2. Seed the Skip List and JLI structural PRNGs!
    // We cast to uint64_t and mix the bits slightly using XOR masks 
    // so the two structures don't generate identical sequence streams.
    rng_state     = ((uint64_t)seed) ^ 0xdeadbeefcafeULL;
    jli_rng_state = ((uint64_t)seed) ^ 0xcafebeefdeadULL;
}

static uint32_t xorshift32(void) {
    uint32_t x = xorshift32_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xorshift32_state = x;
    return x;
}

static double xorshift32_double(void) {
    return (double)xorshift32() / 4294967296.0;   // 2^32
}

/* ═══════════════════════════════════════════════════════════════════════
   BENCHMARK INFRASTRUCTURE
   ═══════════════════════════════════════════════════════════════════════ */

static int32_t fixed_n = 0;
static unsigned int user_seed = 0;

static int cmp_int32(const void *a, const void *b) {
    int32_t x = *(const int32_t*)a, y = *(const int32_t*)b;
    return (x > y) - (x < y);
}

/* ── Timing ──────────────────────────────────────────────────────────── */
static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── Statistics ──────────────────────────────────────────────────────── */
#define MAX_RUNS 30

typedef struct {
    double  samples[MAX_RUNS];
    int     n;
    double  avg, sdev, vmin, vmax;
    double  p50, p95, p99;
    double  ci_lo, ci_hi;
    double  jitter;
} Stats;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double t_crit(int n_samples) {
    static const double tbl[] = {
        0.0,
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
        2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045
    };
    int df = n_samples - 1;
    if (df <= 0)  return 12.706;
    if (df >= 30) return 2.042;
    return tbl[df];
}

static double percentile(double *sorted, int n, double pct) {
    if (n <= 0) return 0.0;
    double idx = pct * (n - 1);
    int lo = (int)idx;
    int hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

static void stats_compute(Stats *s) {
    if (s->n <= 0) { memset(&s->avg, 0, sizeof(Stats) - offsetof(Stats, avg)); return; }
    double sorted[MAX_RUNS];
    memcpy(sorted, s->samples, s->n * sizeof(double));
    qsort(sorted, s->n, sizeof(double), cmp_double);
    s->vmin = sorted[0];
    s->vmax = sorted[s->n - 1];
    s->p50  = percentile(sorted, s->n, 0.50);
    s->p95  = percentile(sorted, s->n, 0.95);
    s->p99  = percentile(sorted, s->n, 0.99);
    s->jitter = s->p99 - s->p50;
    double sum = 0.0;
    for (int i = 0; i < s->n; i++) sum += s->samples[i];
    s->avg = sum / s->n;
    double var = 0.0;
    for (int i = 0; i < s->n; i++) { double d = s->samples[i] - s->avg; var += d * d; }
    s->sdev = s->n > 1 ? sqrt(var / (s->n - 1)) : 0.0;
    double margin = t_crit(s->n) * s->sdev / sqrt((double)s->n);
    s->ci_lo = s->avg - margin;
    s->ci_hi = s->avg + margin;
}

static void stats_record(Stats *s, double us) {
    if (s->n < MAX_RUNS) s->samples[s->n++] = us;
}

static double win_rate(Stats *jli, Stats *sl) {
    int wins = 0, total = jli->n < sl->n ? jli->n : sl->n;
    for (int i = 0; i < total; i++)
        if (jli->samples[i] < sl->samples[i]) wins++;
    return total > 0 ? (double)wins / total : 0.0;
}

/* ── Workload generators ────────────────────────────────────────────── */
typedef enum {
    QPAT_RANDOM, QPAT_SEQUENTIAL, QPAT_HOTSPOT,
    QPAT_ZIPFIAN, QPAT_MISS, QPAT_ADVERSARIAL
} QueryPattern;

static QueryPattern string_to_qpat(const char *s) {
    if (strcasecmp(s, "random") == 0) return QPAT_RANDOM;
    if (strcasecmp(s, "sequential") == 0) return QPAT_SEQUENTIAL;
    if (strcasecmp(s, "hotspot") == 0) return QPAT_HOTSPOT;
    if (strcasecmp(s, "zipfian") == 0) return QPAT_ZIPFIAN;
    if (strcasecmp(s, "miss") == 0) return QPAT_MISS;
    if (strcasecmp(s, "adversarial") == 0) return QPAT_ADVERSARIAL;
    return QPAT_RANDOM;
}

static int32_t *make_pool(int32_t n, uint64_t seed) {
    xorshift32_seed((uint32_t)seed);
    int32_t *p = malloc(n * sizeof *p);
    if (!p) return NULL;
    for (int32_t i = 0; i < n; i++)
        p[i] = i * 3 + 1 + (xorshift32() % 2);
    return p;
}

/* ── Query generation for SEARCH (uses pool keys) ─────────────────── */
static int32_t *make_query_hits(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    if (n <= 0) { free(q); return NULL; }
    for (int32_t i = 0; i < k; i++)
        q[i] = pool[xorshift32() % n];
    return q;
}

static int32_t *make_query_hotspot(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    if (n <= 0) { free(q); return NULL; }

    int32_t hot_lo = (int32_t)(n * 0.90);
    int32_t hot_range = n - hot_lo;
    /* Bulletproof fallback against divide-by-zero for small sizes (n < 10) */
    if (hot_range <= 0) {
        hot_lo = 0;
        hot_range = n;
    }

    for (int32_t i = 0; i < k; i++) {
        if (xorshift32() % 10 < 8)
            q[i] = pool[hot_lo + (xorshift32() % hot_range)];
        else
            q[i] = pool[xorshift32() % n];
    }
    return q;
}

static int32_t *make_query_sequential(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    if (n <= 0) { free(q); return NULL; }
    for (int32_t i = 0; i < k; i++)
        q[i] = pool[i % n];
    return q;
}

static int32_t *make_query_zipfian(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    if (n <= 0) { free(q); return NULL; }

    double *cdf = malloc(n * sizeof *cdf);
    if (!cdf) {
        /* If allocation fails, fallback cleanly to uniform random hits */
        for (int32_t i = 0; i < k; i++) q[i] = pool[xorshift32() % n];
        return q;
    }

    double z = 0.0;
    for (int32_t i = 0; i < n; i++) z += 1.0 / (i + 1);
    double cum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        cum += 1.0 / ((i + 1) * z);
        cdf[i] = cum;
    }

    for (int32_t i = 0; i < k; i++) {
        double r = xorshift32_double();   // uniform in [0,1)
        int32_t lo = 0, hi = n - 1;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (cdf[mid] < r) lo = mid + 1;
            else hi = mid;
        }
        q[i] = pool[lo];
    }
    free(cdf);
    return q;
}

static int32_t *make_query_miss(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    if (n <= 0) { free(q); return NULL; }

    int32_t max_val = pool[n-1];
    int32_t min_val = pool[0];

    for (int32_t i = 0; i < k; i++) {
        if (n > 1) {
            int32_t idx = xorshift32() % (n - 1);
            if (pool[idx+1] - pool[idx] > 1) {
                int32_t gap = pool[idx+1] - pool[idx] - 1;
                q[i] = pool[idx] + 1 + (xorshift32() % gap);
                continue;
            }
        }
        /* Fallback if no internal mathematical gap exists */
        if (xorshift32() % 2)
            q[i] = min_val - 1 - (xorshift32() % 100);
        else
            q[i] = max_val + 1 + (xorshift32() % 100);
    }
    return q;
}

static int32_t *make_query_adversarial(int32_t *pool, int32_t n, int32_t k) {
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    int32_t max_val = (n > 0) ? pool[n-1] : 0;
    for (int32_t i = 0; i < k; i++) q[i] = max_val + 1 + i;
    return q;
}

static int32_t *gen_queries(QueryPattern pat, int32_t *pool, int32_t n, int32_t k) {
    switch (pat) {
        case QPAT_RANDOM:      return make_query_hits(pool, n, k);
        case QPAT_SEQUENTIAL:  return make_query_sequential(pool, n, k);
        case QPAT_HOTSPOT:     return make_query_hotspot(pool, n, k);
        case QPAT_ZIPFIAN:     return make_query_zipfian(pool, n, k);
        case QPAT_MISS:        return make_query_miss(pool, n, k);
        case QPAT_ADVERSARIAL: return make_query_adversarial(pool, n, k);
    }
    return make_query_hits(pool, n, k);
}

/* ── INSERT key generation (keys are NEW, interleaved within existing pool gaps) ─ */
static int32_t *gen_insert_keys(QueryPattern pat, int32_t *pool, int32_t n, int32_t k) {
    if (n <= 0) return make_query_adversarial(pool, n, k);
    int32_t max_val = pool[n-1];
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;

    switch (pat) {
        case QPAT_RANDOM:
            for (int32_t i = 0; i < k; i++) {
                int32_t idx = xorshift32() % n;
                if (idx < n - 1 && pool[idx+1] - pool[idx] > 1) {
                    int32_t gap = pool[idx+1] - pool[idx] - 1;
                    q[i] = pool[idx] + 1 + (xorshift32() % gap);
                } else {
                    q[i] = max_val + 1 + (xorshift32() % (k * 10));
                }
            }
            break;

        case QPAT_SEQUENTIAL:
            for (int32_t i = 0; i < k; i++) {
                int32_t seg = i % n;
                if (seg < n - 1 && pool[seg+1] - pool[seg] > 1) {
                    q[i] = pool[seg] + 1;
                } else {
                    q[i] = max_val + 1 + i * 3;
                }
            }
            break;

        case QPAT_HOTSPOT: {
            int32_t hot_lo = (int32_t)(n * 0.90);
            int32_t hot_range = n - hot_lo;
            if (hot_range <= 0) { hot_lo = 0; hot_range = n; }

            for (int32_t i = 0; i < k; i++) {
                if (xorshift32() % 10 < 8) {
                    int32_t idx = hot_lo + (xorshift32() % hot_range);
                    if (idx < n - 1 && pool[idx+1] - pool[idx] > 1) {
                        int32_t gap = pool[idx+1] - pool[idx] - 1;
                        q[i] = pool[idx] + 1 + (xorshift32() % gap);
                    } else {
                        q[i] = pool[idx] + 1;
                    }
                } else {
                    int32_t idx = xorshift32() % n;
                    if (idx < n - 1 && pool[idx+1] - pool[idx] > 1) {
                        int32_t gap = pool[idx+1] - pool[idx] - 1;
                        q[i] = pool[idx] + 1 + (xorshift32() % gap);
                    } else {
                        q[i] = max_val + 1 + (xorshift32() % (k * 10));
                    }
                }
            }
            break;
        }

        case QPAT_ZIPFIAN: {
            double *cdf = malloc(n * sizeof *cdf);
            if (!cdf) {
                /* Safe fallback logic if CDF memory allocation completely fails */
                for (int32_t i = 0; i < k; i++) {
                    int32_t idx = xorshift32() % n;
                    if (idx < n - 1 && pool[idx+1] - pool[idx] > 1) {
                        int32_t gap = pool[idx+1] - pool[idx] - 1;
                        q[i] = pool[idx] + 1 + (xorshift32() % gap);
                    } else {
                        q[i] = max_val + 1 + (xorshift32() % (k * 10));
                    }
                }
                break;
            }
            double z = 0.0;
            for (int32_t i = 0; i < n; i++) z += 1.0 / (i + 1);
            double cum = 0.0;
            for (int32_t i = 0; i < n; i++) {
                cum += 1.0 / ((i + 1) * z);
                cdf[i] = cum;
            }
            for (int32_t i = 0; i < k; i++) {
                double r = xorshift32_double();
                int32_t lo = 0, hi = n - 1;
                while (lo < hi) {
                    int32_t mid = (lo + hi) >> 1;
                    if (cdf[mid] < r) lo = mid + 1;
                    else hi = mid;
                }
                if (lo < n - 1 && pool[lo+1] - pool[lo] > 1) {
                    int32_t gap = pool[lo+1] - pool[lo] - 1;
                    q[i] = pool[lo] + 1 + (xorshift32() % gap);
                } else if (lo > 0 && pool[lo] - pool[lo-1] > 1) {
                    int32_t gap = pool[lo] - pool[lo-1] - 1;
                    q[i] = pool[lo-1] + 1 + (xorshift32() % gap);
                } else {
                    q[i] = max_val + 1 + (xorshift32() % (k * 10));
                }
            }
            free(cdf);
            break;
        }

        case QPAT_ADVERSARIAL:
            for (int32_t i = 0; i < k; i++)
                q[i] = max_val + 1 + i;
            break;

        default:
            for (int32_t i = 0; i < k; i++) {
                int32_t idx = xorshift32() % n;
                if (idx < n - 1 && pool[idx+1] - pool[idx] > 1) {
                    int32_t gap = pool[idx+1] - pool[idx] - 1;
                    q[i] = pool[idx] + 1 + (xorshift32() % gap);
                } else {
                    q[i] = max_val + 1 + (xorshift32() % (k * 10));
                }
            }
    }
    return q;
}

/* ── DELETE key generation (keys EXIST in pool, except adversarial) ─ */
static int32_t *gen_delete_keys(QueryPattern pat, int32_t *pool, int32_t n, int32_t k) {
    if (n <= 0) return make_query_adversarial(pool, n, k);
    int32_t *q = malloc(k * sizeof *q);
    if (!q) return NULL;
    int32_t max_val = pool[n-1];
    
    switch (pat) {
        case QPAT_RANDOM:
            for (int32_t i = 0; i < k; i++) q[i] = pool[xorshift32() % n];
            break;
        case QPAT_SEQUENTIAL:
            for (int32_t i = 0; i < k; i++) q[i] = pool[i % n];
            break;
        case QPAT_HOTSPOT: {
            int32_t hot_lo = (int32_t)(n * 0.90);
            int32_t hot_range = n - hot_lo;
            if (hot_range <= 0) { hot_lo = 0; hot_range = n; }
            for (int32_t i = 0; i < k; i++) {
                if (xorshift32() % 10 < 8)
                    q[i] = pool[hot_lo + (xorshift32() % hot_range)];
                else
                    q[i] = pool[xorshift32() % n];
            }
            break;
        }
        case QPAT_ZIPFIAN: {
            double *cdf = malloc(n * sizeof *cdf);
            if (!cdf) {
                for (int32_t i = 0; i < k; i++) q[i] = pool[xorshift32() % n];
                break;
            }
            double z = 0.0;
            for (int32_t i = 0; i < n; i++) z += 1.0 / (i + 1);
            double cum = 0.0;
            for (int32_t i = 0; i < n; i++) {
                cum += 1.0 / ((i + 1) * z);
                cdf[i] = cum;
            }
            for (int32_t i = 0; i < k; i++) {
                double r = xorshift32_double();
                int32_t lo = 0, hi = n - 1;
                while (lo < hi) {
                    int32_t mid = (lo + hi) >> 1;
                    if (cdf[mid] < r) lo = mid + 1;
                    else hi = mid;
                }
                q[i] = pool[lo];
            }
            free(cdf);
            break;
        }
        case QPAT_ADVERSARIAL:
            // Delete the largest keys first – the "actual last node"
            for (int32_t i = 0; i < k; i++) {
                int32_t idx = n - 1 - (i % n);   // wrap around if k > n
                q[i] = pool[idx];
            }
            break;
        default:
            // Fallback to random existing keys
            for (int32_t i = 0; i < k; i++) q[i] = pool[xorshift32() % n];
            break;
    }
    return q;
}

/* ── CSV output ──────────────────────────────────────────────────────── */
static void csv_header(FILE *f) {
    fprintf(f,
        "section,pattern,n,nops,param_set,label,"
        "segment_size,shortcuts_per_junction,max_skip_level,level_probability,"
        "local_interval,sub_interval,t_j,soft_pct,hard_pct,"
        "flagged_ratio_limit,min_seg_len_pct,max_suboptimal_segments,"
        "min_suboptimal_events_before_global,emergency_hard_segment_ratio,"
        "stop_crash_local_sub,block_size,"
        "jli_avg_us,jli_min_us,jli_max_us,"
        "jli_p50_us,jli_p95_us,jli_p99_us,"
        "jli_stddev_us,jli_jitter_us,"
        "jli_ci_lo_us,jli_ci_hi_us,"
        "sl_avg_us,sl_min_us,sl_max_us,"
        "sl_p50_us,sl_p95_us,sl_p99_us,"
        "sl_stddev_us,sl_jitter_us,"
        "sl_ci_lo_us,sl_ci_hi_us,"
        "ratio,win_rate,"
        "jli_mem_bytes,sl_mem_bytes,"
        "jli_steps_avg,jli_steps_p99,"
        "jli_dissolve_rate,jli_rebuild_rate,"
        "jli_local_scan,jli_sub_scan,"
        "jli_local_rebuild,jli_sub_rebuild,jli_global_rebuild,"
        "jli_rebuild_node_touches,"
        "jli_effective_cost,jli_efficiency,"
        "mem_ratio,repetition,seed,base_seed,"
        "repetitions_count,section2,pattern2,n2,nops2,eval_id\n");
}

typedef struct {
    const char *section;
    const char *pattern;
    int32_t n;
    int32_t nops;
    int32_t param_set;
    int32_t segment_size;
    int32_t shortcuts_per_junction;
    int32_t max_skip_level;
    double  level_probability;

    // ── Tuning parameters ──
    int32_t local_interval;
    int32_t sub_interval;
    double  t_j;
    double  soft_pct;
    double  hard_pct;
    double  flagged_ratio_limit;
    double  min_seg_len_pct;
    double  max_suboptimal_segments;
    int32_t min_suboptimal_events_before_global;
    double  emergency_hard_segment_ratio;
    double  stop_crash_local_sub;
    int32_t block_size;

    Stats   jli_s, sl_s;
    double  win;
    int64_t jli_mem, sl_mem;
    double  jli_steps_avg, jli_steps_p99;
    double  jli_dissolve_rate, jli_rebuild_rate;
    double  mem_ratio;
    int     repetition;
    int     seed, base_seed;
    int     repetitions_count;
    int     eval_id;
    int64_t jli_local_scan;
    int64_t jli_sub_scan;
    int64_t jli_local_rebuild;
    int64_t jli_sub_rebuild;
    int64_t jli_global_rebuild;
    int64_t jli_rebuild_node_touches;
} Row;

static void csv_row(FILE *f, const Row *r) {
    char label[256];
    snprintf(label, sizeof label, "%d-%d-%d-%.2f",
             r->segment_size, r->shortcuts_per_junction,
             r->max_skip_level, r->level_probability);
    double ratio = r->sl_s.p50 > 0 ? r->jli_s.p50 / r->sl_s.p50 : 0.0;
    double mem_ratio = r->sl_mem > 0 ? (double)r->jli_mem / r->sl_mem : 0.0;
    double jli_eff_cost = r->jli_s.avg * r->jli_mem;
    double jli_efficiency = (jli_eff_cost > 0.0) ? 1.0 / jli_eff_cost : 0.0;

    fprintf(f,
        "%s,%s,%d,%d,%d,%s,"
        "%d,%d,%d,%.6f,"
        "%d,%d,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,"
        "%d,%.6f,%.6f,%d,"
        "%.5f,%.5f,%.5f,"
        "%.5f,%.5f,%.5f,"
        "%.5f,%.5f,"
        "%.5f,%.5f,"
        "%.5f,%.5f,%.5f,"
        "%.5f,%.5f,%.5f,"
        "%.5f,%.5f,"
        "%.5f,%.5f,"
        "%.6f,%.6f,"
        "%lld,%lld,"
        "%.2f,%.2f,"
        "%.6f,%.6f,"
        "%lld,%lld,"
        "%lld,%lld,%lld,"
        "%lld,"
        "%.6e,%.6e,"
        "%.6f,%d,%d,%d,"
        "%d,%s,%s,%d,%d,%d\n",
        r->section, r->pattern, r->n, r->nops, r->param_set, label,
        r->segment_size, r->shortcuts_per_junction,
        r->max_skip_level, r->level_probability,
        // Tuning parameters
        r->local_interval, r->sub_interval,
        r->t_j, r->soft_pct, r->hard_pct,
        r->flagged_ratio_limit, r->min_seg_len_pct,
        r->max_suboptimal_segments,
        r->min_suboptimal_events_before_global,
        r->emergency_hard_segment_ratio,
        r->stop_crash_local_sub,
        r->block_size,
        // Performance metrics (unchanged)
        r->jli_s.avg, r->jli_s.vmin, r->jli_s.vmax,
        r->jli_s.p50, r->jli_s.p95, r->jli_s.p99,
        r->jli_s.sdev, r->jli_s.jitter,
        r->jli_s.ci_lo, r->jli_s.ci_hi,
        r->sl_s.avg,  r->sl_s.vmin,  r->sl_s.vmax,
        r->sl_s.p50,  r->sl_s.p95,   r->sl_s.p99,
        r->sl_s.sdev, r->sl_s.jitter,
        r->sl_s.ci_lo, r->sl_s.ci_hi,
        ratio, r->win,
        (long long)r->jli_mem, (long long)r->sl_mem,
        r->jli_steps_avg, r->jli_steps_p99,
        r->jli_dissolve_rate, r->jli_rebuild_rate,
        (long long)r->jli_local_scan,
        (long long)r->jli_sub_scan,
        (long long)r->jli_local_rebuild,
        (long long)r->jli_sub_rebuild,
        (long long)r->jli_global_rebuild,
        (long long)r->jli_rebuild_node_touches,
        jli_eff_cost, jli_efficiency,
        mem_ratio, r->repetition, r->seed, r->base_seed,
        r->repetitions_count, r->section, r->pattern, r->n, r->nops, r->eval_id);
}
/* ── Table printer ──────────────────────────────────────────────────── */
static void print_table_header(void) {
    printf("\n%-11s %-10s %8s  "
           "%8s %7s %7s  "
           "%8s %7s %7s  "
           "%7s %6s %6s  "
           "%10s %10s  "
           "%8s %8s\n",
           "section","pattern","n",
           "jli_p50","jli_p95","jli_ci",
           "sl_p50","sl_p95","sl_ci",
           "ratio","win%","mem×",
           "dissolves/del","rebuilds/op",
           "steps_avg","steps_p99");
    printf("%s\n", "--------------------------------------------------------------"
                   "--------------------------------------------------------------"
                   "--------------------------------");
}

static void print_table_row(const Row *r) {
    double mem_ratio = r->sl_mem > 0 ? (double)r->jli_mem / r->sl_mem : 0.0;
    double ratio = r->sl_s.p50 > 0 ? r->jli_s.p50 / r->sl_s.p50 : 0.0;
    char   ci_str[32], sl_ci_str[32];
    snprintf(ci_str,    sizeof ci_str,    "±%.3f", (r->jli_s.ci_hi - r->jli_s.p50));
    snprintf(sl_ci_str, sizeof sl_ci_str, "±%.3f", (r->sl_s.ci_hi  - r->sl_s.p50));
    printf("%-11s %-10s %8d  "
           "%8.3f %7.3f %7s  "
           "%8.3f %7.3f %7s  "
           "%7.3f %5.1f%% %6.2f×  "
           "%13.5f %11.5f  "
           "%8.1f %8.1f\n",
           r->section, r->pattern, r->n,
           r->jli_s.p50, r->jli_s.p95, ci_str,
           r->sl_s.p50,  r->sl_s.p95,  sl_ci_str,
           ratio, r->win * 100.0, mem_ratio,
           r->jli_dissolve_rate, r->jli_rebuild_rate,
           r->jli_steps_avg, r->jli_steps_p99);
}

/* ═══════════════════════════════════════════════════════════════════════
   PARAMETER STRUCTS
   ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int32_t segment_size;
    int32_t shortcuts_per_junction;
    int32_t max_skip_level;
} CoreParams;

typedef struct {
    double  level_probability;
    int32_t local_interval;
    int32_t sub_interval;
    double  t_j;
    double  soft_pct;
    double  hard_pct;
    double  flagged_ratio_limit;
    double  min_seg_len_pct;
    double  max_suboptimal_segments;
    int32_t min_suboptimal_events_before_global;
    double  emergency_hard_segment_ratio;
    double  stop_crash_local_sub;
    int32_t block_size;
} TuningParams;

static const CoreParams DEFAULT_BUILD  = {128, 4, 0};
static const CoreParams DEFAULT_STATIC = {256, 6, 0};
static const CoreParams DEFAULT_INSERT = {256, 4, 0};
static const CoreParams DEFAULT_DELETE = {256, 4, 0};

static const TuningParams DEFAULT_TUNING = {
    .level_probability = 0.5,
    .local_interval = 1000,
    .sub_interval   = 2500,
    .t_j            = 0.30,
    .soft_pct       = 0.25,
    .hard_pct       = 0.50,
    .flagged_ratio_limit = 0.30,
    .min_seg_len_pct     = 0.30,
    .max_suboptimal_segments = 0.25,
    .min_suboptimal_events_before_global = 5,
    .emergency_hard_segment_ratio = 0.95,
    .stop_crash_local_sub = 0.08,
    .block_size = 4,
};

#define GET_CORE_PARAMS(section_idx, default_set) \
    (cfg->core_overrides[section_idx] ? cfg->core_values[section_idx] : (default_set))

/* ═══════════════════════════════════════════════════════════════════════
   BENCHMARK RUNNERS
   ═══════════════════════════════════════════════════════════════════════ */

static JLI *create_jli(CoreParams core, const TuningParams *tp, int32_t n) {
    if (!tp) tp = &DEFAULT_TUNING;
    /* JLI block skip-list max level: ceil(log2(num_blocks))
       where num_blocks = ceil(n / (segment_size * block_m)).
       level_probability is fixed at 0.5 for a fair, principled comparison. */
    int jli_max_lvl;
    if (core.max_skip_level > 0) {
        // Use user-supplied value, clamped to safe bounds
        jli_max_lvl = core.max_skip_level;
        if (jli_max_lvl < 1) jli_max_lvl = 1;
        if (jli_max_lvl > 64) jli_max_lvl = 64;
    } else {
        // Automatic calculation (original behaviour)
        jli_max_lvl = bench_jli_block_max_level(n, core.segment_size,
                                                 tp->block_size > 0 ? tp->block_size : 4);
    }
    return jli_create(core.segment_size,
                      core.shortcuts_per_junction,
                      jli_max_lvl,
                      0.5,  /* fixed: p = 0.5 for both structures */
                      tp->local_interval,
                      tp->sub_interval,
                      tp->t_j,
                      tp->soft_pct,
                      tp->hard_pct,
                      tp->flagged_ratio_limit,
                      tp->min_seg_len_pct,
                      tp->max_suboptimal_segments,
                      tp->min_suboptimal_events_before_global,
                      tp->emergency_hard_segment_ratio,
                      tp->stop_crash_local_sub,
                      tp->block_size
                    );
}

/* ── BUILD ── */
static void run_build(int32_t n, CoreParams core, int nruns, int32_t *pool,
                      Stats *jli_out, Stats *sl_out,
                      int64_t *jli_mem_out, int64_t *sl_mem_out,
                      const TuningParams *tp) {
    memset(jli_out, 0, sizeof *jli_out);
    memset(sl_out,  0, sizeof *sl_out);
    *jli_mem_out = *sl_mem_out = 0;

    for (int r = 0; r < nruns; r++) {
        JLI *jli = create_jli(core, tp, n);
        uint64_t t0 = ns_now();
        jli_build_from_sorted(jli, pool, n);
        uint64_t dt = ns_now() - t0;
        stats_record(jli_out, (double)dt / 1000.0);
        if (r == nruns - 1) *jli_mem_out = jli_memory_bytes(jli);
        jli_destroy(jli);

        xorshift32_seed((user_seed ? user_seed : 42) + r);
        SL *sl = sl_new(n);
        t0 = ns_now();
        for (int32_t i = 0; i < n; i++) sl_insert(sl, pool[i], NULL,0);
        dt = ns_now() - t0;
        stats_record(sl_out, (double)dt / 1000.0);
        if (r == nruns - 1) *sl_mem_out = sl_memory_bytes(sl);
        sl_destroy(sl);
    }
}

/* ── SEARCH (STATIC) ── */
static void run_static(int32_t n, CoreParams core, int nruns, int32_t NOPS,
                       int32_t *pool, QueryPattern qpat,
                       Stats *jli_out, Stats *sl_out,
                       int64_t *jli_mem_out, int64_t *sl_mem_out,
                       double *steps_avg_out, double *steps_p99_out,
                       const TuningParams *tp) {
    memset(jli_out, 0, sizeof *jli_out);
    memset(sl_out,  0, sizeof *sl_out);
    *jli_mem_out = *sl_mem_out = 0;

    unsigned int base_seed = user_seed ? user_seed : 42;
    xorshift32_seed(base_seed);
    JLI *jli = create_jli(core, tp, n);
    jli_build_from_sorted(jli, pool, n);
    *jli_mem_out = jli_memory_bytes(jli);

    xorshift32_seed(base_seed);
    SL *sl = sl_new(n);
    for (int32_t i = 0; i < n; i++) sl_insert(sl, pool[i], NULL,0);
    *sl_mem_out = sl_memory_bytes(sl);

    int32_t *queries = gen_queries(qpat, pool, n, NOPS);

    /* ── WARM‑UP (non‑timed searches) ── */
    if (warmup_ops > 0) {
        for (int i = 0; i < warmup_ops; i++) {
            int32_t key = queries[i % NOPS];
            (void)jli_search(jli, key);
            (void)sl_search(sl, key);
        }
    }

    for (int r = 0; r < nruns; r++) {
        xorshift32_seed(base_seed + r);
        double *steps_per_op = malloc(NOPS * sizeof(double));
        uint64_t t0 = ns_now();
        for (int32_t i = 0; i < NOPS; i++) {
            jli_search(jli, queries[i]);
            steps_per_op[i] = (double)jli->last_search_steps;
        }
        uint64_t dt = ns_now() - t0;
        stats_record(jli_out, (double)dt / NOPS / 1000.0);
        qsort(steps_per_op, NOPS, sizeof(double), cmp_double);
        double sum = 0.0;
        for (int32_t i = 0; i < NOPS; i++) sum += steps_per_op[i];
        *steps_avg_out = sum / NOPS;
        *steps_p99_out = percentile(steps_per_op, NOPS, 0.99);
        free(steps_per_op);

        t0 = ns_now();
        for (int32_t i = 0; i < NOPS; i++) {
            volatile SLNode *res = sl_search(sl, queries[i]); (void)res;
        }
        dt = ns_now() - t0;
        stats_record(sl_out, (double)dt / NOPS / 1000.0);
    }
    free(queries);
    jli_destroy(jli);
    sl_destroy(sl);
}

/* ── INSERT (uses gen_insert_keys) ── */
static void run_insert(int32_t n, CoreParams core, int nruns, int32_t NOPS,
                       int32_t *pool, QueryPattern qpat,
                       Stats *jli_out, Stats *sl_out,
                       int64_t *jli_mem_out, int64_t *sl_mem_out,
                       double *dissolve_rate_out, double *rebuild_rate_out,
                       double *steps_avg_out, double *steps_p99_out,
                       int64_t *local_scan_out, int64_t *sub_scan_out,
                       int64_t *local_rebuild_out, int64_t *sub_rebuild_out,
                       int64_t *global_rebuild_out,
                       int64_t *rebuild_node_touches_out,
                       const TuningParams *tp) {
    memset(jli_out, 0, sizeof *jli_out);
    memset(sl_out,  0, sizeof *sl_out);
    *jli_mem_out = *sl_mem_out = 0;
    *dissolve_rate_out = *rebuild_rate_out = 0.0;
    *steps_avg_out = *steps_p99_out = 0.0;
    *local_scan_out = *sub_scan_out = 0;
    *local_rebuild_out = *sub_rebuild_out = *global_rebuild_out = 0;
    *rebuild_node_touches_out = 0;

    unsigned int base_seed = user_seed ? user_seed : 42;

    double steps_avg_sum = 0.0;
    double steps_p99_sum = 0.0;
    int64_t total_dissolve = 0;
    int64_t total_local_scan = 0;
    int64_t total_sub_scan = 0;
    int64_t total_local_rebuild = 0;
    int64_t total_sub_rebuild = 0;
    int64_t total_global_rebuild = 0;
    int64_t total_rebuild_node_touches = 0;

    for (int r = 0; r < nruns; r++) {
        xorshift32_seed(base_seed + r);
        int32_t *ins_keys = gen_insert_keys(qpat, pool, n, NOPS);

        /* Build JLI */
        JLI *jli = create_jli(core, tp, n);
        jli_build_from_sorted(jli, pool, n);

        /* Build SL */
        xorshift32_seed(base_seed + r);
        SL *sl = sl_new(n);
        for (int32_t i = 0; i < n; i++) sl_insert(sl, pool[i], NULL,0);

        /* ── WARM‑UP (non‑timed searches) ── */
        if (warmup_ops > 0) {
            for (int i = 0; i < warmup_ops; i++) {
                int32_t key = pool[xorshift32() % n];
                (void)jli_search(jli, key);
                (void)sl_search(sl, key);
            }
        }

        /* Timed JLI inserts */
        uint64_t t0 = ns_now();
        double *steps_per_op = malloc(NOPS * sizeof(double));
        for (int32_t i = 0; i < NOPS; i++) {
            jli_insert(jli, ins_keys[i], NULL);
            steps_per_op[i] = (double)jli->last_search_steps;
        }
        uint64_t dt = ns_now() - t0;
        stats_record(jli_out, (double)dt / NOPS / 1000.0);
        if (r == nruns - 1) *jli_mem_out = jli_memory_bytes(jli);

        qsort(steps_per_op, NOPS, sizeof(double), cmp_double);
        double sum = 0.0;
        for (int32_t i = 0; i < NOPS; i++) sum += steps_per_op[i];
        double steps_avg_per_run = sum / NOPS;
        double steps_p99_per_run = percentile(steps_per_op, NOPS, 0.99);
        steps_avg_sum += steps_avg_per_run;
        steps_p99_sum += steps_p99_per_run;
        free(steps_per_op);

        /* Collect maintenance stats */
        JLI_MaintenanceStats stats;
        jli_get_maintenance_stats(jli, &stats);
        total_dissolve += jli->dissolve_count;
        total_local_scan += stats.local_scan_count;
        total_sub_scan += stats.sub_scan_count;
        total_local_rebuild += stats.local_rebuild_count;
        total_sub_rebuild += stats.sub_rebuild_count;
        total_global_rebuild += stats.global_rebuild_count;
        total_rebuild_node_touches += stats.local_node_touches + stats.sub_node_touches;

        jli_destroy(jli);
        free(ins_keys);

        /* Timed SL inserts (same insert keys – we generate new ones for SL to be fair) */
        xorshift32_seed(base_seed + r);
        int32_t *sl_ins_keys = gen_insert_keys(qpat, pool, n, NOPS);
        t0 = ns_now();
        for (int32_t i = 0; i < NOPS; i++) sl_insert(sl, sl_ins_keys[i], NULL,0);
        dt = ns_now() - t0;
        stats_record(sl_out, (double)dt / NOPS / 1000.0);
        if (r == nruns - 1) *sl_mem_out = sl_memory_bytes(sl);
        sl_destroy(sl);
        free(sl_ins_keys);
    }

    /* Compute aggregate values */
    int64_t total_ops = (int64_t)nruns * NOPS;
    *dissolve_rate_out = (double)total_dissolve / total_ops;
    *rebuild_rate_out = (double)(total_local_rebuild + total_sub_rebuild + total_global_rebuild) / total_ops;
    *steps_avg_out = steps_avg_sum / nruns;
    *steps_p99_out = steps_p99_sum / nruns;
    *local_scan_out = total_local_scan;
    *sub_scan_out = total_sub_scan;
    *local_rebuild_out = total_local_rebuild;
    *sub_rebuild_out = total_sub_rebuild;
    *global_rebuild_out = total_global_rebuild;
    *rebuild_node_touches_out = total_rebuild_node_touches;
}

/* ── DELETE (uses gen_delete_keys) ── */
static void run_delete(int32_t n, CoreParams core, int nruns, int32_t NOPS,
                       int32_t *pool, QueryPattern qpat,
                       Stats *jli_out, Stats *sl_out,
                       int64_t *jli_mem_out, int64_t *sl_mem_out,
                       double *dissolve_rate_out, double *rebuild_rate_out,
                       double *steps_avg_out, double *steps_p99_out,
                       int64_t *local_scan_out, int64_t *sub_scan_out,
                       int64_t *local_rebuild_out, int64_t *sub_rebuild_out,
                       int64_t *global_rebuild_out,
                       int64_t *rebuild_node_touches_out,
                       const TuningParams *tp) {
    memset(jli_out, 0, sizeof *jli_out);
    memset(sl_out,  0, sizeof *sl_out);
    *jli_mem_out = *sl_mem_out = 0;
    *dissolve_rate_out = *rebuild_rate_out = 0.0;
    *steps_avg_out = *steps_p99_out = 0.0;
    *local_scan_out = *sub_scan_out = 0;
    *local_rebuild_out = *sub_rebuild_out = *global_rebuild_out = 0;
    *rebuild_node_touches_out = 0;

    int32_t nops = NOPS < n ? NOPS : n / 2;
    unsigned int base_seed = user_seed ? user_seed : 42;

    double steps_avg_sum = 0.0;
    double steps_p99_sum = 0.0;
    int64_t total_dissolve = 0;
    int64_t total_local_scan = 0;
    int64_t total_sub_scan = 0;
    int64_t total_local_rebuild = 0;
    int64_t total_sub_rebuild = 0;
    int64_t total_global_rebuild = 0;
    int64_t total_rebuild_node_touches = 0;

    for (int r = 0; r < nruns; r++) {
        xorshift32_seed(base_seed + r);
        int32_t *del_keys = gen_delete_keys(qpat, pool, n, nops);

        JLI *jli = create_jli(core, tp, n);
        jli_build_from_sorted(jli, pool, n);

        xorshift32_seed(base_seed + r);
        SL *sl = sl_new(n);
        for (int32_t i = 0; i < n; i++) sl_insert(sl, pool[i], NULL,0);

        /* ── WARM‑UP (searches only) ── */
        if (warmup_ops > 0) {
            for (int i = 0; i < warmup_ops; i++) {
                int32_t key = pool[xorshift32() % n];
                (void)jli_search(jli, key);
                (void)sl_search(sl, key);
            }
        }

        uint64_t t0 = ns_now();
        double *steps_per_op = malloc(nops * sizeof(double));
        for (int32_t i = 0; i < nops; i++) {
            jli_delete(jli, del_keys[i]);
            steps_per_op[i] = (double)jli->last_search_steps;
        }
        uint64_t dt = ns_now() - t0;
        stats_record(jli_out, (double)dt / nops / 1000.0);
        if (r == nruns - 1) *jli_mem_out = jli_memory_bytes(jli);

        qsort(steps_per_op, nops, sizeof(double), cmp_double);
        double sum = 0.0;
        for (int32_t i = 0; i < nops; i++) sum += steps_per_op[i];
        double steps_avg_per_run = sum / nops;
        double steps_p99_per_run = percentile(steps_per_op, nops, 0.99);
        steps_avg_sum += steps_avg_per_run;
        steps_p99_sum += steps_p99_per_run;
        free(steps_per_op);

        /* Collect maintenance stats */
        JLI_MaintenanceStats stats;
        jli_get_maintenance_stats(jli, &stats);
        total_dissolve += jli->dissolve_count;
        total_local_scan += stats.local_scan_count;
        total_sub_scan += stats.sub_scan_count;
        total_local_rebuild += stats.local_rebuild_count;
        total_sub_rebuild += stats.sub_rebuild_count;
        total_global_rebuild += stats.global_rebuild_count;
        total_rebuild_node_touches += stats.local_node_touches + stats.sub_node_touches;

        jli_destroy(jli);
        free(del_keys);

        /* SL timed deletes */
        xorshift32_seed(base_seed + r);
        int32_t *sl_del_keys = gen_delete_keys(qpat, pool, n, nops);
        t0 = ns_now();
        for (int32_t i = 0; i < nops; i++) sl_delete(sl, sl_del_keys[i]);
        dt = ns_now() - t0;
        stats_record(sl_out, (double)dt / nops / 1000.0);
        if (r == nruns - 1) *sl_mem_out = sl_memory_bytes(sl);
        sl_destroy(sl);
        free(sl_del_keys);
    }

    /* Compute aggregate values */
    int64_t total_ops = (int64_t)nruns * nops;
    *dissolve_rate_out = (double)total_dissolve / total_ops;
    *rebuild_rate_out = (double)(total_local_rebuild + total_sub_rebuild + total_global_rebuild) / total_ops;
    *steps_avg_out = steps_avg_sum / nruns;
    *steps_p99_out = steps_p99_sum / nruns;
    *local_scan_out = total_local_scan;
    *sub_scan_out = total_sub_scan;
    *local_rebuild_out = total_local_rebuild;
    *sub_rebuild_out = total_sub_rebuild;
    *global_rebuild_out = total_global_rebuild;
    *rebuild_node_touches_out = total_rebuild_node_touches;
}

/* ═══════════════════════════════════════════════════════════════════════
   MAIN SWEEP
   ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int32_t *ns; int n_ns;
    int nruns;
    int32_t NOPS;
    bool nops_set;
    bool do_build, do_static, do_insert, do_delete;
    bool csv_out;
    char csv_path[256];
    bool print_table;
    bool split_csv;
    char pattern[64];
    bool core_overrides[4];
    CoreParams core_values[4];
    bool tuning_overrides[4];
    TuningParams tuning_values[4];
} Config;

static Config cfg = {
    .nruns = 3,
    .NOPS = 5000,
    .nops_set = false,
    .do_build = true,
    .do_static = true,
    .do_insert = true,
    .do_delete = true,
    .csv_out = true,
    .print_table = true,
    .split_csv = true,
    .core_overrides = { false },
    .tuning_overrides = { false },
    .pattern = "",
};

static bool parse_core_params(const char *s, CoreParams *cp) {
    return sscanf(s, "%d,%d,%d",
                  &cp->segment_size,
                  &cp->shortcuts_per_junction,
                  &cp->max_skip_level) == 3;
}

static bool parse_tuning_params(const char *s, TuningParams *tp) {
    int n = sscanf(s, "%lf,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%d,%lf,%lf,%d",
                  &tp->level_probability,
                  &tp->local_interval,
                  &tp->sub_interval,
                  &tp->t_j,
                  &tp->soft_pct,
                  &tp->hard_pct,
                  &tp->flagged_ratio_limit,
                  &tp->min_seg_len_pct,
                  &tp->max_suboptimal_segments,          // 9th value
                  &tp->min_suboptimal_events_before_global,
                  &tp->emergency_hard_segment_ratio,
                  &tp->stop_crash_local_sub,
                  &tp->block_size);
    if (n != 13) return false;

    /* level_probability is fixed at 0.5 for both JLI and skip list.
       Any value from the CSV or CLI is silently overridden here. */
    tp->level_probability = 0.5;

    // Enforce 0 ≤ max_suboptimal_segments ≤ 1
    if (tp->max_suboptimal_segments > 1.0)
        tp->max_suboptimal_segments = 1.0;
    if (tp->max_suboptimal_segments < 0.0)
        tp->max_suboptimal_segments = 0.0;

    return true;
}

static void sweep_section(const Config *cfg) {
    int32_t  ns[]   = {10000, 50000, 100000, 250000, 500000, 750000, 1000000,5000000,10000000};
    int      n_ns   = (int)(sizeof ns / sizeof ns[0]);
    int param_set = 0;

    /* ── BUILD ── */
    if (cfg->do_build) {
        if (cfg->pattern[0] && strcasecmp(cfg->pattern, "sorted") != 0) goto skip_build;
        FILE *csv = NULL;
        if (cfg->csv_out) {
            char fname[256];
            snprintf(fname, sizeof fname, cfg->split_csv ? "build_%s" : "%s", cfg->csv_path);
            csv = fopen(fname, "a");
            if (csv) {
                fseek(csv, 0, SEEK_END);
                long sz = ftell(csv);
                if (sz == 0) csv_header(csv);
            }
        }
        if (cfg->print_table) { printf("\nBUILD SECTION\n"); print_table_header(); }
        CoreParams core = GET_CORE_PARAMS(0, DEFAULT_BUILD);
        const TuningParams *tp = cfg->tuning_overrides[0] ? &cfg->tuning_values[0] : NULL;
        for (int ni = 0; ni < n_ns; ni++) {
            int32_t n = ns[ni];
            if (fixed_n && n != fixed_n) continue;
            int32_t *pool = make_pool(n, user_seed ? user_seed : 42 + ni);
            Stats js = {0}, ss = {0}; int64_t jm = 0, sm = 0;
            run_build(n, core, cfg->nruns, pool, &js, &ss, &jm, &sm, tp);
            stats_compute(&js); stats_compute(&ss);

            int effective_max_level;
            if (core.max_skip_level > 0) {
                effective_max_level = core.max_skip_level;
                if (effective_max_level < 1) effective_max_level = 1;
                if (effective_max_level > 64) effective_max_level = 64;
            } else {
                effective_max_level = bench_jli_block_max_level(
                    n, core.segment_size,
                    tp ? (tp->block_size > 0 ? tp->block_size : 4) : 4);
            }

            Row row = {
                .section = "BUILD",
                .pattern = "sorted",
                .n = n,
                .nops = n,
                .param_set = param_set++,
                .segment_size = core.segment_size,
                .shortcuts_per_junction = core.shortcuts_per_junction,
                .max_skip_level = effective_max_level,
                .level_probability = 0.5,

                // Tuning parameters
                .local_interval = tp ? tp->local_interval : DEFAULT_TUNING.local_interval,
                .sub_interval   = tp ? tp->sub_interval   : DEFAULT_TUNING.sub_interval,
                .t_j            = tp ? tp->t_j            : DEFAULT_TUNING.t_j,
                .soft_pct       = tp ? tp->soft_pct       : DEFAULT_TUNING.soft_pct,
                .hard_pct       = tp ? tp->hard_pct       : DEFAULT_TUNING.hard_pct,
                .flagged_ratio_limit = tp ? tp->flagged_ratio_limit : DEFAULT_TUNING.flagged_ratio_limit,
                .min_seg_len_pct     = tp ? tp->min_seg_len_pct     : DEFAULT_TUNING.min_seg_len_pct,
                .max_suboptimal_segments = tp ? tp->max_suboptimal_segments : DEFAULT_TUNING.max_suboptimal_segments,
                .min_suboptimal_events_before_global =
                    tp ? tp->min_suboptimal_events_before_global : DEFAULT_TUNING.min_suboptimal_events_before_global,
                .emergency_hard_segment_ratio =
                    tp ? tp->emergency_hard_segment_ratio : DEFAULT_TUNING.emergency_hard_segment_ratio,
                .stop_crash_local_sub = tp ? tp->stop_crash_local_sub : DEFAULT_TUNING.stop_crash_local_sub,
                .block_size = tp ? tp->block_size : DEFAULT_TUNING.block_size,

                .jli_s = js,
                .sl_s  = ss,
                .win   = win_rate(&js, &ss),
                .jli_mem = jm,
                .sl_mem  = sm,
                .seed = (int)(user_seed ? user_seed : 42),
                .base_seed = (int)(user_seed ? user_seed : 42),
                .repetitions_count = cfg->nruns,
                .eval_id = param_set,
                .jli_local_scan = 0,
                .jli_sub_scan = 0,
                .jli_local_rebuild = 0,
                .jli_sub_rebuild = 0,
                .jli_global_rebuild = 0,
                .jli_rebuild_node_touches = 0
            };
            if (cfg->print_table) print_table_row(&row);
            if (csv) csv_row(csv, &row);
            free(pool);
        }
        if (csv) fclose(csv);
        skip_build:;
    }

    /* ── STATIC (SEARCH) ── */
    if (cfg->do_static) {
        const char *static_patterns[] = {"random","sequential","hotspot","zipfian","miss","adversarial"};
        FILE *csv = NULL;
        if (cfg->csv_out) {
            char fname[256];
            snprintf(fname, sizeof fname, cfg->split_csv ? "static_%s" : "%s", cfg->csv_path);
            csv = fopen(fname, "a");
            if (csv) {
                fseek(csv, 0, SEEK_END);
                long sz = ftell(csv);
                if (sz == 0) csv_header(csv);
            }
        }
        if (cfg->print_table) { printf("\nSTATIC SECTION\n"); print_table_header(); }
        CoreParams core = GET_CORE_PARAMS(1, DEFAULT_STATIC);
        const TuningParams *tp = cfg->tuning_overrides[1] ? &cfg->tuning_values[1] : NULL;
        for (int pi = 0; pi < 6; pi++) {
            const char *pat = static_patterns[pi];
            if (cfg->pattern[0] && strcasecmp(cfg->pattern, pat) != 0) continue;
            for (int ni = 0; ni < n_ns; ni++) {
                int32_t n = ns[ni]; if (fixed_n && n != fixed_n) continue;
                int32_t nops = cfg->nops_set ? cfg->NOPS : (n / 3);
                if (nops < 1) nops = 1;
                int32_t *pool = make_pool(n, user_seed ? user_seed : 42);
                qsort(pool, n, sizeof(int32_t), cmp_int32);
                Stats js = {0}, ss = {0}; int64_t jm = 0, sm = 0;
                double steps_avg = 0, steps_p99 = 0;
                run_static(n, core, cfg->nruns, nops, pool, string_to_qpat(pat),
                           &js, &ss, &jm, &sm, &steps_avg, &steps_p99, tp);
                stats_compute(&js); stats_compute(&ss);

                int effective_max_level;
                if (core.max_skip_level > 0) {
                    effective_max_level = core.max_skip_level;
                    if (effective_max_level < 1) effective_max_level = 1;
                    if (effective_max_level > 64) effective_max_level = 64;
                } else {
                    effective_max_level = bench_jli_block_max_level(
                        n, core.segment_size,
                        tp ? (tp->block_size > 0 ? tp->block_size : 4) : 4);
                }

                Row row = {
                    .section = "STATIC",
                    .pattern = pat,
                    .n = n,
                    .nops = nops,
                    .param_set = param_set++,
                    .segment_size = core.segment_size,
                    .shortcuts_per_junction = core.shortcuts_per_junction,
                    .max_skip_level = effective_max_level,
                    .level_probability = 0.5,

                    // Tuning parameters
                    .local_interval = tp ? tp->local_interval : DEFAULT_TUNING.local_interval,
                    .sub_interval   = tp ? tp->sub_interval   : DEFAULT_TUNING.sub_interval,
                    .t_j            = tp ? tp->t_j            : DEFAULT_TUNING.t_j,
                    .soft_pct       = tp ? tp->soft_pct       : DEFAULT_TUNING.soft_pct,
                    .hard_pct       = tp ? tp->hard_pct       : DEFAULT_TUNING.hard_pct,
                    .flagged_ratio_limit = tp ? tp->flagged_ratio_limit : DEFAULT_TUNING.flagged_ratio_limit,
                    .min_seg_len_pct     = tp ? tp->min_seg_len_pct     : DEFAULT_TUNING.min_seg_len_pct,
                    .max_suboptimal_segments = tp ? tp->max_suboptimal_segments : DEFAULT_TUNING.max_suboptimal_segments,
                    .min_suboptimal_events_before_global =
                        tp ? tp->min_suboptimal_events_before_global : DEFAULT_TUNING.min_suboptimal_events_before_global,
                    .emergency_hard_segment_ratio =
                        tp ? tp->emergency_hard_segment_ratio : DEFAULT_TUNING.emergency_hard_segment_ratio,
                    .stop_crash_local_sub = tp ? tp->stop_crash_local_sub : DEFAULT_TUNING.stop_crash_local_sub,
                    .block_size = tp ? tp->block_size : DEFAULT_TUNING.block_size,

                    .jli_s = js,
                    .sl_s  = ss,
                    .win   = win_rate(&js, &ss),
                    .jli_mem = jm,
                    .sl_mem  = sm,
                    .jli_steps_avg = steps_avg,
                    .jli_steps_p99 = steps_p99,
                    .mem_ratio = sm > 0 ? (double)jm / sm : 0.0,
                    .seed = (int)(user_seed ? user_seed : 42),
                    .base_seed = (int)(user_seed ? user_seed : 42),
                    .repetitions_count = cfg->nruns,
                    .eval_id = param_set,
                    .jli_local_scan = 0,
                    .jli_sub_scan = 0,
                    .jli_local_rebuild = 0,
                    .jli_sub_rebuild = 0,
                    .jli_global_rebuild = 0,
                    .jli_rebuild_node_touches = 0
                };
                if (cfg->print_table) print_table_row(&row);
                if (csv) csv_row(csv, &row);
                free(pool);
            }
        }
        if (csv) fclose(csv);
    }

    /* ── INSERT (5 patterns) ── */
    if (cfg->do_insert) {
        const char *insert_patterns[] = {"random","sequential","hotspot","zipfian","adversarial"};
        FILE *csv = NULL;
        if (cfg->csv_out) {
            char fname[256];
            snprintf(fname, sizeof fname, cfg->split_csv ? "insert_%s" : "%s", cfg->csv_path);
            csv = fopen(fname, "a");
            if (csv) { fseek(csv,0,SEEK_END); if (ftell(csv)==0) csv_header(csv); }
        }
        if (cfg->print_table) { printf("\nINSERT SECTION\n"); print_table_header(); }
        CoreParams core = GET_CORE_PARAMS(2, DEFAULT_INSERT);
        const TuningParams *tp = cfg->tuning_overrides[2] ? &cfg->tuning_values[2] : NULL;
        for (int pi = 0; pi < 5; pi++) {
            const char *pat = insert_patterns[pi];
            if (cfg->pattern[0] && strcasecmp(cfg->pattern, pat) != 0) continue;
            for (int ni = 0; ni < n_ns; ni++) {
                int32_t n = ns[ni]; if (fixed_n && n != fixed_n) continue;
                int32_t nops = cfg->nops_set ? cfg->NOPS : (n / 3);
                if (nops < 1) nops = 1;
                int32_t *pool = make_pool(n, user_seed ? user_seed : 42);
                Stats js = {0}, ss = {0}; int64_t jm = 0, sm = 0;
                double dr = 0, rr = 0, steps_avg = 0, steps_p99 = 0;
                int64_t local_scan = 0, sub_scan = 0;
                int64_t local_rebuild = 0, sub_rebuild = 0, global_rebuild = 0;
                int64_t rebuild_node_touches = 0;
                run_insert(n, core, cfg->nruns, nops, pool, string_to_qpat(pat),
                           &js, &ss, &jm, &sm, &dr, &rr, &steps_avg, &steps_p99,
                           &local_scan, &sub_scan, &local_rebuild, &sub_rebuild,
                           &global_rebuild, &rebuild_node_touches, tp);
                stats_compute(&js); stats_compute(&ss);

                int effective_max_level;
                if (core.max_skip_level > 0) {
                    effective_max_level = core.max_skip_level;
                    if (effective_max_level < 1) effective_max_level = 1;
                    if (effective_max_level > 64) effective_max_level = 64;
                } else {
                    effective_max_level = bench_jli_block_max_level(
                        n, core.segment_size,
                        tp ? (tp->block_size > 0 ? tp->block_size : 4) : 4);
                }

                Row row = {
                    .section = "INSERT",
                    .pattern = pat,
                    .n = n,
                    .nops = nops,
                    .param_set = param_set++,
                    .segment_size = core.segment_size,
                    .shortcuts_per_junction = core.shortcuts_per_junction,
                    .max_skip_level = effective_max_level,
                    .level_probability = 0.5,

                    // Tuning parameters
                    .local_interval = tp ? tp->local_interval : DEFAULT_TUNING.local_interval,
                    .sub_interval   = tp ? tp->sub_interval   : DEFAULT_TUNING.sub_interval,
                    .t_j            = tp ? tp->t_j            : DEFAULT_TUNING.t_j,
                    .soft_pct       = tp ? tp->soft_pct       : DEFAULT_TUNING.soft_pct,
                    .hard_pct       = tp ? tp->hard_pct       : DEFAULT_TUNING.hard_pct,
                    .flagged_ratio_limit = tp ? tp->flagged_ratio_limit : DEFAULT_TUNING.flagged_ratio_limit,
                    .min_seg_len_pct     = tp ? tp->min_seg_len_pct     : DEFAULT_TUNING.min_seg_len_pct,
                    .max_suboptimal_segments = tp ? tp->max_suboptimal_segments : DEFAULT_TUNING.max_suboptimal_segments,
                    .min_suboptimal_events_before_global =
                        tp ? tp->min_suboptimal_events_before_global : DEFAULT_TUNING.min_suboptimal_events_before_global,
                    .emergency_hard_segment_ratio =
                        tp ? tp->emergency_hard_segment_ratio : DEFAULT_TUNING.emergency_hard_segment_ratio,
                    .stop_crash_local_sub = tp ? tp->stop_crash_local_sub : DEFAULT_TUNING.stop_crash_local_sub,
                    .block_size = tp ? tp->block_size : DEFAULT_TUNING.block_size,

                    .jli_s = js,
                    .sl_s  = ss,
                    .win   = win_rate(&js, &ss),
                    .jli_mem = jm,
                    .sl_mem  = sm,
                    .jli_steps_avg = steps_avg,
                    .jli_steps_p99 = steps_p99,
                    .jli_dissolve_rate = dr,
                    .jli_rebuild_rate  = rr,
                    .mem_ratio = sm > 0 ? (double)jm / sm : 0.0,
                    .seed = (int)(user_seed ? user_seed : 42),
                    .base_seed = (int)(user_seed ? user_seed : 42),
                    .repetitions_count = cfg->nruns,
                    .eval_id = param_set,
                    .jli_local_scan = local_scan,
                    .jli_sub_scan = sub_scan,
                    .jli_local_rebuild = local_rebuild,
                    .jli_sub_rebuild = sub_rebuild,
                    .jli_global_rebuild = global_rebuild,
                    .jli_rebuild_node_touches = rebuild_node_touches
                };
                if (cfg->print_table) print_table_row(&row);
                if (csv) csv_row(csv, &row);
                free(pool);
            }
        }
        if (csv) fclose(csv);
    }

    /* ── DELETE (5 patterns) ── */
    if (cfg->do_delete) {
        const char *delete_patterns[] = {"random","sequential","hotspot","zipfian","adversarial"};
        FILE *csv = NULL;
        if (cfg->csv_out) {
            char fname[256];
            snprintf(fname, sizeof fname, cfg->split_csv ? "delete_%s" : "%s", cfg->csv_path);
            csv = fopen(fname, "a");
            if (csv) { fseek(csv,0,SEEK_END); if (ftell(csv)==0) csv_header(csv); }
        }
        if (cfg->print_table) { printf("\nDELETE SECTION\n"); print_table_header(); }
        CoreParams core = GET_CORE_PARAMS(3, DEFAULT_DELETE);
        const TuningParams *tp = cfg->tuning_overrides[3] ? &cfg->tuning_values[3] : NULL;
        for (int pi = 0; pi < 5; pi++) {
            const char *pat = delete_patterns[pi];
            if (cfg->pattern[0] && strcasecmp(cfg->pattern, pat) != 0) continue;
            for (int ni = 0; ni < n_ns; ni++) {
                int32_t n = ns[ni]; if (fixed_n && n != fixed_n) continue;
                int32_t nops = cfg->nops_set ? cfg->NOPS : (n / 3);
                if (nops < 1) nops = 1;
                int32_t *pool = make_pool(n, user_seed ? user_seed : 42);
                Stats js = {0}, ss = {0}; int64_t jm = 0, sm = 0;
                double dr = 0, rr = 0, steps_avg = 0, steps_p99 = 0;
                int64_t local_scan = 0, sub_scan = 0;
                int64_t local_rebuild = 0, sub_rebuild = 0, global_rebuild = 0;
                int64_t rebuild_node_touches = 0;
                run_delete(n, core, cfg->nruns, nops, pool, string_to_qpat(pat),
                           &js, &ss, &jm, &sm, &dr, &rr, &steps_avg, &steps_p99,
                           &local_scan, &sub_scan, &local_rebuild, &sub_rebuild,
                           &global_rebuild, &rebuild_node_touches, tp);
                stats_compute(&js); stats_compute(&ss);

                int effective_max_level;
                if (core.max_skip_level > 0) {
                    effective_max_level = core.max_skip_level;
                    if (effective_max_level < 1) effective_max_level = 1;
                    if (effective_max_level > 64) effective_max_level = 64;
                } else {
                    effective_max_level = bench_jli_block_max_level(
                        n, core.segment_size,
                        tp ? (tp->block_size > 0 ? tp->block_size : 4) : 4);
                }

                Row row = {
                    .section = "DELETE",
                    .pattern = pat,
                    .n = n,
                    .nops = nops,
                    .param_set = param_set++,
                    .segment_size = core.segment_size,
                    .shortcuts_per_junction = core.shortcuts_per_junction,
                    .max_skip_level = effective_max_level,
                    .level_probability = 0.5,

                    // Tuning parameters
                    .local_interval = tp ? tp->local_interval : DEFAULT_TUNING.local_interval,
                    .sub_interval   = tp ? tp->sub_interval   : DEFAULT_TUNING.sub_interval,
                    .t_j            = tp ? tp->t_j            : DEFAULT_TUNING.t_j,
                    .soft_pct       = tp ? tp->soft_pct       : DEFAULT_TUNING.soft_pct,
                    .hard_pct       = tp ? tp->hard_pct       : DEFAULT_TUNING.hard_pct,
                    .flagged_ratio_limit = tp ? tp->flagged_ratio_limit : DEFAULT_TUNING.flagged_ratio_limit,
                    .min_seg_len_pct     = tp ? tp->min_seg_len_pct     : DEFAULT_TUNING.min_seg_len_pct,
                    .max_suboptimal_segments = tp ? tp->max_suboptimal_segments : DEFAULT_TUNING.max_suboptimal_segments,
                    .min_suboptimal_events_before_global =
                        tp ? tp->min_suboptimal_events_before_global : DEFAULT_TUNING.min_suboptimal_events_before_global,
                    .emergency_hard_segment_ratio =
                        tp ? tp->emergency_hard_segment_ratio : DEFAULT_TUNING.emergency_hard_segment_ratio,
                    .stop_crash_local_sub = tp ? tp->stop_crash_local_sub : DEFAULT_TUNING.stop_crash_local_sub,
                    .block_size = tp ? tp->block_size : DEFAULT_TUNING.block_size,

                    .jli_s = js,
                    .sl_s  = ss,
                    .win   = win_rate(&js, &ss),
                    .jli_mem = jm,
                    .sl_mem  = sm,
                    .jli_steps_avg = steps_avg,
                    .jli_steps_p99 = steps_p99,
                    .jli_dissolve_rate = dr,
                    .jli_rebuild_rate  = rr,
                    .mem_ratio = sm > 0 ? (double)jm / sm : 0.0,
                    .seed = (int)(user_seed ? user_seed : 42),
                    .base_seed = (int)(user_seed ? user_seed : 42),
                    .repetitions_count = cfg->nruns,
                    .eval_id = param_set,
                    .jli_local_scan = local_scan,
                    .jli_sub_scan = sub_scan,
                    .jli_local_rebuild = local_rebuild,
                    .jli_sub_rebuild = sub_rebuild,
                    .jli_global_rebuild = global_rebuild,
                    .jli_rebuild_node_touches = rebuild_node_touches
                };
                if (cfg->print_table) print_table_row(&row);
                if (csv) csv_row(csv, &row);
                free(pool);
            }
        }
        if (csv) fclose(csv);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   COMMAND‑LINE PARSING (unchanged)
   ═══════════════════════════════════════════════════════════════════════ */

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --section BUILD|STATIC|INSERT|DELETE|ALL\n"
        "  --pattern <name>          run only this pattern (e.g., random, sequential, ...)\n"
        "  --runs N                  number of repeats (default 3)\n"
        "  --nops N                  operations per timing run (default 5000)\n"
        "  --out FILE.csv            output CSV file (default bench_out.csv)\n"
        "  --no-csv                  disable CSV output\n"
        "  --no-table                disable console table\n"
        "  --split-csv               write separate files per section (default)\n"
        "  --single-csv              write all results to one file\n"
        "  --n N                     run only a specific problem size\n"
        "  --seed S                  random seed\n"
        "  --warmup N               number of warm‑up operations (non‑timed, default 0)\n"
        "\nCore parameter overrides (3 integers):\n"
        "  --build-params  segment_size,shortcuts_per_junction,max_skip_level\n"
        "  --static-params segment_size,shortcuts_per_junction,max_skip_level\n"
        "  --insert-params segment_size,shortcuts_per_junction,max_skip_level\n"
        "  --delete-params segment_size,shortcuts_per_junction,max_skip_level\n"
        "\nFull tuning overrides (12 comma‑separated values):\n"
        "  level_probability,local_interval,sub_interval,\n"
        "  t_j,soft_pct,hard_pct,flagged_ratio_limit,\n"
        "  min_seg_len_pct,max_suboptimal_segments,min_suboptimal_events_before_global,\n"
        "  emergency_hard_segment_ratio,stop_crash_local_sub\n",
        argv0);
}

int main(int argc, char **argv) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.nruns = 3;
    cfg.NOPS = 5000;
    cfg.do_build = cfg.do_static = cfg.do_insert = cfg.do_delete = true;
    cfg.csv_out = true;
    cfg.print_table = true;
    cfg.split_csv = true;
    snprintf(cfg.csv_path, sizeof cfg.csv_path, "bench_out.csv");
    cfg.pattern[0] = '\0';

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) {
            cfg.nruns = atoi(argv[++i]);
            if (cfg.nruns < 1) cfg.nruns = 1;
            if (cfg.nruns > MAX_RUNS) cfg.nruns = MAX_RUNS;
        }
        else if (strcmp(argv[i], "--nops") == 0 && i+1 < argc) {
            cfg.NOPS = atoi(argv[++i]);
            if (cfg.NOPS < 1) cfg.NOPS = 1;
            cfg.nops_set = true;
        }
        else if (strcmp(argv[i], "--out") == 0 && i+1 < argc) {
            snprintf(cfg.csv_path, sizeof cfg.csv_path, "%s", argv[++i]);
        }
        else if (strcmp(argv[i], "--no-csv") == 0) { cfg.csv_out = false; }
        else if (strcmp(argv[i], "--no-table") == 0) { cfg.print_table = false; }
        else if (strcmp(argv[i], "--split-csv") == 0) { cfg.split_csv = true; }
        else if (strcmp(argv[i], "--single-csv") == 0) { cfg.split_csv = false; }
        else if (strcmp(argv[i], "--pattern") == 0 && i+1 < argc) {
            snprintf(cfg.pattern, sizeof cfg.pattern, "%s", argv[++i]);
        }
        else if (strcmp(argv[i], "--n") == 0 && i+1 < argc) {
            fixed_n = atoi(argv[++i]);
            if (fixed_n < 1) fixed_n = 0;
        }
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) {
            user_seed = (unsigned int)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) {
            warmup_ops = atoi(argv[++i]);
            if (warmup_ops < 0) warmup_ops = 0;
       }
        else if (strcmp(argv[i], "--build-params") == 0 && i+1 < argc) {
            cfg.core_overrides[0] = true;
            if (!parse_core_params(argv[++i], &cfg.core_values[0])) {
                fprintf(stderr,"Invalid core params for BUILD\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--static-params") == 0 && i+1 < argc) {
            cfg.core_overrides[1] = true;
            if (!parse_core_params(argv[++i], &cfg.core_values[1])) {
                fprintf(stderr,"Invalid core params for STATIC\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--insert-params") == 0 && i+1 < argc) {
            cfg.core_overrides[2] = true;
            if (!parse_core_params(argv[++i], &cfg.core_values[2])) {
                fprintf(stderr,"Invalid core params for INSERT\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--delete-params") == 0 && i+1 < argc) {
            cfg.core_overrides[3] = true;
            if (!parse_core_params(argv[++i], &cfg.core_values[3])) {
                fprintf(stderr,"Invalid core params for DELETE\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--build-full-params") == 0 && i+1 < argc) {
            cfg.tuning_overrides[0] = true;
            if (!parse_tuning_params(argv[++i], &cfg.tuning_values[0])) {
                fprintf(stderr,"Invalid tuning params for BUILD (need 12 values)\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--static-full-params") == 0 && i+1 < argc) {
            cfg.tuning_overrides[1] = true;
            if (!parse_tuning_params(argv[++i], &cfg.tuning_values[1])) {
                fprintf(stderr,"Invalid tuning params for STATIC (need 12 values)\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--insert-full-params") == 0 && i+1 < argc) {
            cfg.tuning_overrides[2] = true;
            if (!parse_tuning_params(argv[++i], &cfg.tuning_values[2])) {
                fprintf(stderr,"Invalid tuning params for INSERT (need 12 values)\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--delete-full-params") == 0 && i+1 < argc) {
            cfg.tuning_overrides[3] = true;
            if (!parse_tuning_params(argv[++i], &cfg.tuning_values[3])) {
                fprintf(stderr,"Invalid tuning params for DELETE (need 12 values)\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--section") == 0 && i+1 < argc) {
            const char *s = argv[++i];
            if (strcasecmp(s, "ALL") != 0) {
                cfg.do_build = cfg.do_static = cfg.do_insert = cfg.do_delete = false;
                if (strcasecmp(s, "BUILD") == 0) cfg.do_build = true;
                else if (strcasecmp(s, "STATIC") == 0) cfg.do_static = true;
                else if (strcasecmp(s, "INSERT") == 0) cfg.do_insert = true;
                else if (strcasecmp(s, "DELETE") == 0) cfg.do_delete = true;
                else { fprintf(stderr, "Unknown section: %s\n", s); usage(argv[0]); return 1; }
            }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    printf("JLI Benchmark (with level_probability) – FIXED PATTERNS v2\n");
    printf("  runs=%d  nops=%d  csv=%s\n", cfg.nruns, cfg.NOPS,
           cfg.csv_out ? cfg.csv_path : "(disabled)");
    if (cfg.pattern[0]) printf("  pattern filter: %s\n", cfg.pattern);

    /* =================================================================
       VALIDATE PARAMETERS FOR EACH ACTIVE SECTION – EARLY ABORT
       ================================================================= */
    if (cfg.do_build) {
        CoreParams core = cfg.core_overrides[0] ? cfg.core_values[0] : DEFAULT_BUILD;
        const TuningParams *tp = cfg.tuning_overrides[0] ? &cfg.tuning_values[0] : &DEFAULT_TUNING;
        validate_jli_parameters(
            tp->level_probability,
            tp->emergency_hard_segment_ratio,
            tp->stop_crash_local_sub,
            tp->t_j,
            tp->soft_pct,
            tp->hard_pct,
            tp->flagged_ratio_limit,
            tp->min_seg_len_pct,
            tp->max_suboptimal_segments,
            tp->local_interval,
            tp->sub_interval,
            core.segment_size
        );
    }
    if (cfg.do_static) {
        CoreParams core = cfg.core_overrides[1] ? cfg.core_values[1] : DEFAULT_STATIC;
        const TuningParams *tp = cfg.tuning_overrides[1] ? &cfg.tuning_values[1] : &DEFAULT_TUNING;
        validate_jli_parameters(
            tp->level_probability,
            tp->emergency_hard_segment_ratio,
            tp->stop_crash_local_sub,
            tp->t_j,
            tp->soft_pct,
            tp->hard_pct,
            tp->flagged_ratio_limit,
            tp->min_seg_len_pct,
            tp->max_suboptimal_segments,
            tp->local_interval,
            tp->sub_interval,
            core.segment_size
        );
    }
    if (cfg.do_insert) {
        CoreParams core = cfg.core_overrides[2] ? cfg.core_values[2] : DEFAULT_INSERT;
        const TuningParams *tp = cfg.tuning_overrides[2] ? &cfg.tuning_values[2] : &DEFAULT_TUNING;
        validate_jli_parameters(
            tp->level_probability,
            tp->emergency_hard_segment_ratio,
            tp->stop_crash_local_sub,
            tp->t_j,
            tp->soft_pct,
            tp->hard_pct,
            tp->flagged_ratio_limit,
            tp->min_seg_len_pct,
            tp->max_suboptimal_segments,
            tp->local_interval,
            tp->sub_interval,
            core.segment_size
        );
    }
    if (cfg.do_delete) {
        CoreParams core = cfg.core_overrides[3] ? cfg.core_values[3] : DEFAULT_DELETE;
        const TuningParams *tp = cfg.tuning_overrides[3] ? &cfg.tuning_values[3] : &DEFAULT_TUNING;
        validate_jli_parameters(
            tp->level_probability,
            tp->emergency_hard_segment_ratio,
            tp->stop_crash_local_sub,
            tp->t_j,
            tp->soft_pct,
            tp->hard_pct,
            tp->flagged_ratio_limit,
            tp->min_seg_len_pct,
            tp->max_suboptimal_segments,
            tp->local_interval,
            tp->sub_interval,
            core.segment_size
        );
    }

    /* =================================================================
       RUN THE BENCHMARK
       ================================================================= */
    sweep_section(&cfg);

    /* =================================================================
       OUTPUT SUMMARY
       ================================================================= */
    if (cfg.csv_out) {
        if (cfg.split_csv) {
            printf("\nResults written to separate CSV files:\n");
            if (cfg.do_build)   printf("  - build_%s\n", cfg.csv_path);
            if (cfg.do_static)  printf("  - static_%s\n", cfg.csv_path);
            if (cfg.do_insert)  printf("  - insert_%s\n", cfg.csv_path);
            if (cfg.do_delete)  printf("  - delete_%s\n", cfg.csv_path);
        } else {
            printf("\nResults written to: %s\n", cfg.csv_path);
        }
    }

    return 0;
}