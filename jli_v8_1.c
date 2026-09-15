/*
 * ═══════════════════════════════════════════════════════════════════════
 * Junction-Linked List (JLI) — reference implementation, v8.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * High-level idea:
 *   A plain singly linked list is chopped into contiguous "segments" of
 *   roughly `segment_size` nodes. Each segment is described by a
 *   `Junction`, which caches:
 *     - the segment's start/end node,
 *     - a small sorted array of "shortcut" pointers into the segment
 *       (so we don't have to walk every node to get partway through it),
 *     - a single distinguished "junction node" near the segment's middle.
 *   Junctions are themselves kept in a sorted array and grouped into
 *   fixed-capacity `JunctionBlock`s; the blocks are indexed by a
 *   skip list so we can jump close to the right block in O(log n)
 *   expected time before falling back to a linear/binary walk inside
 *   the segment.
 *
 *   The overall aim (per the accompanying paper) is to trade a small
 *   amount of memory (the shortcuts + block skip list) for search
 *   performance closer to an array/skip-list than a raw linked list,
 *   while keeping O(1) splice-style insert/delete once the insertion
 *   point is known — without per-node indices that a plain array or
 *   flat skip list would require to be kept in sync.
 *
 *   Because insertions and deletions gradually make segments too big,
 *   too small, or unbalanced (junction node no longer near the middle),
 *   a maintenance system periodically re-shortcuts ("local rebuild"),
 *   re-partitions a region of segments ("suboptimal rebuild"), or
 *   rebuilds the entire structure from scratch ("global rebuild").
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <errno.h>

/* Branch-prediction hints for the compiler; purely a performance nicety,
 * they don't change behavior. LIKELY(x) says "x is usually true". */
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* Safety cap on how many nodes estimate_target_node() will walk/stage
 * while narrowing in on a target value inside a segment. Bounds a
 * fixed-size on-stack buffer and guarantees termination even if the
 * segment is pathologically large or the estimate is badly wrong. */
#define MAX_ESTIMATE_WINDOW 512

/* ═══════════════════════════════════════════════════════════════════════
   DATA STRUCTURES
   ═══════════════════════════════════════════════════════════════════════ */

/* A single element of the underlying sorted singly linked list.
 * `value` is the sort key; `payload` is an opaque user pointer that this
 * structure never interprets, only carries around. */
typedef struct Node {
    int32_t       value;
    void         *payload;
    struct Node  *next;
} Node;

/* Describes one segment of the list.
 *
 *   segment_start / segment_end : first and last Node in this segment.
 *   node                        : the distinguished "junction node",
 *                                 nominally near the segment midpoint;
 *                                 used as an extra probe point during
 *                                 search so we don't always have to
 *                                 walk from segment_start.
 *   junction_offset             : `node`'s position (0-based) within
 *                                 the segment, i.e. how many nodes
 *                                 precede it starting at segment_start.
 *   shortcuts[]                 : sorted array of pointers into the
 *                                 segment (always includes segment_start
 *                                 at index 0 and segment_end at the
 *                                 last index) used for a binary search
 *                                 within the segment before falling
 *                                 back to linear pointer-chasing.
 *   shortcuts_len               : number of live entries in shortcuts[]
 *                                 (<= K, the flexible array's capacity).
 *   prev_junction / next_junction: doubly-linked ring over all junctions,
 *                                 refreshed by junc_relink() whenever the
 *                                 junction array changes shape.
 *
 * Allocated with a C99 flexible array member so the shortcuts array is
 * sized to exactly K pointers in a single allocation (see junction_new).
 */
typedef struct Junction {
    Node            *node;
    Node            *segment_start;
    Node            *segment_end;
    struct Junction *prev_junction;
    struct Junction *next_junction;
    int32_t          shortcuts_len;
    int32_t          segment_len;
    int32_t          junction_offset;
    Node            *shortcuts[];       /* flexible array: K elements */
} Junction;

/* A candidate "anchor" node used while estimating where a target value
 * lives within a segment (see estimate_target_node()). `est_offset` is
 * this anchor's estimated position within the *live* segment, used to
 * bound how far we need to walk from it. */
typedef struct {
    Node    *node;
    int32_t  est_offset;
} AnchorRef;

/* ── block structure ──
 * A JunctionBlock groups a run of consecutive Junctions so the
 * block-level skip list (see below) has far fewer, coarser-grained
 * entries to index than the full junction array would. A block's
 * value range is never stored explicitly — it's always read live off
 * its first/last junction's segment_start/segment_end — so it can
 * never go stale relative to in-place edits to those junctions. */
typedef struct JunctionBlock {
    Junction **junctions;          /* pointer to array of junction pointers */
    int32_t    count;              /* live junctions (0 .. capacity) */
    int32_t    capacity;           /* max junctions that can be stored here */
    int32_t    index;              /* position in jli->blocks (updated on rebuild) */
    int32_t    skip_level;         /* random level for block skip‑list */
    struct JunctionBlock **next_skip; /* flexible array: [skip_level+1] pointers */
    /* range is computed live: start = junctions[0]->segment_start, end = junctions[count-1]->segment_end */
} JunctionBlock;

/* Top-level handle for the whole structure: the raw linked list plus
 * all of the JLI indexing/maintenance state layered on top of it. */
typedef struct {
    Node    *head;
    int32_t  length;

    /* Sorted (by segment order) array of every live Junction. */
    Junction **junctions;
    int32_t    num_junctions;
    int32_t    junctions_cap;

    int32_t    max_skip_level;

    /* ── block fields ──
     * blocks[] partitions junctions[] into fixed-capacity runs;
     * block_skip_heads[] is the entry point array for the skip list
     * over those blocks (index 0 = bottom/densest level). */
    JunctionBlock **blocks;
    int32_t         num_blocks;
    int32_t         blocks_cap;
    JunctionBlock **block_skip_heads;   /* [max_skip_level+1] */
    int32_t         block_m;            /* soft capacity per block */
    int32_t         block_hard_max;     /* hard limit (block_m + 2) */

    /* ── tunable structural/maintenance parameters (see validate_jli_parameters
     * and jli_create for their meaning and valid ranges) ── */
    int32_t segment_size;
    int32_t K;                         /* maximum shortcuts per junction */
    bool    enable_rebuild;
    int32_t maintenance_depth;
    int32_t local_interval;
    double  t_j;
    int32_t sub_interval;
    double  soft_pct;
    double  hard_pct;
    double  flagged_ratio_limit;
    double  min_seg_len_pct;
    double  max_suboptimal_segments;
    int32_t min_suboptimal_events_before_global;
    double  emergency_hard_segment_ratio;
    double  stop_crash_local_sub;

    double  level_probability;

    /* Cache of the junction (and its predecessor) touched by the most
     * recent search/mutation, so a follow-up delete on the same key
     * can skip re-locating it. Cleared whenever a rebuild invalidates it. */
    Junction *last_junction;
    Junction *last_prev_junction;

    /* ── maintenance scheduling counters ──
     * local_ops_counter / sub_ops_counter count mutations since the
     * last local / suboptimal maintenance pass respectively; when they
     * hit local_interval / sub_interval, maintenance_hook() fires the
     * corresponding pass (see maintenance_hook). */
    int32_t local_ops_counter;
    int32_t sub_ops_counter;
    int32_t sub_event_before_global;

    /* ── running statistics, exposed via jli_get_maintenance_stats() ── */
    int32_t search_count;
    int32_t insert_count;
    int32_t delete_count;
    int32_t local_scan_counter;
    int32_t suboptimal_scan_count;
    int32_t local_rebuild_events;
    int32_t suboptimal_rebuild_events;
    int32_t global_rebuild_events;
    int32_t dissolve_count;
    int64_t local_rebuild_node_touches;
    int64_t suboptimal_rebuild_node_touches;
    int64_t last_search_steps;
} JLI;

/* Two independent xorshift64 PRNG states:
 *   rng_state     — used by the standalone reference skip list (SL) at
 *                   the bottom of this file.
 *   jli_rng_state — used by JLI's own block skip-list level generator
 *                   (random_level()).
 * Kept separate so exercising one structure doesn't perturb the other's
 * random level sequence when both are benchmarked in the same process. */
static uint64_t jli_rng_state = 0xdeadbeefcafeULL;
static uint64_t rng_state = 0xdeadbeefcafeULL;

/* Re-seed both PRNGs from a single base seed, XORed with distinct
 * constants so the two streams diverge even for the same base_seed. */
void seed_structure_prngs(uint64_t base_seed) {
    rng_state = base_seed ^ 0xdeadbeefcafeULL;
    jli_rng_state = base_seed ^ 0xcafebeefdeadULL;
}

/* ═══════════════════════════════════════════════════════════════════════
   PARAMETER VALIDATION GATEKEEPER
   ═══════════════════════════════════════════════════════════════════════ */
/* Sanity-checks every tunable ratio/interval accepted by jli_create().
 * Most parameters are fractions and must lie in [0, 1]; a few pairs have
 * an ordering constraint (e.g. soft_pct must be < hard_pct). Any
 * violation is reported to stderr, and if at least one is invalid the
 * process aborts via exit(22) rather than silently running with a
 * nonsensical configuration. */
void validate_jli_parameters(
    double level_probability,
    double emergency_hard_segment_ratio,
    double stop_crash_local_sub,
    double t_j,
    double soft_pct,
    double hard_pct,
    double flagged_ratio_limit,
    double min_seg_len_pct,
    double max_suboptimal_segments,
    int32_t local_interval,
    int32_t sub_interval,
    int32_t segment_size
) {
    int invalid_param_found = 0;
    if (level_probability < 0.0 || level_probability > 1.0) {
        fprintf(stderr, "[PARAM ERROR] level_probability (%f) must be between 0 and 1\n", level_probability);
        invalid_param_found = 1;
    }
    if (emergency_hard_segment_ratio < 0.0 || emergency_hard_segment_ratio > 1.0) {
        fprintf(stderr, "[PARAM ERROR] emergency_hard_segment_ratio (%f) must be between 0 and 1\n", emergency_hard_segment_ratio);
        invalid_param_found = 1;
    }
    if (stop_crash_local_sub < 0.0 || stop_crash_local_sub > 1.0) {
        fprintf(stderr, "[PARAM ERROR] stop_crash_local_sub (%f) must be between 0 and 1\n", stop_crash_local_sub);
        invalid_param_found = 1;
    }
    if (t_j < 0.0 || t_j > 1.0) {
        fprintf(stderr, "[PARAM ERROR] t_j (%f) must be between 0 and 1\n", t_j);
        invalid_param_found = 1;
    }
    if (soft_pct < 0.0 || soft_pct > 1.0) {
        fprintf(stderr, "[PARAM ERROR] soft_pct (%f) must be between 0 and 1\n", soft_pct);
        invalid_param_found = 1;
    }
    if (hard_pct < 0.0 || hard_pct > 1.0) {
        fprintf(stderr, "[PARAM ERROR] hard_pct (%f) must be between 0 and 1\n", hard_pct);
        invalid_param_found = 1;
    }
    if (flagged_ratio_limit < 0.0 || flagged_ratio_limit > 1.0) {
        fprintf(stderr, "[PARAM ERROR] flagged_ratio_limit (%f) must be between 0 and 1\n", flagged_ratio_limit);
        invalid_param_found = 1;
    }
    if (min_seg_len_pct < 0.0 || min_seg_len_pct > 1.0) {
        fprintf(stderr, "[PARAM ERROR] min_seg_len_pct (%f) must be between 0 and 1\n", min_seg_len_pct);
        invalid_param_found = 1;
    }
    if (max_suboptimal_segments < 0.0 || max_suboptimal_segments > 1.0) {
        fprintf(stderr, "[PARAM ERROR] max_suboptimal_segments (%f) must be between 0 and 1\n", max_suboptimal_segments);
        invalid_param_found = 1;
    }
    if (soft_pct >= hard_pct) {
        fprintf(stderr, "[PARAM ERROR] Inverted threshold: soft_pct (%f) cannot be >= hard_pct (%f)\n", soft_pct, hard_pct);
        invalid_param_found = 1;
    }
    if (local_interval >= sub_interval) {
        fprintf(stderr, "[PARAM ERROR] Interval conflict: local_interval (%d) cannot be >= sub_interval (%d)\n", local_interval, sub_interval);
        invalid_param_found = 1;
    }
    if ((double)segment_size * min_seg_len_pct < 1.0) {
        fprintf(stderr, "[PARAM ERROR] Segment size (%d) * min_seg_len_pct (%f) is less than 1 element\n", segment_size, min_seg_len_pct);
        invalid_param_found = 1;
    }
    if (invalid_param_found) {
        fprintf(stderr, "Aborting process due to structurally invalid parameter matrix. Exit status 22.\n");
        exit(22);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   NODE / JUNCTION HELPERS
   ═══════════════════════════════════════════════════════════════════════ */

/* Allocate a single list node. Caller owns `payload`'s lifetime; JLI
 * never frees it (see junction_free / jli_destroy, which only free the
 * Node/Junction structs themselves). */
static Node *node_new(int32_t value, void *payload) {
    Node *n = malloc(sizeof *n);
    n->value = value; n->payload = payload; n->next = NULL;
    return n;
}

/* Allocate a new Junction with room for K shortcut pointers, using a
 * single calloc so the flexible array member is contiguous with the
 * struct. Initializes it as a degenerate one-node segment: `node`,
 * `segment_start`, and `segment_end` all point at `node` until the
 * caller fills in the real segment via build_shortcuts() or by hand. */
static Junction *junction_new(Node *node, int32_t K) {
    Junction *j = calloc(1, sizeof(Junction) + K * sizeof(Node*));
    if (!j) return NULL;
    j->node = j->segment_start = j->segment_end = node;
    j->shortcuts_len = 0;
    return j;
}

/* Frees a Junction struct itself. Does NOT free the underlying list
 * nodes it points into (segment_start/end, shortcuts, node) — those
 * are owned by the main linked list, not by the Junction. */
static void junction_free(Junction *j) {
    if (j) {
        free(j);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   SHORTCUT PLANNING (K‑driven, no hard limit)  – unchanged
   ═══════════════════════════════════════════════════════════════════════
   The functions below decide WHERE, as fractional offsets within a
   segment of length seg_len, shortcut pointers "ideally" should sit so
   that binary-searching them narrows the search as evenly as possible.
   They operate purely on offsets/indices (not on live Node pointers),
   which is what makes them reusable both when a segment is first built
   (offsets map directly onto real node positions) and later, after
   inserts/deletes have shifted things, when they're used as an
   *estimate* of where a shortcut should logically be relative to the
   segment's current midpoint (see map_planned_offset_to_live). */

/* Plans up to `needed` = (K-2) *interior* shortcut offsets (i.e.
 * excluding the two endpoints, which are always segment_start/end and
 * handled separately) using a breadth-first bisection: start with the
 * two halves [0, mid] and [mid, end], repeatedly bisect whichever
 * interval was queued first, and record its midpoint as a shortcut
 * offset (skipping it if it collides with 0, mid_idx, or the end).
 * This spreads shortcuts roughly evenly across the segment in the
 * order most useful first (coarsest gaps get subdivided earliest).
 * Writes into `offsets` (capacity `cap`) and returns how many were
 * written. `ql`/`qr` form a small fixed-size work queue (256 entries
 * is far more than any realistic K would need). */
static int plan_shortcut_offsets(int32_t seg_len, int32_t K,
                                  int32_t *offsets, int32_t cap) {
    int needed = (K > 2 ? K - 2 : 0);
    if (seg_len <= 1 || needed <= 0) return 0;
    int32_t mid_idx = (seg_len - 1) / 2;
    int32_t ql[256], qr[256]; int qhead = 0, qtail = 0;
    ql[qtail] = 0;       qr[qtail] = mid_idx;     qtail++;
    ql[qtail] = mid_idx; qr[qtail] = seg_len - 1; qtail++;
    int n_out = 0;
    while (needed > 0 && qhead < qtail && n_out < cap) {
        int32_t l = ql[qhead], r = qr[qhead]; qhead++;
        if (r - l <= 1) continue;
        int32_t mid = (l + r) / 2;
        if (mid != 0 && mid != mid_idx && mid != seg_len - 1) {
            offsets[n_out++] = mid; needed--;
        }
        if (needed > 0 && qtail + 2 <= 256) {
            ql[qtail] = l;   qr[qtail] = mid; qtail++;
            ql[qtail] = mid; qr[qtail] = r;   qtail++;
        }
    }
    return n_out;
}

/* Simple insertion sort of an int32 offset array in place, ascending.
 * n is always small here (bounded by K), so O(n^2) is fine. */
static void sort_offsets_asc(int32_t *offsets, int32_t n) {
    for (int32_t i = 1; i < n; i++) {
        int32_t v = offsets[i], j = i - 1;
        while (j >= 0 && offsets[j] > v) { offsets[j+1] = offsets[j]; j--; }
        offsets[j+1] = v;
    }
}

/* Clamp v into [lo, hi]. Used throughout to keep computed offsets
 * within a segment's valid index range despite rounding/estimation. */
static int32_t clamp_offset(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Builds the full sorted, de-duplicated list of "ideal" shortcut offsets
 * for a segment of length seg_len: always 0 and `end`, the midpoint
 * (unless it coincides with an endpoint), plus up to K-2 interior
 * offsets from plan_shortcut_offsets(). Writes into `offsets` (capacity
 * `cap`) and returns the number of unique offsets written. Used by
 * planned_gap_stats() to characterize the *ideal* shortcut spacing,
 * independent of where shortcuts actually sit after mutations. */
static int32_t collect_planned_anchor_offsets(int32_t seg_len, int32_t K,
                                              int32_t *offsets, int32_t cap) {
    if (cap <= 0 || seg_len <= 0) return 0;
    int32_t end = seg_len - 1, mid = end / 2, n = 0;
    offsets[n++] = 0;
    if (n < cap) offsets[n++] = end;
    if (mid != 0 && mid != end && n < cap) offsets[n++] = mid;
    int32_t *int_offs = malloc(K * sizeof(int32_t));
    if (!int_offs) return n;
    int32_t n_int = plan_shortcut_offsets(seg_len, K, int_offs, K);
    sort_offsets_asc(int_offs, n_int);
    for (int32_t i = 0; i < n_int && n < cap; i++) {
        int32_t off = int_offs[i];
        if (off == 0 || off == mid || off == end) continue;
        offsets[n++] = off;
    }
    free(int_offs);
    sort_offsets_asc(offsets, n);
    int32_t uniq = 0;
    for (int32_t i = 0; i < n; i++)
        if (uniq == 0 || offsets[i] != offsets[uniq-1]) offsets[uniq++] = offsets[i];
    return uniq;
}

/* Computes the mean and standard deviation of the gaps between
 * consecutive *ideal* shortcut offsets in a segment of length seg_len.
 * Used by estimate_step_slop() as a proxy for "how far apart are
 * shortcuts typically spaced", to size the search window when walking
 * from an estimated anchor toward a target node. */
static void planned_gap_stats(int32_t seg_len, int32_t K,
                              double *avg_out, double *std_out) {
    int32_t *offs = malloc((K + 3) * sizeof(int32_t));
    if (!offs) { *avg_out = 0.0; *std_out = 0.0; return; }
    int32_t n = collect_planned_anchor_offsets(seg_len, K, offs, K+3);
    if (n < 2) { *avg_out = 0.0; *std_out = 0.0; free(offs); return; }
    double sum = 0.0; int32_t ng = 0;
    for (int32_t i = 1; i < n; i++) { sum += (double)(offs[i]-offs[i-1]); ng++; }
    double avg = ng > 0 ? sum / (double)ng : 0.0;
    double var = 0.0;
    for (int32_t i = 1; i < n; i++) {
        double d = (double)(offs[i]-offs[i-1]) - avg; var += d*d;
    }
    *avg_out = avg;
    *std_out = ng > 0 ? sqrt(var / (double)ng) : 0.0;
    free(offs);
}

/* Returns the *ideal* offset for a single shortcut slot index `slot`
 * within j->shortcuts[] (0 = segment_start, K-1 = segment_end, the
 * slots in between are the interior shortcuts from
 * plan_shortcut_offsets(), sorted ascending). Used when only one
 * slot's planned position is needed, rather than the whole array
 * (see collect_live_anchors's fallback path). */
static int32_t shortcut_slot_planned_offset(int32_t seg_len, int32_t K, int32_t slot) {
    if (seg_len <= 1 || slot <= 0) return 0;
    int32_t end = seg_len - 1;
    if (slot >= K - 1) return end;
    int32_t *int_offs = malloc(K * sizeof(int32_t));
    if (!int_offs) return 0;
    int32_t n_int = plan_shortcut_offsets(seg_len, K, int_offs, K);
    sort_offsets_asc(int_offs, n_int);
    int32_t result = 0;
    if (slot - 1 < n_int) result = int_offs[slot-1];
    free(int_offs);
    return result;
}

/* Fills `slot_offsets[0..nslots-1]` with the ideal offset for every
 * shortcut slot in one pass (slot 0 = 0, slot nslots-1 = end, interior
 * slots from plan_shortcut_offsets()). Equivalent to calling
 * shortcut_slot_planned_offset() for every slot but far cheaper, since
 * it only computes plan_shortcut_offsets() once instead of once per
 * slot. Used by estimate_target_node() to precompute offsets for every
 * live shortcut in a segment before scoring anchors. */
static void fill_shortcut_slot_offsets(int32_t seg_len, int32_t K,
                                       int32_t *slot_offsets, int32_t nslots) {
    if (!slot_offsets || nslots <= 0) return;
    for (int32_t i = 0; i < nslots; i++) slot_offsets[i] = 0;
    if (seg_len <= 1) return;
    int32_t end = seg_len - 1;
    int32_t *int_offs = malloc(K * sizeof(int32_t));
    if (!int_offs) return;
    int32_t n_int = plan_shortcut_offsets(seg_len, K, int_offs, K);
    sort_offsets_asc(int_offs, n_int);
    slot_offsets[0] = 0;
    int32_t slot = 1;
    for (int32_t i = 0; i < n_int && slot < nslots-1; i++) slot_offsets[slot++] = int_offs[i];
    free(int_offs);
    while (slot < nslots-1) slot_offsets[slot++] = 0;
    if (nslots > 1) slot_offsets[nslots-1] = end;
}

/* Maps a "planned" (ideal, as-built) offset onto an estimated "live"
 * offset, given that the junction node currently sits at
 * live_junction_offset instead of its original planned midpoint. This
 * matters because inserts/deletes shift where things actually are
 * without immediately rebuilding shortcuts; rather than assuming the
 * old planned offsets are still accurate, this linearly rescales each
 * half of the segment (before/after the midpoint) independently, using
 * the junction node's live position as the new "center" anchor. Used
 * only as an estimate — the caller (estimate_target_node) still walks
 * pointers to confirm/correct it, bounded by estimate_step_slop(). */
static int32_t map_planned_offset_to_live(int32_t seg_len, int32_t live_junction_offset,
                                          int32_t planned_offset) {
    if (seg_len <= 1) return 0;
    int32_t end = seg_len-1, mid = end/2;
    int32_t live = clamp_offset(live_junction_offset, 0, end);
    int32_t target = clamp_offset(planned_offset, 0, end);
    if (target <= mid) {
        if (mid <= 0) return 0;
        return clamp_offset((int32_t)lround((double)target*(double)live/(double)mid), 0, end);
    }
    if (end <= mid) return end;
    return clamp_offset(live + (int32_t)lround((double)(target-mid)*(double)(end-live)/(double)(end-mid)), 0, end);
}

/* Estimates how much extra walking slack ("slop") to allow beyond the
 * naive step_goal when walking from a lower-bound anchor toward
 * target_offset, to absorb the imprecision of
 * map_planned_offset_to_live()'s linear rescaling. Scales the
 * (avg_gap + std_gap) shortcut spacing by how stretched the relevant
 * half of the segment is relative to its planned proportions, and
 * always allows at least 1 extra step. */
static int32_t estimate_step_slop(int32_t seg_len, int32_t ref_junction_offset,
                                  int32_t target_offset, double avg_gap, double std_gap) {
    if (seg_len <= 1) return 0;
    int32_t end = seg_len-1, mid = end/2;
    int32_t ref = clamp_offset(ref_junction_offset, 0, end);
    int32_t target = clamp_offset(target_offset, 0, end);
    double scale = 1.0;
    if (target <= mid) { if (mid > 0) scale = (double)ref/(double)mid; }
    else if (end > mid) scale = (double)(end-ref)/(double)(end-mid);
    if (scale <= 0.0) scale = 1.0;
    double slop = ((avg_gap + std_gap) * scale) * 0.5;
    if (slop < 1.0) slop = 1.0;
    return (int32_t)ceil(slop);
}

/* Appends `node` (with its estimated offset) to the anchors array if
 * it isn't already present and there's room; no-op for a NULL node.
 * Returns the (possibly unchanged) count `n`. Used to build a
 * deduplicated candidate anchor list in collect_live_anchors(). */
static int32_t anchor_push_unique(AnchorRef *anchors, int32_t n, int32_t cap,
                                  Node *node, int32_t est_offset) {
    if (!node || n >= cap) return n;
    for (int32_t i = 0; i < n; i++) if (anchors[i].node == node) return n;
    anchors[n].node = node; anchors[n].est_offset = est_offset;
    return n + 1;
}

/* Insertion sort of anchors by node value ascending (ties broken by
 * estimated offset). n is always small (bounded by K+few), so O(n^2)
 * is fine. */
static void sort_anchors_by_value(AnchorRef *anchors, int32_t n) {
    for (int32_t i = 1; i < n; i++) {
        AnchorRef cur = anchors[i]; int32_t j = i-1;
        while (j >= 0) {
            bool move = anchors[j].node->value > cur.node->value;
            if (!move && anchors[j].node->value == cur.node->value)
                move = anchors[j].est_offset > cur.est_offset;
            if (!move) break;
            anchors[j+1] = anchors[j]; j--;
        }
        anchors[j+1] = cur;
    }
}

/* Gathers a deduplicated, value-sorted list of candidate "anchor"
 * nodes within junction j's segment: segment_start (offset 0), the
 * junction node itself (unless it's the node being skipped, e.g.
 * because it was just deleted), every live shortcut (with its
 * estimated live offset via map_planned_offset_to_live, using
 * slot_offsets if precomputed or falling back to
 * shortcut_slot_planned_offset per-shortcut otherwise), and
 * segment_end (offset = end). `skip` lets the caller exclude a
 * specific node (typically one mid-deletion) from being used as an
 * anchor. Returns the number of anchors written into `anchors`
 * (capacity `cap`), sorted ascending by node value. */
static int32_t collect_live_anchors(const Junction *j, int32_t K, Node *skip,
                                    const int32_t *slot_offsets,
                                    int32_t ref_junction_offset,
                                    AnchorRef *anchors, int32_t cap) {
    if (!j || cap <= 0 || j->segment_len <= 0) return 0;
    int32_t end = j->segment_len-1;
    int32_t ref = clamp_offset(ref_junction_offset, 0, end);
    int32_t n = 0;
    n = anchor_push_unique(anchors, n, cap, j->segment_start, 0);
    n = anchor_push_unique(anchors, n, cap, (j->node != skip) ? j->node : NULL, ref);
    for (int32_t i = 0; i < j->shortcuts_len && n < cap; i++) {
        Node *cand = j->shortcuts[i];
        if (!cand || cand == skip) continue;
        int32_t planned = slot_offsets ? slot_offsets[i]
                                       : shortcut_slot_planned_offset(j->segment_len, K, i);
        int32_t est = (i == 0) ? 0
                    : (i == j->shortcuts_len-1) ? end
                    : map_planned_offset_to_live(j->segment_len, ref, planned);
        n = anchor_push_unique(anchors, n, cap, cand, est);
    }
    n = anchor_push_unique(anchors, n, cap, j->segment_end, end);
    sort_anchors_by_value(anchors, n);
    return n;
}

/* Core estimator: given a junction j and a target key (target_value,
 * with target_offset as its estimated live position within the
 * segment), returns the Node in the segment closest to where target
 * should be inserted/found — i.e. the node the caller should start
 * walking from, rather than always starting at segment_start.
 *
 * Algorithm:
 *   1. Compute planned_gap_stats() to characterize typical shortcut
 *      spacing (used to size the walk-slop allowance).
 *   2. Precompute live offsets for every shortcut slot
 *      (fill_shortcut_slot_offsets), bounded by MAX_ESTIMATE_WINDOW to
 *      avoid unbounded work on segments with an unusually large K.
 *   3. Collect all candidate anchors (collect_live_anchors) and find
 *      the tightest bracket [lower, upper] around target_value by
 *      value (not by walking the list — anchors are already sorted).
 *   4. Walk forward from `lower` toward `upper`, staging visited nodes
 *      into a fixed-size buffer, up to (step_goal + slop) steps or
 *      MAX_ESTIMATE_WINDOW nodes, whichever comes first. If we
 *      "overshoot" target_value before reaching `upper`, that's a
 *      signal the linear offset estimate was off; in that case (or if
 *      we do reach `upper`) we don't trust the exact stepped-to
 *      position and instead return the middle of what we've staged,
 *      which is still a much better starting point than segment_start.
 *   5. Otherwise (no overshoot, walk_limit reached cleanly) return the
 *      node at the estimated step_goal position.
 *
 * This function never fails outright — on allocation failure it falls
 * back to segment_start/segment_end so callers can always proceed with
 * a normal linear walk. */
static Node *estimate_target_node(Junction *j, int32_t K, Node *skip,
                                  int32_t ref_junction_offset,
                                  int32_t target_value, int32_t target_offset) {
    if (!j) return NULL;
    if (j->segment_len <= 0)
        return j->segment_start ? j->segment_start : j->segment_end;
    int32_t end = j->segment_len-1;
    int32_t target = clamp_offset(target_offset, 0, end);
    double avg_gap = 0.0, std_gap = 0.0;
    planned_gap_stats(j->segment_len, K, &avg_gap, &std_gap);

    int32_t *slot_offsets = NULL;
    if (j->shortcuts_len > 0 && j->shortcuts_len <= MAX_ESTIMATE_WINDOW) {
        slot_offsets = malloc(j->shortcuts_len * sizeof(int32_t));
        if (slot_offsets) {
            fill_shortcut_slot_offsets(j->segment_len, K, slot_offsets, j->shortcuts_len);
        }
    }

    int32_t max_anchors = K + 4;
    AnchorRef *anchors = malloc(max_anchors * sizeof(AnchorRef));
    if (!anchors) {
        free(slot_offsets);
        return j->segment_start ? j->segment_start : j->segment_end;
    }

    int32_t n_anchors = collect_live_anchors(j, K, skip, slot_offsets,
                                             ref_junction_offset, anchors, max_anchors);
    free(slot_offsets);

    if (n_anchors <= 0) {
        free(anchors);
        return j->segment_start ? j->segment_start : j->segment_end;
    }

    AnchorRef lower = anchors[0], upper = anchors[n_anchors-1];
    bool has_lower = false, has_upper = false;
    for (int32_t i = 0; i < n_anchors; i++) {
        if (anchors[i].node->value < target_value) { lower = anchors[i]; has_lower = true; }
        else if (!has_upper && anchors[i].node->value > target_value) { upper = anchors[i]; has_upper = true; }
    }
    if (!has_lower) lower = anchors[0];
    if (!has_upper) upper = anchors[n_anchors-1];
    int32_t step_goal = target - lower.est_offset;
    if (step_goal < 0) step_goal = 0;
    int32_t walk_limit = step_goal
                       + estimate_step_slop(j->segment_len, ref_junction_offset,
                                            target, avg_gap, std_gap);
    Node *stored[MAX_ESTIMATE_WINDOW];
    int32_t count = 0, step = 0;
    bool overshoot = false, reached_upper = false;
    Node *cur = lower.node;
    while (cur && count < MAX_ESTIMATE_WINDOW) {
        stored[count++] = cur;
        if (cur->value > target_value) overshoot = true;
        if (cur == upper.node) { reached_upper = true; break; }
        if (step >= walk_limit) break;
        cur = cur->next; step++;
    }
    if (overshoot && !reached_upper && upper.node) {
        cur = stored[count-1];
        while (cur && cur != upper.node && count < MAX_ESTIMATE_WINDOW) {
            cur = cur->next;
            if (!cur) break;
            stored[count++] = cur;
            if (cur == upper.node) { reached_upper = true; break; }
        }
    }
    free(anchors);
    if (count <= 0) return lower.node;
    if (overshoot || reached_upper) return stored[(count-1)/2];
    if (step_goal >= count) step_goal = count-1;
    return stored[step_goal];
}

/* ═══════════════════════════════════════════════════════════════════════
   BUILD SHORTCUTS  (unchanged)
   ═══════════════════════════════════════════════════════════════════════ */
/* (Re)builds junction j's shortcut array and junction node from scratch
 * by walking the segment once, given its (possibly just-changed)
 * length seg_len. This is the "ground truth" rebuild used whenever a
 * segment's shortcuts can no longer be trusted to be well-positioned
 * (e.g. after local_rebuild, merges, or an off-center junction node).
 *
 * Steps:
 *   1. Plan interior offsets via plan_shortcut_offsets() and sort them.
 *   2. Determine total shortcut count: 2 endpoints + interior offsets,
 *      clamped to exactly K (padding with segment_start if short).
 *   3. Single walk from segment_start to segment_end, recording:
 *        - the node at the planned junction offset (new `node`),
 *        - the node at each planned interior offset (into shortcuts[]).
 *   4. Fill any unused interior slots with segment_start (defensive —
 *      only happens if n_int came up short) and always set the last
 *      slot to segment_end.
 * Mutates j->shortcuts, j->shortcuts_len, j->node, j->junction_offset,
 * and j->segment_len in place. */
static void build_shortcuts(Junction *j, int32_t seg_len, int32_t K) {
    j->shortcuts_len = 0;
    int32_t junc_off = (seg_len > 1) ? (seg_len-1)/2 : 0;
    int32_t *int_offs = malloc(K * sizeof(int32_t));
    if (!int_offs) return;
    int32_t n_int = plan_shortcut_offsets(seg_len, K, int_offs, K);
    for (int a = 0; a < n_int-1; a++)
        for (int b = a+1; b < n_int; b++)
            if (int_offs[a] > int_offs[b]) { int32_t t = int_offs[a]; int_offs[a] = int_offs[b]; int_offs[b] = t; }
    int32_t total = 2 + n_int;
    if (total < K) total = K;
    if (total > K) total = K;
    j->shortcuts_len = total;
    j->shortcuts[0] = j->segment_start;
    int int_slot = 1;
    Node *cur = j->segment_start;
    int32_t idx = 0, sc_next = 0;
    Node *jn = j->segment_start;
    while (cur) {
        if (idx == junc_off) jn = cur;
        if (sc_next < n_int && idx == int_offs[sc_next] && int_slot < total-1) {
            j->shortcuts[int_slot++] = cur; sc_next++;
        }
        if (cur == j->segment_end) break;
        cur = cur->next; idx++;
    }
    while (int_slot < total-1) j->shortcuts[int_slot++] = j->segment_start;
    j->shortcuts[total-1] = j->segment_end;
    j->node = jn;
    j->junction_offset = junc_off;
    j->segment_len = seg_len;
    free(int_offs);
}

/* ═══════════════════════════════════════════════════════════════════════
   JUNCTION ARRAY & SKIP LIST (unchanged except RNG)
   ═══════════════════════════════════════════════════════════════════════ */
/* Appends junction j to jli->junctions[], growing the array (doubling,
 * starting at 16) if needed. On allocation failure, frees j and
 * returns false so the caller can unwind; on success, returns true. */
static bool junc_arr_push(JLI *jli, Junction *j) {
    if (jli->num_junctions >= jli->junctions_cap) {
        int32_t newcap = jli->junctions_cap ? jli->junctions_cap * 2 : 16;
        Junction **tmp = realloc(jli->junctions, newcap * sizeof *tmp);
        if (!tmp) {
            junction_free(j);
            return false;
        }
        jli->junctions = tmp;
        jli->junctions_cap = newcap;
    }
    jli->junctions[jli->num_junctions++] = j;
    return true;
}

/* Rebuilds the prev_junction/next_junction doubly-linked ring over all
 * junctions to match the current jli->junctions[] array order. Must be
 * called after any operation that changes the junction array's shape
 * (insert/remove/reorder) — mutation callers that touch a single
 * junction in place without resizing may skip it, but anything that
 * shifts the array (dissolves, merges, region rebuilds) calls this. */
static void junc_relink(JLI *jli) {
    int32_t n = jli->num_junctions;
    for (int32_t i = 0; i < n; i++) {
        jli->junctions[i]->prev_junction = jli->junctions[(i-1+n)%n];
        jli->junctions[i]->next_junction = jli->junctions[(i+1)%n];
    }
}

/* Generates a random skip-list level in [0, max_level-1] using
 * jli_rng_state (xorshift64), with each level having independent
 * probability `p` of "leveling up" from the one below — i.e. a
 * geometric distribution truncated at max_level-1. Degenerate cases:
 * max_level<=1 or p<=0 always returns 0 (no extra levels); p>=1.0
 * always returns the max possible level. */
static uint8_t random_level(int max_level, double p) {
    uint8_t lvl = 0;
    if (max_level <= 1 || p <= 0.0) return 0;
    if (p >= 1.0) return (uint8_t)(max_level - 1);
    uint8_t max_lvl = (uint8_t)(max_level - 1);
    while (lvl < max_lvl) {
        jli_rng_state ^= jli_rng_state << 13;
        jli_rng_state ^= jli_rng_state >> 7;
        jli_rng_state ^= jli_rng_state << 17;
        double u = (double)jli_rng_state / 18446744073709551616.0;
        if (u < p) lvl++;
        else break;
    }
    return lvl;
}

/* ═══════════════════════════════════════════════════════════════════════
   BLOCK STRUCTURE & SKIP‑LIST (now with live range & index)
   ═══════════════════════════════════════════════════════════════════════ */

/* Allocates an empty JunctionBlock with room for `capacity` junction
 * pointers. `index` starts at -1 (unset — the caller is responsible
 * for assigning its real position once it's inserted into
 * jli->blocks[]). On partial allocation failure, cleans up and
 * returns NULL. */
static JunctionBlock *block_new(int32_t capacity) {
    JunctionBlock *b = calloc(1, sizeof(JunctionBlock));
    if (!b) return NULL;
    b->junctions = malloc(capacity * sizeof(Junction*));
    if (!b->junctions) {
        free(b);
        return NULL;
    }
    b->capacity = capacity;
    b->count = 0;
    b->index = -1;
    b->skip_level = 0;
    b->next_skip = NULL;
    return b;
}

/* Frees a JunctionBlock and its owned arrays (the junction-pointer
 * array and the skip-list forward-pointer array). Does NOT free the
 * Junctions it points to — those are owned by jli->junctions[]. */
static void block_free(JunctionBlock *b) {
    if (b) {
        free(b->junctions);
        free(b->next_skip);
        free(b);
    }
}

/* build blocks from current junctions array (greedy, up to hard_max) */
/* Discards all existing blocks and repartitions the current
 * jli->junctions[] array into fresh blocks, greedily filling each one
 * up to block_hard_max before starting the next. Grows jli->blocks[]
 * if needed. This is the full "from scratch" block build used after a
 * global rebuild or whenever build_junctions() runs; incremental
 * mutations instead use rebuild_blocks_for_region() to avoid redoing
 * this work for the whole structure. */
static void build_blocks(JLI *jli) {
    for (int32_t i = 0; i < jli->num_blocks; i++) block_free(jli->blocks[i]);
    jli->num_blocks = 0;
    if (jli->num_junctions == 0) return;

    int32_t hard_max = jli->block_hard_max;
    int32_t needed = (jli->num_junctions + hard_max - 1) / hard_max;
    if (needed > jli->blocks_cap) {
        int32_t newcap = needed * 2;
        JunctionBlock **tmp = realloc(jli->blocks, newcap * sizeof(JunctionBlock*));
        if (!tmp) return;
        jli->blocks = tmp;
        jli->blocks_cap = newcap;
    }

    int32_t idx = 0;
    while (idx < jli->num_junctions) {
        JunctionBlock *b = block_new(hard_max);
        if (!b) break;
        b->count = 0;
        while (idx < jli->num_junctions && b->count < hard_max) {
            b->junctions[b->count++] = jli->junctions[idx];
            idx++;
        }
        b->index = jli->num_blocks;
        jli->blocks[jli->num_blocks++] = b;
    }
}

/* skip‑list over blocks (uses live range via junctions[0] and junctions[count-1]) */
/* Rebuilds the entire block-level skip list from scratch: assigns each
 * block a fresh random level (random_level()), reallocates its
 * next_skip[] forward-pointer array to match, then links every block
 * into every level <= its own level by walking blocks in array order
 * (which is value-sorted) and threading a `last[level]` pointer per
 * level. block_skip_heads[level] becomes the first block present at
 * that level.
 *
 * If the `last` bookkeeping array itself fails to allocate, this falls
 * back to forcing every block to level 0 (a plain linked scan with no
 * skip levels) rather than leaving the skip list half-built. */
static void build_block_skip_list(JLI *jli) {
    int max_lvl = jli->max_skip_level;
    for (int32_t i = 0; i <= max_lvl; i++) jli->block_skip_heads[i] = NULL;
    int32_t n = jli->num_blocks;
    if (n == 0) return;

    for (int32_t i = 0; i < n; i++) {
        JunctionBlock *b = jli->blocks[i];
        uint8_t new_level = random_level(max_lvl, jli->level_probability);
        JunctionBlock **new_skip = calloc(new_level + 1, sizeof(JunctionBlock*));
        if (!new_skip) {
            new_skip = calloc(1, sizeof(JunctionBlock*));
            new_level = 0;
        }
        free(b->next_skip);
        b->next_skip = new_skip;
        b->skip_level = new_level;
    }

    JunctionBlock **last = calloc(max_lvl + 1, sizeof(JunctionBlock*));
    if (!last) {
        for (int32_t i = 0; i < n; i++) {
            free(jli->blocks[i]->next_skip);
            jli->blocks[i]->next_skip = calloc(1, sizeof(JunctionBlock*));
            jli->blocks[i]->skip_level = 0;
        }
        jli->max_skip_level = 0;
        max_lvl = 0;
        last = calloc(1, sizeof(JunctionBlock*));
        if (!last) {
            fprintf(stderr, "FATAL: build_block_skip_list out of memory\n");
            abort();
        }
        for (int32_t i = 0; i <= jli->max_skip_level; i++) jli->block_skip_heads[i] = NULL;
    }

    for (int32_t i = 0; i < n; i++) {
        JunctionBlock *b = jli->blocks[i];
        for (int32_t l = 0; l <= b->skip_level; l++) {
            if (last[l] == NULL)
                jli->block_skip_heads[l] = b;
            else
                last[l]->next_skip[l] = b;
            last[l] = b;
        }
    }
    free(last);
}

/* find the block whose range might contain target (skip‑list search) – uses live range
 *
 * NOTE: block_skip_heads[lvl] is NOT a sentinel -- it is the first real block that
 * happens to have skip_level >= lvl, and levels are assigned independently of key
 * order. That block's own key can already exceed target. A classic skip list avoids
 * this by prefixing the list with a sentinel node (key = -infinity) that exists at
 * every level, so descending from any level's head can never itself overshoot. We
 * don't have a sentinel here, so we must explicitly treat any head/next candidate
 * whose start value exceeds target as inadmissible, and — critically — falling back
 * to a lower level must re-consider that level's own head as a candidate, not just
 * keep walking forward from wherever the previous (higher) level's excursion left
 * off, since that excursion may have overshot past a valid block reachable only from
 * the lower level's own head. */
/* Walks the block skip list top-down to find the last block whose
 * range could still contain `target` — i.e. the entry point for a
 * within-block search. Increments *steps (if non-NULL) once per block
 * visited/advanced-to, for instrumentation. See the detailed comment
 * above for why this needs extra care in the absence of a sentinel
 * head node. Returns NULL only if jli has no blocks at all. */
static JunctionBlock *block_skip_search(JLI *jli, int32_t target, int64_t *steps) {
    if (jli->num_blocks == 0) return NULL;
    int max_lvl = jli->max_skip_level;
    JunctionBlock *cur = NULL; /* best candidate found so far; NULL = none admissible yet */
    for (int32_t lvl = max_lvl; lvl >= 0; lvl--) {
        JunctionBlock *head = jli->block_skip_heads[lvl];
        if (!cur) {
            /* No admissible candidate yet: consider this level's own head, but only
             * if it doesn't itself overshoot target. */
            if (head && head->junctions[0]->segment_start->value <= target) {
                cur = head;
                if (steps) (*steps)++;
            } else {
                continue; /* this level's head overshoots too; try a lower level */
            }
        } else {
            /* We have a candidate from a higher level. It's possible the lower
             * level's own head is further along (but still <= target) than cur,
             * if cur was reached via a level that skipped past nodes visible only
             * at this lower level from the true start. Prefer whichever is later. */
            if (head && head->junctions[0]->segment_start->value <= target &&
                head->junctions[0]->segment_start->value > cur->junctions[0]->segment_start->value) {
                cur = head;
                if (steps) (*steps)++;
            }
        }
        while (cur->next_skip[lvl] &&
               cur->next_skip[lvl]->junctions[0]->segment_start->value <= target) {
            if (steps) (*steps)++;
            cur = cur->next_skip[lvl];
        }
    }
    return cur;
}

/* within a block, locate the junction whose segment contains target, and optionally its index */
/* Binary search within a single block's junctions[] array for the
 * junction whose [segment_start, segment_end] range brackets `target`.
 * If out_idx is non-NULL, writes the matching junction's index within
 * the block. Returns NULL if no junction in this block covers target
 * (i.e. target falls in a gap — shouldn't normally happen for values
 * that exist in the list, since segments are contiguous, but can occur
 * when searching for a non-existent value). The `hi >= 0` fallback
 * check after the loop catches the case where standard binary search
 * converges just past the correct bucket. */
static Junction *block_find_junction_and_idx(JunctionBlock *b, int32_t target, int32_t *out_idx) {
    if (!b || b->count == 0) return NULL;
    int32_t lo = 0, hi = b->count - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) >> 1;
        Junction *j = b->junctions[mid];
        if (target < j->segment_start->value) {
            hi = mid - 1;
        } else if (target > j->segment_end->value) {
            lo = mid + 1;
        } else {
            if (out_idx) *out_idx = mid;
            return j;
        }
    }
    if (hi >= 0) {
        Junction *cand = b->junctions[hi];
        if (target >= cand->segment_start->value && target <= cand->segment_end->value) {
            if (out_idx) *out_idx = hi;
            return cand;
        }
    }
    return NULL;
}

/* Convenience wrapper over block_find_junction_and_idx() for callers
 * that don't need the index. */
static Junction *block_find_junction(JunctionBlock *b, int32_t target) {
    return block_find_junction_and_idx(b, target, NULL);
}

/* O(1) removal of a junction from its block, given its index (if known) */
/* Removes junction j from block bk. If known_idx (its index within the
 * block) is already known from a prior search, this is O(remaining
 * elements in block) for the memmove and O(1) for the lookup;
 * otherwise it falls back to a linear scan of the block first. If
 * removal empties the block, the block itself is freed and spliced out
 * of jli->blocks[] (using its cached `index`), and every subsequent
 * block's `index` is shifted down by one to stay accurate. */
static void block_remove_junction_at(JLI *jli, Junction *j, JunctionBlock *bk, int32_t known_idx) {
    if (!bk || bk->count == 0) return;
    int32_t idx = known_idx;
    if (idx < 0) {
        for (int32_t i = 0; i < bk->count; i++) {
            if (bk->junctions[i] == j) { idx = i; break; }
        }
        if (idx < 0) return;
    }

    if (idx < bk->count - 1)
        memmove(&bk->junctions[idx], &bk->junctions[idx+1],
                (bk->count - idx - 1) * sizeof(Junction*));
    bk->count--;

    if (bk->count > 0) {
        /* No need to update range – it is live */
    } else {
        /* Block is now empty → remove it from the array using its stored index */
        int32_t bi = bk->index;
        if (bi >= 0 && bi < jli->num_blocks) {
            block_free(bk);
            memmove(&jli->blocks[bi], &jli->blocks[bi+1],
                    (jli->num_blocks - bi - 1) * sizeof(JunctionBlock*));
            jli->num_blocks--;
            /* Update indices of blocks after the removed one */
            for (int32_t i = bi; i < jli->num_blocks; i++)
                jli->blocks[i]->index = i;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   BUILD FROM SORTED (now wires blocks with index)
   ═══════════════════════════════════════════════════════════════════════ */
/* Full "global rebuild" of the junction layer from the current linked
 * list (jli->head), assumed already sorted. Frees all existing
 * junctions and blocks, then walks the list once, carving it into
 * fixed-size segments of jli->segment_size (the last segment may be
 * shorter) and building a fresh Junction — including its shortcuts and
 * junction node, computed inline rather than via build_shortcuts() —
 * for each one. Finishes by relinking the junction ring
 * (junc_relink), repartitioning into blocks (build_blocks), and
 * rebuilding the block skip list (build_block_skip_list).
 *
 * If junc_arr_push() ever fails (OOM) partway through, everything
 * built so far is torn down and the JLI is left with zero junctions/
 * blocks rather than a partially-built, inconsistent state. */
static void build_junctions(JLI *jli) {
    for (int32_t i = 0; i < jli->num_junctions; i++) junction_free(jli->junctions[i]);
    jli->num_junctions = 0;
    jli->last_junction = jli->last_prev_junction = NULL;
    if (!jli->head) {
        for (int32_t i = 0; i < jli->num_blocks; i++) block_free(jli->blocks[i]);
        jli->num_blocks = 0;
        return;
    }
    int32_t n = jli->length, seg_size = jli->segment_size, K = jli->K;
    Node *cur = jli->head;
    int32_t remaining = n, start_pos = 1;
    while (cur && remaining > 0) {
        int32_t end_pos = start_pos + seg_size - 1;
        if (end_pos > n) end_pos = n;
        int32_t actual = end_pos - start_pos + 1;
        Node *seg_start = cur, *seg_end = cur, *jn = cur;
        int32_t junc_off = (actual > 1) ? (actual-1)/2 : 0;
        for (int32_t off = 0; off < actual-1 && seg_end->next; off++) seg_end = seg_end->next;
        Junction *j = junction_new(seg_start, K);
        j->segment_start = seg_start; j->segment_end = seg_end; j->segment_len = actual;
        {
            int32_t *int_offs = malloc(K * sizeof(int32_t));
            if (int_offs) {
                int32_t n_int = plan_shortcut_offsets(actual, K, int_offs, K);
                for (int a = 0; a < n_int-1; a++)
                    for (int b = a+1; b < n_int; b++)
                        if (int_offs[a] > int_offs[b]) { int32_t t = int_offs[a]; int_offs[a] = int_offs[b]; int_offs[b] = t; }
                int32_t total = 2 + n_int;
                if (total < K) total = K;
                if (total > K) total = K;
                j->shortcuts_len = total;
                j->shortcuts[0] = seg_start;
                Node *c2 = seg_start;
                int32_t idx2 = 0, sc_next = 0, int_slot = 1;
                while (c2) {
                    if (idx2 == junc_off) { jn = c2; j->junction_offset = idx2; }
                    if (sc_next < n_int && idx2 == int_offs[sc_next] && int_slot < total-1) {
                        j->shortcuts[int_slot++] = c2; sc_next++;
                    }
                    if (c2 == seg_end) break;
                    c2 = c2->next; idx2++;
                }
                while (int_slot < total-1) j->shortcuts[int_slot++] = seg_start;
                j->shortcuts[total-1] = seg_end;
                j->node = jn;
                free(int_offs);
            }
        }
        if (!junc_arr_push(jli, j)) {
            for (int32_t k = 0; k < jli->num_junctions; k++)
                junction_free(jli->junctions[k]);
            free(jli->junctions);
            jli->junctions = NULL;
            jli->num_junctions = 0;
            jli->junctions_cap = 0;
            for (int32_t k = 0; k < jli->num_blocks; k++) block_free(jli->blocks[k]);
            jli->num_blocks = 0;
            return;
        }
        cur = seg_end->next;
        remaining -= actual;
        start_pos = end_pos + 1;
    }
    junc_relink(jli);
    build_blocks(jli);
    build_block_skip_list(jli);
}

/* Public entry point: replaces the entire contents of jli with a fresh
 * linked list built from `values[0..n-1]`, which the caller must
 * already have sorted ascending (this function does not sort). Frees
 * the previous list, then delegates to build_junctions() to
 * (re)construct the junction/block layers on top of it. */
void jli_build_from_sorted(JLI *jli, const int32_t *values, int32_t n) {
    Node *c = jli->head;
    while (c) { Node *nx = c->next; free(c); c = nx; }
    jli->head = NULL; jli->length = 0;
    Node *prev = NULL;
    for (int32_t i = 0; i < n; i++) {
        Node *nd = node_new(values[i], NULL);
        if (!prev) jli->head = nd;
        else       prev->next = nd;
        prev = nd;
    }
    jli->length = n;
    build_junctions(jli);
}

/* ═══════════════════════════════════════════════════════════════════════
   SEARCH (uses live range)
   ═══════════════════════════════════════════════════════════════════════ */
/* Public read-only search for `target`. Returns the matching Node, or
 * NULL if not present. Records the step count into
 * jli->last_search_steps for benchmarking, and caches the containing
 * junction into jli->last_junction on success.
 *
 * Path:
 *   1. block_skip_search() to find the candidate block in expected
 *      O(log num_blocks) via the block skip list.
 *   2. Bounds-check target against that block's live range; bail out
 *      early if it falls outside (list doesn't contain it).
 *   3. block_find_junction() binary-searches within the block for the
 *      junction whose segment covers target.
 *   4. Fast path: if the junction's own node already matches, return
 *      immediately.
 *   5. Otherwise binary search the junction's shortcuts[] array to
 *      find the tightest known lower bound (best_idx), then walk
 *      forward node-by-node from there (also considering j->node as a
 *      possibly-tighter starting point) until we hit target, pass it
 *      (not found), or run out of segment. __builtin_prefetch hints
 *      the next node's cache line during this walk. */
Node *jli_search(JLI *jli, int32_t target) {
    int64_t steps = 0;
    jli->search_count++;

    if (!jli->head || jli->num_junctions == 0) {
        jli->last_search_steps = steps;
        return NULL;
    }

    JunctionBlock *bk = block_skip_search(jli, target, &steps);
    if (!bk) {
        jli->last_search_steps = steps;
        return NULL;
    }

    if (target < bk->junctions[0]->segment_start->value ||
        target > bk->junctions[bk->count-1]->segment_end->value) {
        jli->last_search_steps = steps;
        return NULL;
    }

    Junction *cj = block_find_junction(bk, target);
    if (!cj) {
        jli->last_search_steps = steps;
        return NULL;
    }

    jli->last_junction = cj;

    if (cj->node && cj->node->value == target) {
        steps++;
        jli->last_search_steps = steps;
        return cj->node;
    }

    int32_t lo = 0, hi = cj->shortcuts_len - 1;
    int32_t best_idx = -1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) >> 1;
        Node *cur = cj->shortcuts[mid];
        steps++;
        int32_t v = cur->value;
        if (v == target) {
            jli->last_junction = cj;
            jli->last_search_steps = steps;
            return cur;
        } else if (v < target) {
            best_idx = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    Node *start = (best_idx >= 0) ? cj->shortcuts[best_idx] : cj->segment_start;
    if (cj->node && cj->node->value <= target && cj->node->value > start->value)
        start = cj->node;

    Node *cur = start;
    while (cur) {
        steps++;
        if (cur->value == target) {
            jli->last_junction = cj;
            jli->last_search_steps = steps;
            return cur;
        }
        if (cur->value > target) break;
        __builtin_prefetch(cur->next, 0, 1);
        cur = cur->next;
    }

    jli->last_search_steps = steps;
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
   MUTATION SEARCH (extended result with block info)
   ═══════════════════════════════════════════════════════════════════════ */
/* Result of a mutation-oriented search: unlike jli_search() (which
 * only needs to answer "is it present"), insert/delete need the
 * surrounding splice points and enough context to update the junction/
 * block layers without re-searching from scratch.
 *   prev / cur : cur is the first node with value >= target (or NULL
 *                if target would go at the very end); prev is the
 *                node immediately before it (or NULL if cur is head).
 *                This is exactly the pair needed to splice a new node
 *                in, or to identify cur as the node to delete if
 *                cur->value == target.
 *   j          : the junction whose segment contains this splice
 *                point (or the nearest one, in edge-of-list cases).
 *   block / bi / ji : if the junction was located via the block skip-
 *                list path, these cache which block/index it came
 *                from, so callers like jli_delete can remove it from
 *                its block in O(1) instead of re-scanning all blocks.
 *                Left as NULL/-1/-1 when the fallback binary-search-
 *                over-junctions path was used instead (no block info
 *                available in that path). */
typedef struct {
    Node *prev;
    Node *cur;
    Junction *j;
    JunctionBlock *block;   /* block containing the returned junction, if found via block path */
    int32_t bi;             /* index of that block in jli->blocks */
    int32_t ji;             /* index of the junction inside the block */
} MutResult;

/* Fallback: plain O(n) linear walk of the raw linked list to find the
 * splice point for `target`, used when the junction/block layers are
 * absent, inconsistent, or a bounded walk elsewhere gave up. Includes
 * a runaway-loop guard (aborts with a diagnostic if it walks more than
 * length+10 steps, which should be structurally impossible for a
 * well-formed list but guards against a corrupted `next` chain). */
static MutResult linear_scan(JLI *jli, int32_t target) {
    MutResult r = {NULL, jli->head, NULL, NULL, -1, -1};
    int64_t steps = 0;
    while (r.cur && r.cur->value < target) {
        steps++;
        r.prev = r.cur;
        r.cur = r.cur->next;
        if (steps > jli->length + 10) {
            fprintf(stderr, "linear_scan infinite loop? steps=%lld\n", (long long)steps);
            break;
        }
    }
    jli->last_search_steps = steps;
    return r;
}

/* Core search used by both jli_insert() and jli_delete() to locate the
 * splice point for `target`, along with the owning junction and (where
 * available) its block context — see MutResult's field comments above.
 *
 * Path, roughly mirroring jli_search() but returning richer state and
 * being tolerant of `target` not being present:
 *   1. block_skip_search() to find the candidate block.
 *   2. If the target falls outside that block's own junctions but
 *      still could belong to an adjacent junction/block (edge cases:
 *      target is just below the block's first junction or just above
 *      its last), step to the neighboring junction/block as needed.
 *   3. Once a candidate junction cj is settled and target falls inside
 *      its live [segment_start, segment_end] range: binary-search its
 *      shortcuts (and consider j->node) for the tightest lower bound,
 *      then walk forward from there up to a bounded number of steps
 *      (segment_len + 5); if that bound is exceeded (shortcuts/segment
 *      state is somehow inconsistent with segment_len), give up and
 *      fall back to a full linear_scan() rather than risk a wrong
 *      answer.
 *   4. If the block path didn't yield a usable candidate at all, fall
 *      back to a binary search directly over jli->junctions[] (no
 *      block info available in this path, so MutResult.block/bi stay
 *      unset), with the same shortcut-search-then-walk logic.
 *   5. If binary search over junctions also fails to bracket target
 *      inside any single junction's range, target falls in a
 *      between-junctions gap; use whichever of the two bracketing
 *      junctions is appropriate to report prev/cur, or fall back to
 *      linear_scan() for the remaining edge cases. */
static MutResult mutation_search(JLI *jli, int32_t target) {
    int64_t steps = 0;
    MutResult empty = {NULL, NULL, NULL, NULL, -1, -1};
    if (!jli->head || jli->num_junctions == 0)
        return linear_scan(jli, target);

    JunctionBlock *bk = block_skip_search(jli, target, &steps);
    Junction *cj = NULL;
    int32_t j_idx_in_block = -1;
    if (bk) {
        cj = block_find_junction_and_idx(bk, target, &j_idx_in_block);
        if (!cj) {
            int32_t bi = bk->index;
            if (bi < 0) return linear_scan(jli, target); // shouldn't happen

            if (target < bk->junctions[0]->segment_start->value) {
                cj = bk->junctions[0];
                j_idx_in_block = 0;
                if (target < cj->segment_start->value) {
                    if (bi > 0) {
                        JunctionBlock *prev_bk = jli->blocks[bi - 1];
                        cj = prev_bk->junctions[prev_bk->count - 1];
                        bk = prev_bk;
                        j_idx_in_block = prev_bk->count - 1;
                    } else {
                        cj = NULL;
                        bk = NULL;
                        j_idx_in_block = -1;
                    }
                }
            } else if (target > bk->junctions[bk->count-1]->segment_end->value) {
                cj = bk->junctions[bk->count - 1];
                j_idx_in_block = bk->count - 1;
                if (target > cj->segment_end->value) {
                    if (bi < jli->num_blocks - 1) {
                        JunctionBlock *next_bk = jli->blocks[bi + 1];
                        cj = next_bk->junctions[0];
                        bk = next_bk;
                        j_idx_in_block = 0;
                    } else {
                        cj = NULL;
                        bk = NULL;
                        j_idx_in_block = -1;
                    }
                }
            } else {
                cj = bk->junctions[bk->count - 1];
                j_idx_in_block = bk->count - 1;
            }
        }
    }

    if (cj) {
        int32_t ss = cj->segment_start->value, se = cj->segment_end->value;
        if (target >= ss && target <= se) {
            bool is_first = (jli->head == cj->segment_start);
            Node *seg_prev = is_first ? NULL : (cj->prev_junction ? cj->prev_junction->segment_end : NULL);

            Node *best_start = NULL;
            if (cj->shortcuts && cj->shortcuts_len > 0) {
                int32_t lo = 0, hi = cj->shortcuts_len - 1;
                int32_t best_idx = -1;
                while (lo <= hi) {
                    int32_t mid = (lo + hi) >> 1;
                    Node *cand = cj->shortcuts[mid];
                    steps++;
                    if (cand->value < target) {
                        best_idx = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                if (best_idx >= 0) best_start = cj->shortcuts[best_idx];
            }

            if (cj->node && cj->node->value < target) {
                steps++;
                if (!best_start || cj->node->value > best_start->value)
                    best_start = cj->node;
            }

            Node *prev = seg_prev;
            Node *cur = cj->segment_start;
            if (best_start && best_start->value < target) {
                prev = best_start;
                cur = best_start->next;
            }

            int walk_limit = cj->segment_len + 5;
            int walk_steps = 0;
            while (cur && cur->value < target && walk_steps < walk_limit) {
                steps++;
                prev = cur;
                cur = cur->next;
                walk_steps++;
            }
            if (walk_steps >= walk_limit) {
                jli->last_search_steps = steps;
                return linear_scan(jli, target);
            }

            jli->last_junction = cj;
            jli->last_prev_junction = is_first ? NULL : cj->prev_junction;
            jli->last_search_steps = steps;
            return (MutResult){ prev, cur, cj, bk, bk ? bk->index : -1, j_idx_in_block };
        }
    }

    /* Fallback binary search over junctions – we don't have block info here */
    int32_t left = 0, right = jli->num_junctions - 1;
    while (left <= right) {
        int32_t mid = (left + right) >> 1;
        cj = jli->junctions[mid];
        int32_t ss = cj->segment_start->value, se = cj->segment_end->value;
        if (target >= ss && target <= se) {
            Node *seg_prev = (mid > 0) ? jli->junctions[mid-1]->segment_end : NULL;
            Node *best_start = NULL;
            if (cj->shortcuts && cj->shortcuts_len > 0) {
                int32_t lo = 0, hi = cj->shortcuts_len - 1;
                int32_t best_idx = -1;
                while (lo <= hi) {
                    int32_t mid2 = (lo + hi) >> 1;
                    Node *cand = cj->shortcuts[mid2];
                    steps++;
                    if (cand->value < target) {
                        best_idx = mid2;
                        lo = mid2 + 1;
                    } else {
                        hi = mid2 - 1;
                    }
                }
                if (best_idx >= 0) best_start = cj->shortcuts[best_idx];
            }
            if (cj->node && cj->node->value < target) {
                steps++;
                if (!best_start || cj->node->value > best_start->value)
                    best_start = cj->node;
            }
            Node *prev = seg_prev;
            Node *cur = cj->segment_start;
            if (best_start && best_start->value < target) {
                prev = best_start;
                cur = best_start->next;
            }
            int walk_limit = cj->segment_len + 5;
            int walk_steps = 0;
            while (cur && cur->value < target && walk_steps < walk_limit) {
                steps++;
                prev = cur;
                cur = cur->next;
                walk_steps++;
            }
            if (walk_steps >= walk_limit) {
                jli->last_search_steps = steps;
                return linear_scan(jli, target);
            }
            jli->last_junction = cj;
            jli->last_prev_junction = (mid > 0) ? jli->junctions[mid-1] : NULL;
            jli->last_search_steps = steps;
            return (MutResult){ prev, cur, cj, NULL, -1, -1 }; /* no block info */
        } else if (target < ss) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    int32_t left_j = right, right_j = left;
    if (left_j >= 0 && right_j < jli->num_junctions) {
        Junction *lj = jli->junctions[left_j];
        Node *prev = lj->segment_end;
        Node *cur = jli->junctions[right_j]->segment_start;
        jli->last_prev_junction = (left_j > 0) ? jli->junctions[left_j-1] : NULL;
        jli->last_junction = lj;
        jli->last_search_steps = steps;
        return (MutResult){ prev, cur, lj, NULL, -1, -1 };
    }
    if (left_j < 0) {
        Junction *fj = jli->junctions[0];
        jli->last_prev_junction = NULL;
        jli->last_junction = fj;
        jli->last_search_steps = steps;
        return (MutResult){ NULL, fj->segment_start, fj, NULL, -1, -1 };
    }
    if (right_j >= jli->num_junctions) {
        Junction *lj = jli->junctions[left_j];
        jli->last_prev_junction = (left_j > 0) ? jli->junctions[left_j-1] : NULL;
        jli->last_junction = lj;
        jli->last_search_steps = steps;
        return (MutResult){ lj->segment_end, NULL, lj, NULL, -1, -1 };
    }

    jli->last_search_steps = steps;
    return linear_scan(jli, target);
}

/* ═══════════════════════════════════════════════════════════════════════
   MAINTENANCE FORWARD DECLARATIONS
   ═══════════════════════════════════════════════════════════════════════ */
static void maintenance_hook(JLI *jli);
static void perform_global_rebuild(JLI *jli);
static void suboptimal_rebuild_step(JLI *jli, int32_t *flagged, int32_t nf);

/* ═══════════════════════════════════════════════════════════════════════
   INSERT  (no more block_update_range_if_boundary)
   ═══════════════════════════════════════════════════════════════════════ */
/* Public insert. Returns false without modifying anything if `value`
 * is already present (no duplicates); otherwise splices a new node in
 * and returns true.
 *
 * Special case: if the list is currently empty, just create the first
 * node and do a full build_junctions() (there's nothing incremental to
 * update yet).
 *
 * Normal case:
 *   1. mutation_search() locates the splice point and owning junction.
 *   2. Splice the new node in between m.prev and m.cur.
 *   3. Patch up the owning junction j's bookkeeping in O(1) depending
 *      on where the insert landed:
 *        - before the old segment_start (or list head) → new node
 *          becomes the segment_start (and shortcuts[0] if that
 *          shortcut pointed at the old start); junction_offset and
 *          segment_len both grow by 1 since everything shifted right.
 *        - after the old segment_end → new node becomes segment_end
 *          (and the last shortcut slot); only segment_len grows.
 *        - in the interior → just segment_len grows, plus
 *          junction_offset grows by 1 if the insert landed before the
 *          junction node (shifting its position).
 *      (Shortcuts in the interior are deliberately NOT updated here —
 *      they go gradually stale until a maintenance pass notices and
 *      rebuilds them; see should_rebuild/local_rebuild.)
 *   4. If enable_rebuild, bump the maintenance counters and let
 *      maintenance_hook() decide whether a local/suboptimal/global
 *      rebuild is due. */
bool jli_insert(JLI *jli, int32_t value, void *payload) {
    Node *nn = node_new(value, payload);
    if (!jli->head) {
        jli->head = nn; jli->length = 1;
        build_junctions(jli);
        jli->insert_count++; return true;
    }
    MutResult m = mutation_search(jli, value);
    if (m.cur && m.cur->value == value) { free(nn); return false; }
    nn->next = m.cur;
    if (m.prev) m.prev->next = nn;
    else        jli->head    = nn;
    jli->length++;
    Junction *j = m.j;
    if (j) {
        if (!m.prev || value < j->segment_start->value) {
            j->segment_start = nn;
            if (j->shortcuts_len > 0 && j->shortcuts[0] == m.cur) j->shortcuts[0] = nn;
            j->junction_offset++;
            j->segment_len++;
        } else if (m.prev == j->segment_end) {
            j->segment_end = nn;
            if (j->shortcuts_len > 0) j->shortcuts[j->shortcuts_len-1] = nn;
            j->segment_len++;
        } else {
            if (m.prev && j->node && m.prev->value < j->node->value) j->junction_offset++;
            j->segment_len++;
        }
        /* range is live, no explicit update needed */
    }
    if (jli->enable_rebuild) {
        jli->local_ops_counter++;
        jli->sub_ops_counter++;
        maintenance_hook(jli);
    }
    jli->insert_count++;
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
   DELETE  (uses cached block info to avoid scans)
   ═══════════════════════════════════════════════════════════════════════ */
/* Finds an approximate midpoint node strictly between `left` and
 * `right` (both assumed to lie within junction j's segment), used to
 * pick a replacement junction node or shortcut after the previous one
 * is deleted, without doing a full build_shortcuts() rebuild.
 *
 * Primary approach: Floyd-style slow/fast pointer walk from `left`
 * (fast moves 2 steps per iteration, slow moves 1) bounded by
 * `max_steps` derived from segment_len, so slow lands near the
 * midpoint between left and right when fast reaches (or passes) right.
 *
 * Fallback (if the fast/slow walk doesn't cleanly terminate within
 * max_steps, e.g. because right isn't actually reachable from left
 * within the expected bound): a plain count-then-walk-halfway pass
 * between left and right->next.
 *
 * Either way, the result is validated to actually lie strictly between
 * left and right by value before being returned; if not, it falls back
 * to returning `left` itself, letting the caller detect the
 * degenerate case and trigger a full build_shortcuts() instead. */
static Node *safe_middle_between(Junction *j, Node *left, Node *right) {
    if (!left || !right) return left ? left : right;
    int32_t max_steps = (j->segment_len > 0 ? j->segment_len : 100) + 10;
    Node *slow = left;
    Node *fast = left;
    int32_t steps = 0;
    while (fast && fast->next && fast != right && fast->next != right && steps < max_steps) {
        fast = fast->next->next;
        slow = slow->next;
        steps++;
        if (slow == right || slow == NULL) break;
    }
    if (steps >= max_steps || slow == NULL) {
        int32_t count = 0;
        Node *p = left;
        while (p && p != right->next && count < max_steps) {
            count++;
            p = p->next;
        }
        if (count == 0) return left;
        int32_t mid = count / 2;
        p = left;
        for (int32_t i = 0; i < mid && p; i++) p = p->next;
        if (p && (p->value <= left->value || p->value >= right->value)) p = left;
        return p ? p : left;
    }
    if (slow->value <= left->value || slow->value >= right->value) return left;
    return slow;
}

/* Public delete. Returns false without modifying anything if `value`
 * isn't present; otherwise removes it and returns true.
 *
 * High-level flow:
 *   1. mutation_search() locates the node and its junction.
 *   2. Splice it out of the raw linked list.
 *   3. If the list is now empty, tear down all junctions/blocks and
 *      return early.
 *   4. Otherwise patch the owning junction j:
 *        - if the deleted node was segment_start/segment_end/a
 *          shortcut, repoint those fields at the appropriate neighbor.
 *        - adjust junction_offset if the deletion happened before the
 *          junction node (shifting its position left by one).
 *        - segment_len shrinks by 1.
 *   5. If j->segment_len has dropped to <= 2, the segment is too small
 *      to be worth keeping on its own: "dissolve" it — either merge it
 *      into a neighboring junction (extending that junction's segment
 *      to absorb this one's remaining nodes and rebuilding its
 *      shortcuts), or, if it's now truly empty (segment_len == 0),
 *      simply splice it out of the list and remove the junction
 *      entirely with no merge target needed. This path updates the
 *      block layer directly (block_remove_junction_at) rather than
 *      going through the general per-op maintenance counters, and
 *      jumps via `goto skip_update` to the shared relink+skip-list-
 *      rebuild tail once done.
 *   6. Otherwise (segment_len still > 2, normal case): if the deleted
 *      node was the junction node itself, find a new one via
 *      safe_middle_between() (falling back to a full
 *      build_shortcuts() if that doesn't yield something valid);
 *      similarly, if it was an interior shortcut, try to patch just
 *      that one slot via safe_middle_between() between its neighbors,
 *      falling back to a full rebuild if that's not clean either.
 *      Then bump maintenance counters and let maintenance_hook() run
 *      as usual.
 *
 * `skip_update:` is reached only by the dissolve/merge path (step 5)
 * and performs the bookkeeping common to both of its sub-cases:
 * clearing the search cache, relinking the junction ring, and fully
 * rebuilding the block skip list (needed because dissolve/merge can
 * change which junctions exist, not just their contents). */
bool jli_delete(JLI *jli, int32_t value) {
    MutResult m = mutation_search(jli, value);
    if (!m.cur || m.cur->value != value) return false;
    Node *nd = m.cur;
    Node *nxt = nd->next;
    if (m.prev) m.prev->next = nxt;
    else        jli->head    = nxt;
    jli->length--;

    if (jli->length == 0) {
        for (int32_t i = 0; i < jli->num_junctions; i++) junction_free(jli->junctions[i]);
        free(jli->junctions);
        jli->junctions = NULL;
        jli->num_junctions = 0;
        jli->junctions_cap = 0;
        for (int32_t i = 0; i < jli->num_blocks; i++) block_free(jli->blocks[i]);
        jli->num_blocks = 0;
        free(nd);
        jli->delete_count++;
        return true;
    }

    Junction *j = m.j;
    if (!j) {
        free(nd);
        jli->delete_count++;
        return true;
    }

    bool is_seg_start = (nd == j->segment_start);
    bool is_seg_end   = (nd == j->segment_end);
    bool is_junc_node = (nd == j->node);
    bool is_shortcut = false;
    int32_t sc_idx = -1;
    for (int32_t i = 0; i < j->shortcuts_len; i++) {
        if (j->shortcuts[i] == nd) {
            is_shortcut = true;
            sc_idx = i;
            break;
        }
    }

    if (is_seg_start) {
        j->segment_start = nxt;
        for (int32_t i = 0; i < j->shortcuts_len; i++)
            if (j->shortcuts[i] == nd) j->shortcuts[i] = nxt;
    }
    if (is_seg_end) {
        j->segment_end = m.prev;
        for (int32_t i = 0; i < j->shortcuts_len; i++)
            if (j->shortcuts[i] == nd) j->shortcuts[i] = m.prev;
    }
    if (!is_junc_node) {
        if (is_seg_start && j->junction_offset > 0) j->junction_offset--;
        else if (!is_seg_start && nd->value < j->node->value) j->junction_offset--;
    }
    j->segment_len--;

    /* ── dissolve / merge case ── */
    if (j->segment_len <= 2) {
        JunctionBlock *bk = m.block;
        int32_t j_idx     = m.ji;
        if (!bk) {
            bk = NULL;
            for (int32_t i = 0; i < jli->num_blocks; i++) {
                JunctionBlock *b = jli->blocks[i];
                for (int32_t k = 0; k < b->count; k++) {
                    if (b->junctions[k] == j) { bk = b; j_idx = k; break; }
                }
                if (bk) break;
            }
        }

        if (jli->num_junctions == 1) {
            build_shortcuts(j, j->segment_len, jli->K);
            free(nd);
            jli->delete_count++;
            return true;
        }

        int32_t ji = -1;
        for (int32_t i = 0; i < jli->num_junctions; i++) {
            if (jli->junctions[i] == j) { ji = i; break; }
        }
        if (ji == -1) {
            build_shortcuts(j, j->segment_len, jli->K);
            free(nd);
            jli->delete_count++;
            return true;
        }

        Junction *target = NULL;
        if (ji == 0 && ji + 1 < jli->num_junctions) target = jli->junctions[ji+1];
        else if (ji > 0) target = jli->junctions[ji-1];

        if (target) {
            if (j->segment_len == 0) {
                Node *after = (j->segment_end ? j->segment_end->next : NULL);
                Node *before = (ji > 0) ? jli->junctions[ji-1]->segment_end : NULL;
                if (before) before->next = after;
                else        jli->head    = after;

                block_remove_junction_at(jli, j, bk, j_idx);
                junction_free(j);
                memmove(&jli->junctions[ji], &jli->junctions[ji+1],
                        (jli->num_junctions - ji - 1) * sizeof(Junction*));
                jli->num_junctions--;
                free(nd);
                jli->dissolve_count++;
                goto skip_update;
            }

            /* ── Merge the two junctions ── */
            if (ji == 0) {
                target->segment_start = j->segment_start;
                target->segment_len += j->segment_len;
                target->junction_offset += j->segment_len;
                if (target->shortcuts_len > 0) target->shortcuts[0] = j->segment_start;
            } else {
                target->segment_end = j->segment_end;
                target->segment_len += j->segment_len;
                if (target->shortcuts_len > 0)
                    target->shortcuts[target->shortcuts_len - 1] = j->segment_end;
            }

            /* ── FIX: Rebuild target shortcuts and node ── */
            build_shortcuts(target, target->segment_len, jli->K);
            /* Recompute the junction node (middle element) */
            int32_t off = 0;
            Node *p = target->segment_start;
            while (p && off < target->segment_len / 2) {
                p = p->next;
                off++;
            }
            target->node = p;
            target->junction_offset = off;

            block_remove_junction_at(jli, j, bk, j_idx);
            junction_free(j);
            memmove(&jli->junctions[ji], &jli->junctions[ji+1],
                    (jli->num_junctions - ji - 1) * sizeof(Junction*));
            jli->num_junctions--;
            if (ji == 0) jli->head = target->segment_start;

            free(nd);
            jli->dissolve_count++;
            goto skip_update;
        }

        build_shortcuts(j, j->segment_len, jli->K);
        free(nd);
        jli->delete_count++;
        return true;
    }

    /* ── normal deletion (segment_len > 2) ── */
    if (is_junc_node) {
        Node *new_mid = safe_middle_between(j, j->segment_start, j->segment_end);
        if (new_mid &&
            new_mid->value >= j->segment_start->value &&
            new_mid->value <= j->segment_end->value) {
            j->node = new_mid;
            int32_t off = 0;
            Node *p = j->segment_start;
            while (p && p != new_mid && off <= j->segment_len) {
                off++;
                p = p->next;
            }
            j->junction_offset = off;
        } else {
            build_shortcuts(j, j->segment_len, jli->K);
        }
    }
    if (is_shortcut && !is_seg_start && !is_seg_end) {
        Node *left  = (sc_idx > 0) ? j->shortcuts[sc_idx - 1] : j->segment_start;
        Node *right = (sc_idx + 1 < j->shortcuts_len) ? j->shortcuts[sc_idx + 1] : j->segment_end;
        if (left && right &&
            left->value >= j->segment_start->value &&
            right->value <= j->segment_end->value) {
            Node *replacement = safe_middle_between(j, left, right);
            if (replacement && replacement->value > left->value && replacement->value < right->value)
                j->shortcuts[sc_idx] = replacement;
            else
                build_shortcuts(j, j->segment_len, jli->K);
        } else {
            build_shortcuts(j, j->segment_len, jli->K);
        }
    }

    free(nd);

    jli->local_ops_counter++;
    jli->sub_ops_counter++;
    if (jli->enable_rebuild) maintenance_hook(jli);
    jli->delete_count++;
    return true;

skip_update:
    jli->last_junction = NULL;
    jli->last_prev_junction = NULL;
    junc_relink(jli);
    build_block_skip_list(jli);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
   MAINTENANCE  (now with local block rebuild)
   ═══════════════════════════════════════════════════════════════════════
   Three tiers of maintenance, cheapest to most expensive:
     - local_rebuild        : re-shortcut ONE junction whose node has
                               drifted too far from center (see
                               should_rebuild). O(segment_len).
     - suboptimal_rebuild_step : re-partition a contiguous RANGE of
                               junctions whose segment lengths have
                               drifted too far from segment_size.
                               O(size of the affected region).
     - perform_global_rebuild : throw away everything and rebuild from
                               the raw list via build_junctions().
                               O(n). Used only as a last resort when
                               the other two tiers can't keep up.
   maintenance_hook() (further below) decides which tier(s) to run,
   based on op counters and periodic scans. */

/* Re-derives junction j's shortcuts/junction-node from scratch (via
 * build_shortcuts) because its junction node has drifted too far off
 * center (see should_rebuild). Also tallies how many nodes were
 * touched (walked) doing so, into jli->local_rebuild_node_touches, for
 * benchmarking/instrumentation. */
static void local_rebuild(JLI *jli, Junction *j, int32_t seg_len) {
    if (seg_len <= 1) return;
    int64_t touches = 0;
    Node *cur = j->segment_start;
    while (cur) {
        touches++;
        if (cur == j->segment_end) break;
        cur = cur->next;
    }
    jli->local_rebuild_node_touches += touches;
    build_shortcuts(j, seg_len, jli->K);
    jli->local_rebuild_events++;
}

/* Decides whether junction j's junction node has drifted far enough
 * off-center to warrant a local_rebuild(). Compares
 * junction_offset/segment_len (the node's position as a fraction of
 * the segment) against 0.5 ± t_j — i.e. t_j is a tolerance band around
 * the ideal midpoint; drifting outside that band triggers a rebuild.
 *
 * Defensive path: if junction_offset looks stale/out-of-range (e.g.
 * negative or >= segment_len, which can happen after a run of edits
 * whose incremental bookkeeping didn't perfectly track it), this
 * re-derives it by walking the segment to find where `node` actually
 * is; if `node` isn't found in the segment at all (shouldn't normally
 * happen), it conservatively reports "no rebuild needed" rather than
 * risk acting on bad data. */
static bool should_rebuild(Junction *j, double t_j) {
    int32_t total = j->segment_len;
    if (total <= 1 || !j->node) return false;
    int32_t left = j->junction_offset;
    if (left < 0 || left >= total) {
        left = 0; bool seen = false;
        Node *c = j->segment_start;
        while (c) {
            if (c == j->node) { seen = true; break; }
            left++;
            if (c == j->segment_end) break;
            c = c->next;
        }
        if (!seen) return false;
        j->junction_offset = left;
    }
    double ratio = (double)left / total;
    return (ratio < 0.5 - t_j) || (ratio > 0.5 + t_j);
}

/* The "local" maintenance pass: scans every junction and local-
 * rebuilds any whose junction node has drifted off-center per
 * should_rebuild(). Triggered periodically by maintenance_hook() based
 * on jli->local_interval mutations having occurred. */
static void maintenance_step(JLI *jli) {
    for (int32_t i = 0; i < jli->num_junctions; i++) {
        Junction *j = jli->junctions[i];
        if (should_rebuild(j, jli->t_j)) local_rebuild(jli, j, j->segment_len);
    }
    jli->local_scan_counter++;
}

/* Scans every junction's segment_len against the target segment_size
 * and classifies its relative drift:
 *   drift = |segment_len - segment_size| / segment_size
 * Junctions with drift >= hard_pct go into *hard_out (severely
 * oversized/undersized); junctions with drift >= soft_pct (but below
 * hard_pct) go into *flagged_out only. Every hard-flagged junction is
 * also included in *flagged_out, so *flagged_out is always a superset
 * of *hard_out. *seg_ratio is the fraction of all junctions that were
 * flagged (soft or hard) — a coarse "how out of shape is the whole
 * structure" signal used by maintenance_hook() to decide whether
 * suboptimal rebuilds are keeping up or a global rebuild is needed.
 * Caller owns and must free() *flagged_out and *hard_out. */
static void scan_suboptimal(JLI *jli, int32_t **flagged_out, int32_t *nf,
                             int32_t **hard_out, int32_t *nh, double *seg_ratio) {
    int32_t n = jli->num_junctions;
    *flagged_out = malloc(n * sizeof(int32_t));
    *hard_out    = malloc(n * sizeof(int32_t));
    *nf = *nh = 0;
    for (int32_t i = 0; i < n; i++) {
        Junction *j = jli->junctions[i];
        int32_t sl  = j->segment_len;
        double drift = (double)(sl > jli->segment_size ? sl-jli->segment_size : jli->segment_size-sl)
                       / jli->segment_size;
        if (drift >= jli->hard_pct) { (*hard_out)[(*nh)++] = i; (*flagged_out)[(*nf)++] = i; }
        else if (drift >= jli->soft_pct) { (*flagged_out)[(*nf)++] = i; }
    }
    *seg_ratio = n > 0 ? (double)*nf / n : 0.0;
}

/* The heaviest maintenance tier: throws away the entire junction/block
 * layer and rebuilds it from scratch off the raw sorted list via
 * build_junctions(). Resets the maintenance-depth guard and all
 * per-interval operation counters (since everything is now freshly
 * optimal), and bumps global_rebuild_events for instrumentation. */
static void perform_global_rebuild(JLI *jli) {
    jli->maintenance_depth = 0;
    build_junctions(jli);
    jli->last_junction = jli->last_prev_junction = NULL;
    jli->local_ops_counter = jli->sub_ops_counter = 0;
    jli->dissolve_count = 0;
    jli->global_rebuild_events++;
}

/*
 * New helper: rebuild only the blocks that intersect the junction range [si, ei].
 * The new junctions (new_js, count new_count) are packed into fresh blocks
 * according to the hard_max rule, and the block array is spliced in place.
 *
 * Summary of the 4 phases below:
 *   1. Identify every block containing at least one of old_js (by
 *      pointer identity, not value — see inline comment) and collect
 *      any "survivor" junctions in those blocks that aren't part of
 *      this rebuild, so they aren't lost when the block is discarded.
 *   2. Merge survivors back in around new_js (preserving ascending
 *      order) and pack the result into fresh blocks up to
 *      block_hard_max each.
 *   3. Splice the new blocks into jli->blocks[] in place of every
 *      matched block, compacting out gaps — this is correct even when
 *      matched blocks aren't contiguous in the array.
 *   4. Renumber every block's `index` field to match its new position.
 */
static void rebuild_blocks_for_region(JLI *jli, Junction **old_js, int32_t old_count,
                                       Junction **new_js, int32_t new_count) {
    if (new_count == 0 && old_count == 0) return;

    /* 1. Find every block that holds at least one of the OLD junctions, by
     * pointer identity. Value-bound heuristics are not safe here: blocks
     * get progressively fragmented by repeated rebuilds, and a stale block
     * elsewhere in the array can coincidentally satisfy a value comparison
     * while still holding pointers to junctions freed by an earlier,
     * unrelated rebuild. We track every matching block explicitly (not just
     * a [first,last] span) because matches are not guaranteed contiguous;
     * collapsing to a span would wrongly discard unrelated blocks sitting
     * between two matches. */
    bool *is_match = old_count > 0 ? calloc(jli->num_blocks, sizeof(bool)) : NULL;
    int32_t match_count = 0;
    /* Junctions living in a matched block that are NOT themselves part of
     * old_js ("survivors") would otherwise be silently dropped: a matched
     * block is deleted wholesale below, but it may be only partially owned
     * by this rebuild (block_hard_max packs unrelated neighboring junctions
     * together regardless of maintenance history). Collect survivors here
     * and splice them back in with new_js so nothing is lost. */
    Junction **survivors = NULL;
    int32_t survivor_count = 0;
    if (old_count > 0) {
        int32_t survivor_cap = 0;
        for (int32_t i = 0; i < jli->num_blocks; i++) {
            JunctionBlock *b = jli->blocks[i];
            if (b->count == 0) continue;
            bool hit = false;
            for (int32_t k = 0; k < b->count && !hit; k++) {
                for (int32_t oi = 0; oi < old_count; oi++) {
                    if (b->junctions[k] == old_js[oi]) { hit = true; break; }
                }
            }
            if (hit) {
                is_match[i] = true;
                match_count++;
                for (int32_t k = 0; k < b->count; k++) {
                    bool is_old = false;
                    for (int32_t oi = 0; oi < old_count; oi++) {
                        if (b->junctions[k] == old_js[oi]) { is_old = true; break; }
                    }
                    if (!is_old) {
                        if (survivor_count == survivor_cap) {
                            survivor_cap = survivor_cap ? survivor_cap * 2 : 4;
                            survivors = realloc(survivors, survivor_cap * sizeof(Junction*));
                        }
                        survivors[survivor_count++] = b->junctions[k];
                    }
                }
            }
        }
    }

    /* Merge survivors with new_js, keeping ascending key order. Survivors can
     * only fall strictly before or strictly after the new_js range (a matched
     * block's junctions are contiguous in jli->junctions[], and old_js occupy
     * one contiguous run within that block), so a simple partition suffices. */
    Junction **merged_new_js = new_js;
    int32_t merged_new_count = new_count;
    if (survivor_count > 0) {
        merged_new_count = new_count + survivor_count;
        merged_new_js = malloc(merged_new_count * sizeof(Junction*));
        int32_t before = 0;
        int32_t ref_value = new_count > 0 ? new_js[0]->segment_start->value
                                           : (old_count > 0 ? old_js[0]->segment_start->value : 0);
        for (int32_t i = 0; i < survivor_count; i++)
            if (survivors[i]->segment_start->value < ref_value) before++;
        int32_t w2 = 0;
        for (int32_t i = 0; i < before; i++) merged_new_js[w2++] = survivors[i];
        for (int32_t i = 0; i < new_count; i++) merged_new_js[w2++] = new_js[i];
        for (int32_t i = before; i < survivor_count; i++) merged_new_js[w2++] = survivors[i];
    }
    free(survivors);

    /* 2. Calculate configuration for new blocks based on jli->block_hard_max */
    int32_t hard_max = jli->block_hard_max;
    int32_t num_new_blocks = (merged_new_count + hard_max - 1) / hard_max;
    if (num_new_blocks == 0 && merged_new_count > 0) num_new_blocks = 1;

    JunctionBlock **new_blocks = malloc(num_new_blocks * sizeof(JunctionBlock*));
    if (!new_blocks && num_new_blocks > 0) {
        free(is_match);
        if (merged_new_js != new_js) free(merged_new_js);
        return;
    }

    int32_t src = 0;
    for (int32_t b = 0; b < num_new_blocks; b++) {
        int32_t sz = merged_new_count - src;
        if (sz > hard_max) sz = hard_max;

        JunctionBlock *block = block_new(sz);
        if (!block) {
            for (int32_t k = 0; k < b; k++) block_free(new_blocks[k]);
            free(new_blocks);
            free(is_match);
            if (merged_new_js != new_js) free(merged_new_js);
            return;
        }
        block->index = -1;
        for (int32_t k = 0; k < sz; k++) {
            block->junctions[k] = merged_new_js[src++];
            block->count++;
        }
        new_blocks[b] = block;
    }

    /* 3. Splice new blocks back into the structure by compacting out every
     * matched block in place (regardless of whether matches are contiguous)
     * and inserting new_blocks at the position of the first match. This is
     * always correct, unlike a [first_bi,last_bi] span removal, which would
     * silently drop any unrelated block sitting between two non-adjacent
     * matches. */
    int32_t old_num_blocks = jli->num_blocks;
    int32_t new_total_blocks = old_num_blocks - match_count + num_new_blocks;

    JunctionBlock **rebuilt = malloc((new_total_blocks > 0 ? new_total_blocks : 1) * sizeof(JunctionBlock*));
    if (!rebuilt) {
        for (int32_t k = 0; k < num_new_blocks; k++) block_free(new_blocks[k]);
        free(new_blocks);
        free(is_match);
        return;
    }

    int32_t w = 0;
    bool inserted = false;
    for (int32_t i = 0; i < old_num_blocks; i++) {
        if (is_match && is_match[i]) {
            if (!inserted) {
                for (int32_t k = 0; k < num_new_blocks; k++) rebuilt[w++] = new_blocks[k];
                inserted = true;
            }
            block_free(jli->blocks[i]);
        } else {
            rebuilt[w++] = jli->blocks[i];
        }
    }
    if (!inserted) {
        /* No matches at all (pure insertion case) -- prepend new_blocks. */
        memmove(rebuilt + num_new_blocks, rebuilt, w * sizeof(JunctionBlock*));
        for (int32_t k = 0; k < num_new_blocks; k++) rebuilt[k] = new_blocks[k];
        w += num_new_blocks;
    }

    free(jli->blocks);
    jli->blocks = rebuilt;
    jli->blocks_cap = new_total_blocks > 0 ? new_total_blocks : 1;
    jli->num_blocks = w;

    /* 4. Update index numbers across blocks to keep positional layout accurate */
    for (int32_t i = 0; i < jli->num_blocks; i++) {
        jli->blocks[i]->index = i;
    }

    free(new_blocks);
    free(is_match);
    if (merged_new_js != new_js) free(merged_new_js);
}

/* The "suboptimal" (mid-tier) maintenance pass: given a sorted array
 * of flagged junction indices (`flagged`, count `nf`, from
 * scan_suboptimal), groups them into contiguous runs ("regions"),
 * then for each region (processed back-to-front so earlier indices
 * stay valid as later regions shift the array) re-partitions that
 * region's total node count into fresh, evenly-sized segments of
 * ~segment_size each and rebuilds junctions for them from scratch.
 *
 * Per region:
 *   - If the region's total length L is below min_reg (too small to
 *     be worth repartitioning on its own), grow it by pulling in one
 *     neighboring junction (prefer the one before, else the one
 *     after) before proceeding.
 *   - Compute how many new segments are needed and their sizes
 *     (seg_sizes), walk the region's nodes once, and build a fresh
 *     Junction (with shortcuts) for each new segment.
 *   - Splice the new junctions into jli->junctions[] in place of the
 *     old range [si, ei], growing the array first if the new count
 *     exceeds the old count.
 *   - Delegate to rebuild_blocks_for_region() to patch the block
 *     layer, then only AFTER that call free the old junctions — see
 *     the inline comment at that point for why the ordering matters
 *     (the blocks still reference the old junctions until
 *     rebuild_blocks_for_region has replaced those references).
 *
 * After all regions are processed: relinks the junction ring
 * (junc_relink) and does a full block-skip-list rebuild (cheaper to
 * just redo it globally than to patch it incrementally, since
 * potentially many blocks changed). */
static void suboptimal_rebuild_step(JLI *jli, int32_t *flagged, int32_t nf) {
    if (nf == 0) return;
    int32_t seg_size = jli->segment_size;
    int32_t min_reg  = (int32_t)(seg_size * jli->min_seg_len_pct);
    typedef struct { int32_t si, ei; } Region;
    Region *regions = malloc(nf * sizeof *regions);
    int32_t nr = 0, rs = flagged[0], re = flagged[0];
    for (int32_t i = 1; i < nf; i++) {
        if (flagged[i] == re+1) re = flagged[i];
        else { regions[nr++] = (Region){rs, re}; rs = re = flagged[i]; }
    }
    regions[nr++] = (Region){rs, re};
    int K = jli->K;
    int max_lvl = jli->max_skip_level;
    for (int32_t ri = nr-1; ri >= 0; ri--) {
        int32_t si = regions[ri].si, ei = regions[ri].ei;
        int32_t L = 0;
        for (int32_t i = si; i <= ei; i++) L += jli->junctions[i]->segment_len;
        if (L == 0) continue;
        if (L < min_reg) {
            if (si > 0) si--;
            else if (ei+1 < jli->num_junctions) ei++;
            else continue;
            L = 0;
            for (int32_t i = si; i <= ei; i++) L += jli->junctions[i]->segment_len;
        }
        int64_t region_touches = 0;
        for (int32_t i = si; i <= ei; i++) region_touches += jli->junctions[i]->segment_len;
        jli->suboptimal_rebuild_node_touches += region_touches;
        Node *region_start = jli->junctions[si]->segment_start;
        Node *region_end   = jli->junctions[ei]->segment_end;
        int32_t rem = L;
        int32_t *seg_sizes = malloc((L/seg_size+2) * sizeof(int32_t));
        int32_t ns = 0;
        while (rem > 0) { int32_t sz = rem < seg_size ? rem : seg_size; seg_sizes[ns++] = sz; rem -= sz; }
        Junction **new_js = malloc(ns * sizeof *new_js);
        Node *cur = region_start; int32_t s_i = 0;
        while (cur && s_i < ns) {
            int32_t sl = seg_sizes[s_i];
            Node *ss2 = cur, *se2 = cur;
            for (int32_t k = 0; k < sl-1 && se2->next; k++) se2 = se2->next;
            Junction *j2 = junction_new(ss2, K);
            j2->segment_start = ss2; j2->segment_end = se2;
            build_shortcuts(j2, sl, K);
            new_js[s_i] = j2; s_i++;
            if (cur == region_end) break;
            cur = se2->next; if (!cur) break;
        }
        int32_t old_count = ei-si+1, new_count = s_i, diff = new_count-old_count;
        /* NOTE: do NOT free the old junctions yet. rebuild_blocks_for_region
         * (called below) walks jli->blocks[], whose entries still point at
         * these old Junction objects -- freeing them here first leaves those
         * blocks holding dangling pointers and causes a heap-use-after-free.
         * The old junctions are freed further below, once nothing -- not
         * jli->junctions[] and not jli->blocks[] -- still references them.
         * Save the pointers now since jli->junctions[si..ei] is about to be
         * overwritten in place with the new junctions. */
        Junction **old_js = malloc(old_count * sizeof *old_js);
        for (int32_t i = 0; i < old_count; i++) old_js[i] = jli->junctions[si + i];
        int32_t new_total = jli->num_junctions + diff;
        if (new_total > jli->junctions_cap) {
            Junction **new_juncs = realloc(jli->junctions, new_total * 2 * sizeof(Junction*));
            if (!new_juncs) {
                for (int32_t k = 0; k < s_i; k++) junction_free(new_js[k]);
                free(seg_sizes);
                free(new_js);
                free(old_js);
                continue;
            }
            jli->junctions = new_juncs;
            jli->junctions_cap = new_total * 2;
        }
        memmove(&jli->junctions[si+new_count], &jli->junctions[ei+1],
                (jli->num_junctions-ei-1)*sizeof(Junction*));
        for (int32_t i = 0; i < new_count; i++) jli->junctions[si+i] = new_js[i];
        jli->num_junctions = new_total;
        free(seg_sizes);
        /* Rebuild blocks for this region */
        rebuild_blocks_for_region(jli, old_js, old_count, new_js, new_count);
        /* Safe now: rebuild_blocks_for_region has replaced every block
         * pointer that referenced the old junctions, so nothing dangles. */
        for (int32_t i = 0; i < old_count; i++) junction_free(old_js[i]);
        free(old_js);
        free(new_js);
    }
    free(regions);
    junc_relink(jli);
    /* Block skip‑list must be rebuilt globally because many blocks may have changed */
    build_block_skip_list(jli);
    jli->last_junction = jli->last_prev_junction = NULL;
    jli->suboptimal_rebuild_events++;
}

/* Called after every mutation (when enable_rebuild is set) to decide
 * whether any maintenance tier is due, and run it. Two independent
 * triggers, checked in order:
 *
 *   1. Local tier — if local_ops_counter has reached local_interval:
 *      normally runs maintenance_step() (local_rebuild on drifted
 *      junctions) and resets the counter. EXCEPT: if a suboptimal
 *      scan is imminent (within `stop_crash_local_sub` fraction of
 *      sub_interval), skip running local maintenance this time and
 *      only partially reset the counter (to local_interval/2) — this
 *      avoids doing local work that's likely about to be superseded
 *      by an imminent, more thorough suboptimal pass.
 *
 *   2. Suboptimal tier — if sub_ops_counter has reached sub_interval:
 *      run scan_suboptimal() to classify drift across all junctions.
 *        - If the *hard*-drift ratio already exceeds
 *          emergency_hard_segment_ratio, skip straight to a global
 *          rebuild (the structure is bad enough that a global rebuild
 *          is more effective than trying to patch it regionally).
 *        - Else if the *soft-or-hard* flagged ratio exceeds
 *          flagged_ratio_limit, run suboptimal_rebuild_step() over all
 *          flagged junctions.
 *        - Else if there's at least one hard-drifted junction (but not
 *          enough to hit flagged_ratio_limit), still run
 *          suboptimal_rebuild_step() but scoped to just the hard ones.
 *        - Track how many suboptimal rebuilds have fired since the
 *          last global rebuild (sub_event_before_global); once that
 *          count reaches min_suboptimal_events_before_global AND the
 *          flagged ratio is STILL >= max_suboptimal_segments (i.e.
 *          suboptimal rebuilds aren't converging fast enough), escalate
 *          to a global rebuild instead of continuing to chip away
 *          regionally.
 *
 * maintenance_depth guards against unbounded reentrant recursion (a
 * rebuild triggering further rebuilds); it's incremented on entry and
 * decremented on every exit path, and the function refuses to do any
 * work at all past depth 10, logging instead. */
static void maintenance_hook(JLI *jli) {
    if (jli->maintenance_depth > 10) {
        fprintf(stderr, "maintenance_hook recursion depth exceeded (%d)\n", jli->maintenance_depth);
        return;
    }
    jli->maintenance_depth++;
    if (jli->local_ops_counter >= jli->local_interval) {
        int32_t bw = (int32_t)(jli->sub_interval * jli->stop_crash_local_sub);
        if (jli->sub_interval > 0 && (jli->sub_interval - jli->sub_ops_counter) <= bw) {
            jli->local_ops_counter = jli->local_interval / 2;
            if (jli->local_ops_counter < 1) jli->local_ops_counter = 1;
        } else {
            jli->local_ops_counter = 0;
            maintenance_step(jli);
        }
    }
    if (jli->sub_ops_counter >= jli->sub_interval) {
        jli->sub_ops_counter = 0;
        jli->suboptimal_scan_count++;
        int32_t *flagged, *hard; int32_t nf, nh; double seg_ratio;
        scan_suboptimal(jli, &flagged, &nf, &hard, &nh, &seg_ratio);
        int32_t total = jli->num_junctions;
        double hr = total > 0 ? (double)nh/total : 0.0;
        if (hr >= jli->emergency_hard_segment_ratio) {
            jli->sub_event_before_global = 0;
            free(flagged); free(hard);
            perform_global_rebuild(jli);
            jli->maintenance_depth--;
            return;
        }
        double fr = total > 0 ? (double)nf/total : 0.0;
        bool rebuild_performed = false;
        if (fr >= jli->flagged_ratio_limit) {
            suboptimal_rebuild_step(jli, flagged, nf);
            rebuild_performed = true;
        } else if (nh > 0) {
            suboptimal_rebuild_step(jli, hard, nh);
            rebuild_performed = true;
        }
        /* increment only on sub rebuild — counts how many sub rebuilds have
         * fired since the last global rebuild, not how many scans occurred */
        if (rebuild_performed) {
            jli->sub_event_before_global++;
        }
        /* Gate 2: sub rebuilds are not converging — seg_ratio still high
         * after enough sub rebuilds → escalate to global */
        if (jli->sub_event_before_global >= jli->min_suboptimal_events_before_global
                && seg_ratio >= jli->max_suboptimal_segments) {
            jli->sub_event_before_global = 0;
            free(flagged); free(hard);
            perform_global_rebuild(jli);
            jli->maintenance_depth--;
            return;
        }
        free(flagged); free(hard);
    }
    jli->maintenance_depth--;
}

/* ═══════════════════════════════════════════════════════════════════════
   LIFECYCLE
   ═══════════════════════════════════════════════════════════════════════ */
/* Constructs and returns a new, empty JLI configured with the given
 * tunables (see the JLI struct's field comments and
 * validate_jli_parameters for what each one controls). Parameters are
 * validated first (validate_jli_parameters aborts the process on
 * invalid input); most numeric parameters that are <= 0 fall back to a
 * sane default rather than being rejected outright (segment_size,
 * shortcuts_per_junc, max_skip_level, local_interval, sub_interval,
 * block_size). Returns NULL on allocation failure. */
JLI *jli_create(int32_t segment_size,
                int32_t shortcuts_per_junc,
                int32_t max_skip_level,
                double  level_probability,
                int32_t local_interval,
                int32_t sub_interval,
                double  t_j,
                double  soft_pct,
                double  hard_pct,
                double  flagged_ratio,
                double  min_seg_len_pct,
                double  max_suboptimal,
                int32_t min_sub_events_before_global,
                double  emergency_hard_ratio,
                double  stop_crash_local_sub,
                int32_t block_size) {
    validate_jli_parameters(level_probability, emergency_hard_ratio, stop_crash_local_sub,
                             t_j, soft_pct, hard_pct, flagged_ratio, min_seg_len_pct,
                             max_suboptimal, local_interval, sub_interval, segment_size);
    JLI *jli = calloc(1, sizeof *jli);
    jli->segment_size     = segment_size > 0 ? segment_size : 100;
    jli->K                = shortcuts_per_junc > 2 ? shortcuts_per_junc : 3;
    jli->max_skip_level   = max_skip_level > 0 ? max_skip_level : 16;
    if (jli->max_skip_level > 255) jli->max_skip_level = 255;
    jli->level_probability = (level_probability >= 0.0 && level_probability <= 1.0)
                             ? level_probability : 0.5;
    jli->enable_rebuild   = true;
    jli->local_interval   = local_interval > 0 ? local_interval : 1000;
    jli->sub_interval     = sub_interval   > 0 ? sub_interval   : 2500;
    jli->t_j              = t_j;
    jli->soft_pct         = soft_pct;
    jli->hard_pct         = hard_pct;
    jli->flagged_ratio_limit                 = flagged_ratio;
    jli->min_seg_len_pct                     = min_seg_len_pct;
    jli->max_suboptimal_segments             = max_suboptimal;
    if (jli->max_suboptimal_segments > 1.0) jli->max_suboptimal_segments = 1.0;
    if (jli->max_suboptimal_segments < 0.0) jli->max_suboptimal_segments = 0.0;
    jli->min_suboptimal_events_before_global = min_sub_events_before_global;
    jli->emergency_hard_segment_ratio        = emergency_hard_ratio;
    jli->stop_crash_local_sub                = stop_crash_local_sub;
    jli->junctions_cap    = 64;
    jli->junctions        = malloc(64 * sizeof(Junction*));
    if (!jli->junctions) { free(jli); return NULL; }
    jli->block_m          = block_size > 0 ? block_size : 4;
    jli->block_hard_max   = jli->block_m + 2;
    jli->blocks           = NULL;
    jli->num_blocks       = 0;
    jli->blocks_cap       = 0;
    jli->block_skip_heads = calloc(jli->max_skip_level + 1, sizeof(JunctionBlock*));
    if (!jli->block_skip_heads) {
        free(jli->junctions);
        free(jli);
        return NULL;
    }
    jli->local_rebuild_node_touches = 0;
    jli->suboptimal_rebuild_node_touches = 0;
    return jli;
}

/* Frees an entire JLI: every list node, every junction, every block,
 * the block skip-list head array, and finally the JLI struct itself.
 * Safe to call with jli == NULL (no-op). Does not free node payloads —
 * those remain the caller's responsibility. */
void jli_destroy(JLI *jli) {
    if (!jli) return;
    Node *c = jli->head;
    while (c) { Node *nx = c->next; free(c); c = nx; }
    for (int32_t i = 0; i < jli->num_junctions; i++) junction_free(jli->junctions[i]);
    free(jli->junctions);
    for (int32_t i = 0; i < jli->num_blocks; i++) block_free(jli->blocks[i]);
    free(jli->blocks);
    free(jli->block_skip_heads);
    free(jli);
}

/* Returns the number of elements currently stored, or 0 for a NULL jli. */
int32_t jli_size(JLI *jli) { return jli ? jli->length : 0; }

/* Estimates total heap memory used by the structure itself (JLI struct
 * + every Node + junction array/structs/shortcuts + block array/
 * structs/skip-pointers), NOT including any external node payloads.
 * Used for the paper's memory/speed trade-off measurements. */
static int64_t jli_memory_bytes(JLI *jli) {
    int64_t bytes = (int64_t)sizeof(JLI);
    bytes += (int64_t)jli->length * sizeof(Node);
    bytes += (int64_t)jli->junctions_cap * (int64_t)sizeof(Junction *);
    for (int32_t i = 0; i < jli->num_junctions; i++) {
        Junction *j = jli->junctions[i];
        bytes += (int64_t)sizeof(Junction)
               + (int64_t)j->shortcuts_len * (int64_t)sizeof(Node *);
    }
    bytes += (int64_t)jli->blocks_cap * sizeof(JunctionBlock*);
    bytes += (int64_t)(jli->max_skip_level + 1) * sizeof(JunctionBlock*);
    for (int32_t i = 0; i < jli->num_blocks; i++) {
        JunctionBlock *b = jli->blocks[i];
        bytes += (int64_t)sizeof(JunctionBlock)
               + (int64_t)b->capacity * sizeof(Junction*)
               + (int64_t)(b->skip_level + 1) * sizeof(JunctionBlock*);
    }
    return bytes;
}

/* Same as jli_memory_bytes(), but additionally accounts for a uniform
 * per-node payload_size (in bytes) across every element — useful when
 * every node's payload is the same fixed size and its memory should be
 * counted as part of the structure's total footprint. */
static int64_t jli_memory_bytes_with_payload(JLI *jli, size_t payload_size) {
    int64_t bytes = jli_memory_bytes(jli);
    if (payload_size > 0) bytes += (int64_t)jli->length * (int64_t)payload_size;
    return bytes;
}

/* Snapshot of the running maintenance/benchmarking counters accumulated
 * in the JLI struct, exposed to callers via jli_get_maintenance_stats()
 * so external benchmark code doesn't need to reach into JLI's
 * otherwise-opaque internals. */
typedef struct {
    int64_t local_scan_count;
    int64_t sub_scan_count;
    int64_t local_rebuild_count;
    int64_t sub_rebuild_count;
    int64_t global_rebuild_count;
    int64_t local_node_touches;
    int64_t sub_node_touches;
    int64_t dissolve_count;                /* NEW: total dissolves (merges) */
    int64_t total_node_touches;            /* NEW: local+sub node touches */
} JLI_MaintenanceStats;

/* Copies the current maintenance/instrumentation counters out of jli
 * into *stats. Purely a read — does not reset any counters. */
void jli_get_maintenance_stats(JLI *jli, JLI_MaintenanceStats *stats) {
    stats->local_scan_count      = jli->local_scan_counter;
    stats->sub_scan_count        = jli->suboptimal_scan_count;
    stats->local_rebuild_count   = jli->local_rebuild_events;
    stats->sub_rebuild_count     = jli->suboptimal_rebuild_events;
    stats->global_rebuild_count  = jli->global_rebuild_events;
    stats->local_node_touches    = jli->local_rebuild_node_touches;
    stats->sub_node_touches      = jli->suboptimal_rebuild_node_touches;
    stats->dissolve_count        = jli->dissolve_count;               /* NEW */
    stats->total_node_touches    = jli->local_rebuild_node_touches
                                 + jli->suboptimal_rebuild_node_touches; /* NEW */
}

/* ═══════════════════════════════════════════════════════════════════════
   REFERENCE SKIP‑LIST (unchanged)
   ═══════════════════════════════════════════════════════════════════════
   A standard, self-contained skip list (William Pugh style), included
   here purely as a baseline to benchmark JLI against. It shares no
   code or state with the JLI structures above except the separate
   `rng_state` PRNG. p=0.5 per level, level count capped at a value
   derived from n via sl_max_level_for_n() rather than a fixed constant,
   so the cap scales sensibly with the dataset size used in a given
   benchmark run. */
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

/* SL_MAX_LEVEL is no longer a fixed constant.
   The cap is computed at runtime from n using ceil(log2(n)),
   then passed into sl_new().  We keep a hard ceiling of 64
   as a safety guard only. */
#define SL_HARD_MAX_LEVEL 64

/* Compute the appropriate max skip level for a skip list over n elements.
   Formula: ceil(log2(n)), clamped to [1, SL_HARD_MAX_LEVEL]. */
static int sl_max_level_for_n(int32_t n) {
    if (n <= 1) return 1;
    int lvl = 0;
    int32_t v = n - 1;
    while (v > 0) { v >>= 1; lvl++; }   /* lvl = ceil(log2(n)) */
    if (lvl < 1) lvl = 1;
    if (lvl > SL_HARD_MAX_LEVEL) lvl = SL_HARD_MAX_LEVEL;
    return lvl;
}

typedef struct SLNode {
    int32_t        val;
    int32_t        level;
    void          *payload;
    struct SLNode *fwd[];
} SLNode;

typedef struct {
    SLNode  *head;
    int      level;
    int      max_level;   /* runtime cap derived from n */
    int64_t  count;
    int64_t  total_node_bytes;
    int64_t  total_payload_bytes;
} SL;

/* Create a skip list whose level cap is derived from n via sl_max_level_for_n().
   level_probability is fixed at 0.5; the random-level generator uses a bit-test
   which is exactly p=0.5 per level. */
static SL *sl_new(int32_t n) {
    int max_lvl = sl_max_level_for_n(n);
    SL *s = calloc(1, sizeof *s);
    s->max_level = max_lvl;
    s->head = malloc(sizeof(SLNode) + (max_lvl + 1) * sizeof(SLNode *));
    s->head->val = INT32_MIN;
    s->head->level = max_lvl;
    s->head->payload = NULL;
    for (int i = 0; i <= max_lvl; i++) s->head->fwd[i] = NULL;
    s->level = 0;
    s->total_node_bytes = (int64_t)(sizeof(SLNode) + (max_lvl + 1) * sizeof(SLNode *));
    s->total_payload_bytes = 0;
    return s;
}

/* Random level generator for p = 0.5.
   Each bit of the xorshift output is an independent fair coin flip,
   so testing successive bits gives exactly geometric(0.5) levels.
   Capped at s->max_level (the runtime formula-derived cap). */
static int sl_rand_level_n(SL *s) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    int l = 0;
    uint64_t r = rng_state;
    while ((r & 1) && l < s->max_level) { r >>= 1; l++; }
    return l;
}

/* Standard skip-list insert: walk down from the top level recording
 * the last-visited node at each level into `upd[]` (the nodes whose
 * forward pointers will need to change), then splice a new node in at
 * a freshly-rolled random level. No-op if v is already present. Copies
 * `payload_size` bytes from `payload` into a fresh allocation owned by
 * the new node (or leaves payload NULL if payload_size is 0). */
static void sl_insert(SL *s, int32_t v,
                      const void *payload, size_t payload_size) {
    SLNode *upd[SL_HARD_MAX_LEVEL + 1];
    SLNode *cur = s->head;
    for (int i = s->level; i >= 0; i--) {
        while (cur->fwd[i] && cur->fwd[i]->val < v) cur = cur->fwd[i];
        upd[i] = cur;
    }
    if (cur->fwd[0] && cur->fwd[0]->val == v) return;
    int l = sl_rand_level_n(s);
    if (l > s->level) {
        for (int i = s->level + 1; i <= l; i++) upd[i] = s->head;
        s->level = l;
    }
    SLNode *nd = malloc(sizeof(SLNode) + (l + 1) * sizeof(SLNode *));
    nd->val = v;
    nd->level = l;
    if (payload && payload_size > 0) {
        nd->payload = malloc(payload_size);
        memcpy(nd->payload, payload, payload_size);
        s->total_payload_bytes += payload_size;
    } else {
        nd->payload = NULL;
    }
    for (int i = 0; i <= l; i++) {
        nd->fwd[i] = upd[i]->fwd[i];
        upd[i]->fwd[i] = nd;
    }
    s->count++;
    s->total_node_bytes += (int64_t)(sizeof(SLNode) + (l + 1) * sizeof(SLNode *));
}

/* Standard skip-list search: descend from the top level, advancing
 * forward at each level while the next node's value is still < v, then
 * drop a level; at level 0 the immediate next node is checked for an
 * exact match. Returns the matching SLNode, or NULL if not present. */
static SLNode *sl_search(SL *s, int32_t v) {
    SLNode *cur = s->head;
    for (int i = s->level; i >= 0; i--)
        while (cur->fwd[i] && cur->fwd[i]->val < v) cur = cur->fwd[i];
    cur = cur->fwd[0];
    return (cur && cur->val == v) ? cur : NULL;
}

/* Standard skip-list delete: same top-down walk as insert to populate
 * `upd[]`, then unlink the target node at every level it appears in
 * and shrink `s->level` while the top level(s) have become empty.
 * Returns false without modifying anything if v isn't present. */
static bool sl_delete(SL *s, int32_t v) {
    SLNode *upd[SL_HARD_MAX_LEVEL + 1];
    SLNode *cur = s->head;
    for (int i = s->level; i >= 0; i--) {
        while (cur->fwd[i] && cur->fwd[i]->val < v) cur = cur->fwd[i];
        upd[i] = cur;
    }
    SLNode *target = cur->fwd[0];
    if (!target || target->val != v) return false;
    for (int i = 0; i <= s->level; i++) {
        if (upd[i]->fwd[i] != target) break;
        upd[i]->fwd[i] = target->fwd[i];
    }
    while (s->level > 0 && !s->head->fwd[s->level]) s->level--;
    if (target->payload) {
        /* payload_size no longer tracked per-node (matches JLI's Node, which
           carries a raw payload pointer with no size field); total_payload_bytes
           is therefore no longer decremented precisely on delete. In this
           benchmark payload is always NULL, so this branch never executes. */
        free(target->payload);
    }
    s->total_node_bytes -= (int64_t)(sizeof(SLNode) + (target->level + 1) * sizeof(SLNode *));
    s->count--;
    free(target);
    return true;
}

/* Frees every node in the skip list (including any per-node payload
 * allocations) and the list header itself. */
static void sl_destroy(SL *s) {
    SLNode *cur = s->head;
    while (cur) {
        SLNode *nx = cur->fwd[0];
        if (cur->payload) free(cur->payload);
        free(cur);
        cur = nx;
    }
    free(s);
}

/* Total heap memory used by the skip list's structural nodes (header +
 * every SLNode, including their variable-length forward-pointer
 * arrays), NOT including payload bytes. Mirrors jli_memory_bytes() for
 * comparable benchmarking against JLI. */
static int64_t sl_memory_bytes(SL *s) {
    return (int64_t)sizeof(SL) + s->total_node_bytes;
}

/* Same as sl_memory_bytes(), but also includes payload bytes copied in
 * via sl_insert(). Mirrors jli_memory_bytes_with_payload(). */
static int64_t sl_mem_payload(SL *s) {
    return (int64_t)sizeof(SL) + s->total_node_bytes + s->total_payload_bytes;
}
