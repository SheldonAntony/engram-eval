"""v5 vs v8 transition diagnostic.

Compares per-question hit@3 and hit@40 between v5 (xsmall CE) and v8 (bge-reranker-v2-m3).
Reports transition counts, category breakdowns, gold rank movements, and sample questions
for each failure/gain bucket so we understand WHAT bge-v2-m3 fixed and what it broke.

Run from C:\\Users\\Sheldon Antony\\.config\\preflight:
    python diag_v8.py
"""
import json
from collections import defaultdict

V5 = 'locomo_recall_v5_ce_guard40_pool200.json'
V8 = 'locomo_recall_v8_bge_reranker_v2m3.json'

with open(V5) as f: v5 = json.load(f)
with open(V8) as f: v8 = json.load(f)

v5_pq = {q['question']: q for q in v5['per_question'] if q.get('has_evidence')}
v8_pq = {q['question']: q for q in v8['per_question'] if q.get('has_evidence')}
common = set(v5_pq) & set(v8_pq)

CATS = ['single_hop', 'multi_hop', 'temporal', 'open_domain']

def _rank(q): return q.get('gold_rrf_rank_best') or 9999

# ── Hit@3 transitions ──────────────────────────────────────────────────────────
# GAIN:  v5 miss @3, v8 hit @3  → bge-v2-m3 promoted gold into top-3
# LOSS:  v5 hit  @3, v8 miss @3 → bge-v2-m3 demoted gold out of top-3
# BOTH_HIT:  both in top-3
# BOTH_MISS: both miss top-3 (gold in pool 4-40 or not in pool)

gain3, loss3, both_hit3, both_miss3 = [], [], [], []
for q in common:
    a, b = v5_pq[q], v8_pq[q]
    h5, h8 = a.get('hit@3', False), b.get('hit@3', False)
    if not h5 and h8:      gain3.append((q, a, b))
    elif h5 and not h8:    loss3.append((q, a, b))
    elif h5 and h8:        both_hit3.append((q, a, b))
    else:                  both_miss3.append((q, a, b))

# ── Hit@40 transitions ─────────────────────────────────────────────────────────
gain40, loss40, both_hit40, both_miss40 = [], [], [], []
for q in common:
    a, b = v5_pq[q], v8_pq[q]
    h5, h8 = a.get('hit@40', False), b.get('hit@40', False)
    if not h5 and h8:      gain40.append((q, a, b))
    elif h5 and not h8:    loss40.append((q, a, b))
    elif h5 and h8:        both_hit40.append((q, a, b))
    else:                  both_miss40.append((q, a, b))

def cat_counts(bucket):
    counts = defaultdict(int)
    for _, _, b in bucket:
        counts[b.get('category', '?')] += 1
    return counts

def rank_stats(bucket, src='v8'):
    """Return (min, median, max) of gold_rrf_rank_best from v8 (or v5) perspective."""
    ranks = sorted(_rank(b if src == 'v8' else a) for _, a, b in bucket)
    if not ranks: return None, None, None
    mid = len(ranks) // 2
    return ranks[0], ranks[mid], ranks[-1]

def show_samples(bucket, label, n=15, src_key='v8'):
    if not bucket: return
    sorted_b = sorted(bucket, key=lambda x: _rank(x[2] if src_key == 'v8' else x[1]))
    print(f'\n  {label} (top {min(n, len(sorted_b))} by gold rank in {src_key}):')
    for q, a, b in sorted_b[:n]:
        r5 = _rank(a)
        r8 = _rank(b)
        cat = b.get('category', '?')
        print(f'    v5={r5:>4} v8={r8:>4}  [{cat:>11}]  {q[:68]}')

def rank_movement_histogram(bucket, bins=None):
    """Show how much rank changed between v5 and v8 for a bucket."""
    if bins is None:
        bins = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 100), (101, 9999)]
    counts_v5  = defaultdict(int)
    counts_v8  = defaultdict(int)
    for _, a, b in bucket:
        r5 = _rank(a)
        r8 = _rank(b)
        for lo, hi in bins:
            if lo <= r5 <= hi: counts_v5[(lo, hi)] += 1
            if lo <= r8 <= hi: counts_v8[(lo, hi)] += 1
    return counts_v5, counts_v8, bins


# ── Headline numbers ───────────────────────────────────────────────────────────
TOTAL = len(common)
print('=' * 70)
print(f'  V5 vs V8 TRANSITION DIAGNOSTIC   (n={TOTAL} common questions)')
print('=' * 70)
print(f'\n  Summary metrics:')
print(f'    v5  R@1={v5["recall_at_k"]["1"]:.2f}  R@3={v5["recall_at_k"]["3"]:.2f}  '
      f'R@5={v5["recall_at_k"]["5"]:.2f}  R@40={v5["recall_at_k"]["40"]:.2f}')
print(f'    v8  R@1={v8["recall_at_k"]["1"]:.2f}  R@3={v8["recall_at_k"]["3"]:.2f}  '
      f'R@5={v8["recall_at_k"]["5"]:.2f}  R@40={v8["recall_at_k"]["40"]:.2f}')
delta3  = v8["recall_at_k"]["3"]  - v5["recall_at_k"]["3"]
delta40 = v8["recall_at_k"]["40"] - v5["recall_at_k"]["40"]
print(f'    ΔR@3={delta3:+.2f}  ΔR@40={delta40:+.2f}')

# ── Hit@3 summary ──────────────────────────────────────────────────────────────
print('\n' + '─' * 70)
print('  HIT@3 TRANSITIONS')
print('─' * 70)
print(f'  GAIN  (v5 miss → v8 hit):   {len(gain3):>4}  (+{len(gain3)/TOTAL*100:.2f}pp)')
print(f'  LOSS  (v5 hit  → v8 miss):  {len(loss3):>4}  (-{len(loss3)/TOTAL*100:.2f}pp)')
print(f'  BOTH HIT:                   {len(both_hit3):>4}')
print(f'  BOTH MISS:                  {len(both_miss3):>4}')
print(f'  Net: +{len(gain3)}/-{len(loss3)} = {len(gain3)-len(loss3):+d}')

print('\n  GAIN by category:')
for cat, cnt in sorted(cat_counts(gain3).items(), key=lambda x: -x[1]):
    print(f'    {cat:>12}: {cnt:>3}')

print('\n  LOSS by category:')
for cat, cnt in sorted(cat_counts(loss3).items(), key=lambda x: -x[1]):
    print(f'    {cat:>12}: {cnt:>3}')

print('\n  BOTH MISS by category:')
for cat, cnt in sorted(cat_counts(both_miss3).items(), key=lambda x: -x[1]):
    print(f'    {cat:>12}: {cnt:>3}')

# Where does gold sit in both-miss cases?
if both_miss3:
    rmin, rmed, rmax = rank_stats(both_miss3)
    print(f'\n  BOTH MISS gold ranks (v8): min={rmin} med={rmed} max={rmax}')
    in40   = sum(1 for _, _, b in both_miss3 if _rank(b) <= 40)
    in41_100 = sum(1 for _, _, b in both_miss3 if 41 <= _rank(b) <= 100)
    gt100  = sum(1 for _, _, b in both_miss3 if _rank(b) > 100)
    print(f'    in pool (≤40): {in40}  rank 41-100: {in41_100}  rank>100: {gt100}')

# Rank movement for GAIN bucket (where did gold come from?)
if gain3:
    print(f'\n  GAIN: where was gold in v5 before v8 promoted it to top-3?')
    rank_buckets = [(1,3),(4,5),(6,10),(11,20),(21,40),(41,100),(101,9999)]
    counts = defaultdict(int)
    for _, a, _ in gain3:
        r = _rank(a)
        for lo, hi in rank_buckets:
            if lo <= r <= hi: counts[(lo,hi)] += 1; break
    for lo, hi in rank_buckets:
        label = f'{lo}-{hi}' if hi < 9999 else f'>{lo-1}'
        print(f'    rank {label:>8}: {counts[(lo,hi)]:>3}')

show_samples(gain3,  'GAIN samples', n=20, src_key='v5')
show_samples(loss3,  'LOSS samples', n=20, src_key='v8')

# ── Hit@40 summary ─────────────────────────────────────────────────────────────
print('\n' + '─' * 70)
print('  HIT@40 TRANSITIONS (pool ceiling)')
print('─' * 70)
print(f'  GAIN  (v5 miss → v8 hit):   {len(gain40):>4}')
print(f'  LOSS  (v5 hit  → v8 miss):  {len(loss40):>4}')
print(f'  BOTH HIT:                   {len(both_hit40):>4}')
print(f'  BOTH MISS:                  {len(both_miss40):>4}')
print(f'  Net: +{len(gain40)}/-{len(loss40)} = {len(gain40)-len(loss40):+d}')

print('\n  BOTH MISS by category (true pool ceiling):')
for cat, cnt in sorted(cat_counts(both_miss40).items(), key=lambda x: -x[1]):
    print(f'    {cat:>12}: {cnt:>3}')

if both_miss40:
    rmin, rmed, rmax = rank_stats(both_miss40)
    print(f'\n  BOTH MISS gold ranks (v8): min={rmin} med={rmed} max={rmax}')
    gt100 = sum(1 for _, _, b in both_miss40 if _rank(b) > 100)
    gt200 = sum(1 for _, _, b in both_miss40 if _rank(b) > 200)
    print(f'    rank>100: {gt100}  rank>200: {gt200}')

show_samples(both_miss40, 'BOTH MISS pool samples', n=20, src_key='v8')

# ── Per-category hit@3 detailed breakdown ─────────────────────────────────────
print('\n' + '─' * 70)
print('  PER-CATEGORY HIT@3 BREAKDOWN')
print('─' * 70)
for cat in CATS:
    cat_q = [q for q in common if v5_pq[q].get('category') == cat]
    n = len(cat_q)
    if not n: continue
    v5_h3 = sum(1 for q in cat_q if v5_pq[q].get('hit@3'))
    v8_h3 = sum(1 for q in cat_q if v8_pq[q].get('hit@3'))
    v5_h5 = sum(1 for q in cat_q if v5_pq[q].get('hit@5'))
    v8_h5 = sum(1 for q in cat_q if v8_pq[q].get('hit@5'))
    g = sum(1 for q in cat_q if not v5_pq[q].get('hit@3') and v8_pq[q].get('hit@3'))
    l = sum(1 for q in cat_q if v5_pq[q].get('hit@3') and not v8_pq[q].get('hit@3'))
    print(f'\n  {cat} (n={n}):')
    print(f'    R@3  v5={v5_h3/n*100:.1f}%  v8={v8_h3/n*100:.1f}%  Δ={( v8_h3-v5_h3)/n*100:+.1f}pp')
    print(f'    R@5  v5={v5_h5/n*100:.1f}%  v8={v8_h5/n*100:.1f}%  Δ={(v8_h5-v5_h5)/n*100:+.1f}pp')
    print(f'    GAIN={g}  LOSS={l}  net={g-l:+d}')

# ── Single-hop regression deep-dive ───────────────────────────────────────────
print('\n' + '─' * 70)
print('  SINGLE-HOP REGRESSION DEEP-DIVE')
print('─' * 70)
sh_loss = [(q, v5_pq[q], v8_pq[q]) for q in common
           if v5_pq[q].get('category') == 'single_hop'
           and v5_pq[q].get('hit@3') and not v8_pq[q].get('hit@3')]
sh_gain = [(q, v5_pq[q], v8_pq[q]) for q in common
           if v5_pq[q].get('category') == 'single_hop'
           and not v5_pq[q].get('hit@3') and v8_pq[q].get('hit@3')]

print(f'\n  Single-hop LOSS (v8 demoted out of @3): {len(sh_loss)}')
for q, a, b in sorted(sh_loss, key=lambda x: _rank(x[2])):
    r5, r8 = _rank(a), _rank(b)
    h5_5 = a.get('hit@5', False)
    h8_5 = b.get('hit@5', False)
    print(f'    v5_rank={r5:>4} v8_rank={r8:>4}  v5@5={"Y" if h5_5 else "N"} v8@5={"Y" if h8_5 else "N"}  {q[:70]}')

print(f'\n  Single-hop GAIN (v8 promoted into @3):  {len(sh_gain)}')
for q, a, b in sorted(sh_gain, key=lambda x: _rank(x[1])):
    r5, r8 = _rank(a), _rank(b)
    print(f'    v5_rank={r5:>4} v8_rank={r8:>4}  {q[:70]}')

# ── Both-miss @3 — rank distribution in v8 (hardest cases) ───────────────────
print('\n' + '─' * 70)
print('  BOTH-MISS@3 RANK DISTRIBUTION IN V8 (hardest to fix)')
print('─' * 70)
buckets = [(1,3),(4,5),(6,10),(11,20),(21,40),(41,100),(101,9999)]
counts_v5 = defaultdict(int)
counts_v8 = defaultdict(int)
for _, a, b in both_miss3:
    r5, r8 = _rank(a), _rank(b)
    for lo, hi in buckets:
        if lo <= r5 <= hi: counts_v5[(lo,hi)] += 1; break
    for lo, hi in buckets:
        if lo <= r8 <= hi: counts_v8[(lo,hi)] += 1; break
print(f'  {"rank":>12}  {"v5":>5}  {"v8":>5}  (smaller bucket = easier to fix)')
for lo, hi in buckets:
    label = f'{lo}-{hi}' if hi < 9999 else f'>{lo-1}'
    print(f'  {label:>12}  {counts_v5[(lo,hi)]:>5}  {counts_v8[(lo,hi)]:>5}')

print('\n' + '=' * 70)
print('  Done.')
print('=' * 70)
