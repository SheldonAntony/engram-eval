import json

GBM = 'locomo_recall_v4_broad200_cov40_18feat.json'
CE5 = 'locomo_recall_v5_ce_guard40_pool200.json'

with open(GBM) as f: gbm = json.load(f)
with open(CE5) as f: ce5 = json.load(f)

gbm_pq = {q['question']: q for q in gbm['per_question'] if q.get('has_evidence')}
ce5_pq = {q['question']: q for q in ce5['per_question'] if q.get('has_evidence')}
common = set(gbm_pq) & set(ce5_pq)

A, B, C, D = [], [], [], []
for q in common:
    g, c = gbm_pq[q], ce5_pq[q]
    if     g['hit@40'] and not c['hit@40']: A.append((q, g, c))
    elif not g['hit@40'] and not c['hit@40']: B.append((q, g, c))
    elif not g['hit@40'] and     c['hit@40']: C.append((q, g, c))
    else: D.append((q, g, c))

print(f'CE DEMOTION (GBM=hit, v5=miss): {len(A)}')
print(f'POOL MISS   (both miss):         {len(B)}')
print(f'CE GAIN     (GBM=miss, v5=hit):  {len(C)}')
print(f'BOTH HIT:                        {len(D)}')
print(f'Net R@40 change: +{len(C)}/-{len(A)} = {len(C)-len(A):+d}')

if A:
    print()
    print('CE DEMOTIONS remaining:')
    for q, g, c in sorted(A, key=lambda x: x[2].get('gold_rrf_rank_best', 9999))[:12]:
        rg = g.get('gold_rrf_rank_best', '?')
        rc = c.get('gold_rrf_rank_best', '?')
        cat = c.get('category', '?')
        print(f'  gbm={rg!s:>4} v5={rc!s:>4}  [{cat:>11}] {q[:62]}')

if B:
    ranks = sorted(c.get('gold_rrf_rank_best', 9999) for _, g, c in B
                   if c.get('gold_rrf_rank_best') is not None)
    mid = len(ranks) // 2
    print()
    print(f'POOL MISSES: n={len(B)} ranks min={ranks[0]} med={ranks[mid]} max={ranks[-1]}')
    print(f'  rank>100: {sum(1 for r in ranks if r>100)}  rank>200: {sum(1 for r in ranks if r>200)}')
    cats = {}
    for _, g, c in B:
        cats[c['category']] = cats.get(c['category'], 0) + 1
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')
