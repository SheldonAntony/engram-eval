import json

with open('locomo_recall_v4_broad200_cov40_18feat.json') as f:
    d = json.load(f)

pq = [q for q in d['per_question'] if q.get('has_evidence')]

for cat in ['single_hop', 'temporal', 'multi_hop', 'open_domain']:
    qs = [q for q in pq if q['category'] == cat]
    fail3 = [q for q in qs if not q['hit@3']]
    fail40 = [q for q in qs if not q['hit@40']]
    ranks = sorted(q['gold_rrf_rank_best'] for q in qs if q.get('gold_rrf_rank_best') is not None)
    if not ranks:
        continue
    mid = len(ranks) // 2
    b = [0] * 5
    for r in ranks:
        if r <= 3:     b[0] += 1
        elif r <= 5:   b[1] += 1
        elif r <= 10:  b[2] += 1
        elif r <= 40:  b[3] += 1
        else:           b[4] += 1
    print(f"{cat}: n={len(qs)} fail@3={len(fail3)} fail@40={len(fail40)} rank_med={ranks[mid]}")
    print(f"  top3={b[0]} top5={b[0]+b[1]} top10={sum(b[:3])} top40={sum(b[:4])} gt40={b[4]}")
    print("  fail@3 examples (sorted by rank):")
    for q in sorted(fail3, key=lambda x: x['gold_rrf_rank_best'])[:4]:
        rk = q['gold_rrf_rank_best']
        print(f"    rk={rk:4d}  {q['question'][:72]}")
    print()
