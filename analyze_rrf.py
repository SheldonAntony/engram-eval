import json, collections, sys

fname = sys.argv[1] if len(sys.argv) > 1 else 'locomo_recall_sw0_bm0p75_ce0_sp0.json'
r = json.load(open(fname))
qs = [q for q in r['per_question'] if q['has_evidence'] and not q['hit@40']]

def bucket(c):
    if c <= 40: return '01-40'
    if c <= 60: return '41-60'
    if c <= 100: return '61-100'
    if c <= 300: return '101-300'
    return 'gt300'

print(f"File: {fname}")
print(f"Total failed@40: {len(qs)}")
print("By cos-rank bucket:", dict(collections.Counter(bucket(q['gold_cos_rank_best']) for q in qs)))

rrf_lost = [q for q in qs if q['gold_cos_rank_best'] <= 40]
cos_hard  = [q for q in qs if q['gold_cos_rank_best'] > 40]
print(f"RRF-lost (cos<=40, RRF demotes): {len(rrf_lost)}")
print(f"Cosine-hard (cos>40): {len(cos_hard)}")
print(f"  By category: {dict(collections.Counter(q['category'] for q in cos_hard))}")
print()
print("RRF-lost sample (sorted by RRF rank):")
for q in sorted(rrf_lost, key=lambda x: x['gold_rrf_rank_best'])[:15]:
    cos = q['gold_cos_rank_best']
    rrf = q['gold_rrf_rank_best']
    cat = q['category']
    qtext = q['question'][:65]
    print(f"  cos={cos:2d} rrf={rrf:4d}  {cat:<12}  {qtext}")
