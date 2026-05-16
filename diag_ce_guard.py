"""Cross-result diagnostic: compare GBM-only vs CE results to classify R@40 failures.

Loads:
  locomo_recall_v4_broad200_cov40_18feat.json       (GBM, no CE)
  locomo_recall_v4_broad200_cov40_18feat_ce.json    (GBM + pure CE)

Classifies per-question failures into:
  A) CE DEMOTION  — hit@40=True in GBM-only, hit@40=False in CE  (CE pushed gold out)
  B) POOL MISS    — hit@40=False in both  (gold not in pool regardless of CE)
  C) CE GAIN      — hit@40=False in GBM-only, hit@40=True in CE  (CE rescued it)
  D) BOTH HIT     — hit@40=True in both
"""
import json

GBM_JSON = "locomo_recall_v4_broad200_cov40_18feat.json"
CE_JSON  = "locomo_recall_v4_broad200_cov40_18feat_ce.json"

with open(GBM_JSON) as f:
    gbm = json.load(f)
with open(CE_JSON) as f:
    ce = json.load(f)

gbm_pq = {q["question"]: q for q in gbm["per_question"] if q.get("has_evidence")}
ce_pq  = {q["question"]: q for q in ce["per_question"]  if q.get("has_evidence")}
common = set(gbm_pq) & set(ce_pq)

cases = {"A_ce_demotion": [], "B_pool_miss": [], "C_ce_gain": [], "D_both_hit": []}
for qtext in common:
    g = gbm_pq[qtext]
    c = ce_pq[qtext]
    hit_gbm = g["hit@40"]
    hit_ce  = c["hit@40"]
    if hit_gbm and not hit_ce:
        cases["A_ce_demotion"].append((qtext, g, c))
    elif not hit_gbm and not hit_ce:
        cases["B_pool_miss"].append((qtext, g, c))
    elif not hit_gbm and hit_ce:
        cases["C_ce_gain"].append((qtext, g, c))
    else:
        cases["D_both_hit"].append((qtext, g, c))

print("=" * 60)
print("  CE COVERAGE DIAGNOSTIC  (R@40 failure analysis)")
print("=" * 60)
total = len(common)
for key, label in [
    ("A_ce_demotion", "CE DEMOTION (GBM=hit, CE=miss)"),
    ("B_pool_miss",   "POOL MISS   (both miss)"),
    ("C_ce_gain",     "CE GAIN     (GBM=miss, CE=hit)"),
    ("D_both_hit",    "BOTH HIT    (both hit)"),
]:
    n = len(cases[key])
    print(f"  {label}: {n:4d} / {total} ({n/total*100:.1f}%)")

print()
print(f"  Net CE R@40 change: +{len(cases['C_ce_gain'])} / -{len(cases['A_ce_demotion'])} "
      f"= {len(cases['C_ce_gain']) - len(cases['A_ce_demotion']):+d}")

# Detail the demotions (most actionable for CE guard)
if cases["A_ce_demotion"]:
    print()
    print(f"  CE DEMOTION details (gold rank after CE > 40, was ≤ 40 with GBM):")
    for qtext, g, c in sorted(cases["A_ce_demotion"],
                               key=lambda x: x[2].get("gold_rrf_rank_best", 9999)):
        rk_gbm = g.get("gold_rrf_rank_best", "?")
        rk_ce  = c.get("gold_rrf_rank_best", "?")
        cat    = c.get("category", "?")
        print(f"    [{cat:>11}] gbm_rank={rk_gbm:>5} -> ce_rank={rk_ce:>5}  {qtext[:65]}")

# Detail the pool misses (need wider BROAD_POOL)
if cases["B_pool_miss"]:
    print()
    n_pm = len(cases["B_pool_miss"])
    ranks = [c.get("gold_rrf_rank_best", 9999) for _, g, c in cases["B_pool_miss"]
             if c.get("gold_rrf_rank_best") is not None]
    ranks.sort()
    print(f"  POOL MISS details ({n_pm} questions, gold never reaches top-40):")
    cats = {}
    for _, g, c in cases["B_pool_miss"]:
        cats.setdefault(c.get("category", "?"), 0)
        cats[c.get("category", "?")] += 1
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")
    if ranks:
        mid = len(ranks) // 2
        print(f"    gold_rrf_rank_best: min={ranks[0]} median={ranks[mid]} max={ranks[-1]}")
        gt100 = sum(1 for r in ranks if r > 100)
        gt200 = sum(1 for r in ranks if r > 200)
        print(f"    rank>100: {gt100}, rank>200: {gt200} (out of reranker pool)")

print()
print("  RECOMMENDATION:")
if len(cases["A_ce_demotion"]) > 0:
    print(f"  -> CE guard (PREFLIGHT_CE_GUARD_K=40) will recover {len(cases['A_ce_demotion'])} CE demotions")
if len(cases["B_pool_miss"]) > 0:
    print(f"  -> Wider BROAD_POOL will help the {len(cases['B_pool_miss'])} pool misses")
    ranks_pm = [c.get("gold_rrf_rank_best", 9999) for _, g, c in cases["B_pool_miss"]
                if c.get("gold_rrf_rank_best") is not None]
    gt100 = sum(1 for r in ranks_pm if r > 100)
    if gt100 > 0:
        print(f"    ({gt100} pool misses have gold rank > 100 — these need BROAD_POOL expansion)")
