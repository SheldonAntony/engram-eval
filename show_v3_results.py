import json

v3  = json.load(open(r'C:\Users\Sheldon Antony\.config\preflight\locomo_recall_engram_v3.json'))
gdb = json.load(open(r'C:\Users\Sheldon Antony\.config\preflight\locomo_recall_base_st_control.json'))

print("=== H.db (engram_v3) vs G.db (base_st_control) ===")
nev = v3["questions_with_evidence"]
gnev = gdb["questions_with_evidence"]
print(f"n_ev  : {nev}  (G.db: {gnev})")

rk  = v3["recall_at_k"]
grk = gdb["recall_at_k"]
for k in sorted(rk.keys(), key=int):
    delta = rk[k] - grk[k]
    sign = "+" if delta >= 0 else ""
    print(f"R@{k:<3}: {rk[k]:.2f}  (G.db: {grk[k]:.2f})  delta={sign}{delta:.2f}")

print()
print("Category R@5:")
cat_v3  = v3["recall_at_5_by_category"]
cat_gdb = gdb["recall_at_5_by_category"]
for cat in ["single_hop", "multi_hop", "temporal", "open_domain"]:
    v = cat_v3.get(cat, float("nan"))
    g = cat_gdb.get(cat, float("nan"))
    delta = v - g
    sign = "+" if delta >= 0 else ""
    gate = "  <-- HARD GATE" if cat == "multi_hop" else ""
    print(f"  {cat:<12}: {v:.2f}  (G.db: {g:.2f})  delta={sign}{delta:.2f}{gate}")

print()
print("=== ACCEPTANCE VERDICT ===")
gates = [
    ("R@5 >= 75.10",        rk["5"]  >= 75.10,  rk["5"],  75.10),
    ("R@40 >= 93.96",       rk["40"] >= 93.96,  rk["40"], 93.96),
    ("multi_hop R@5 >= 61.92", cat_v3.get("multi_hop", 0) >= 61.92, cat_v3.get("multi_hop", 0), 61.92),
]
all_pass = True
for label, passed, got, target in gates:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {label}  (got {got:.2f})")

print()
if all_pass:
    print(">>> v3 ACCEPTED — beats G.db on all gates. Update baseline_frozen.json.")
else:
    print(">>> v3 REJECTED — at least one gate failed. Keep G.db as production.")
