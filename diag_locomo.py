#!/usr/bin/env python3
"""Pipeline stage diagnostic for LoCoMo benchmark.

Uses the existing locomo_eval_B.db (no re-ingestion).
For each QA pair with a known gold fact, calls retrieve_facts()
with _gold_fid to log where the gold fact sits at each pipeline stage.

Outputs a CSV + summary table to stdout.

Usage:
    python3.13 diag_locomo.py
"""
from __future__ import annotations
import json, sqlite3, struct, sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "opencode"))
import memory as _mem

# ── Config ────────────────────────────────────────────────────────────────────
DATASET   = Path(__file__).parent / "locomo10.json"
DB_B      = Path(__file__).parent / "locomo_eval_B.db"
OUT_CSV   = Path(__file__).parent / "diag_stages.csv"
MAX_QA    = 300   # cap for speed — set to 9999 for full run

# ── Helpers from eval_locomo.py ───────────────────────────────────────────────
def iter_sessions(conv: dict):
    for sn_str, session in conv.items():
        try:
            sn = int(sn_str)
        except ValueError:
            continue
        turns = session.get("dialogue", [])
        date  = session.get("date", "")
        yield sn, date, turns

def build_dia_id_map(samples: list) -> dict:
    """Map project_id → {dia_id_str → fact_id} using [curr] line matching."""
    conn = sqlite3.connect(str(DB_B))
    result: dict[str, dict[str, int]] = {}
    for ci, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", ci))
        pid     = f"locomo_{sid_str}"
        rows    = conn.execute(
            "SELECT id, content FROM facts WHERE project_id = ? AND superseded_at IS NULL",
            (pid,),
        ).fetchall()
        content_to_id: dict[str, int] = {}
        for fid, content in rows:
            for line in content.split("\n"):
                if line.startswith("[curr] "):
                    content_to_id[line[len("[curr] "):]] = fid
                    break
            else:
                content_to_id[content] = fid
        pid_map: dict[str, int] = {}
        conv = sample.get("conversation", {})
        for sn, _date, turns in iter_sessions(conv):
            for turn in turns:
                dia_id  = turn.get("dia_id")
                speaker = str(turn.get("speaker", "?"))
                text    = str(turn.get("text", ""))
                if not text.strip() or dia_id is None:
                    continue
                content = f"{speaker}: {text}"
                fid = content_to_id.get(content)
                if fid is not None:
                    pid_map[str(dia_id)] = fid
        result[pid] = pid_map
    conn.close()
    return result


def main():
    if not DB_B.exists():
        print(f"ERROR: {DB_B} not found. Run eval_locomo.py first (Mode B).", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset...", flush=True)
    samples = json.loads(DATASET.read_text(encoding="utf-8"))
    print(f"  {len(samples)} conversations")

    print("Building dia_id map...", flush=True)
    dia_id_map = build_dia_id_map(samples)
    n_mapped = sum(len(v) for v in dia_id_map.values())
    print(f"  {n_mapped} turns mapped to fact IDs")

    # Counters per stage
    total_qa   = 0
    has_gold   = 0
    stages = {
        "pool":   {"hit": 0, "miss": 0, "positions": []},
        "scored": {"hit": 0, "miss": 0, "positions": []},
        "gated":  {"hit": 0, "miss": 0, "positions": []},
        "mmr":    {"hit": 0, "miss": 0, "in_top20": 0, "positions": []},
        "ce":     {"hit": 0, "miss": 0, "positions": []},
        "final":  {"hit": 0, "miss": 0},
    }

    csv_rows: list[str] = ["qid,pid,dia_id,gold_fid,pool_pos,pool_size,scored_pos,scored_size,"
                            "gated_pos,gated_size,mmr_pos,in_mmr_top20,ce_pos,final_pos"]

    print(f"\nRunning diagnostic (cap={MAX_QA} QA pairs)...", flush=True)
    t0 = time.time()

    for ci, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", ci))
        pid     = f"locomo_{sid_str}"
        pid_map = dia_id_map.get(pid, {})
        qas     = sample.get("qa", [])

        for qa in qas:
            cat = qa.get("category", 0)
            if cat == 5:
                continue   # adversarial
            total_qa += 1
            if total_qa > MAX_QA:
                break

            question  = str(qa.get("question", ""))
            evidence  = qa.get("evidence", [])
            gold_fids = [pid_map[str(e)] for e in evidence if str(e) in pid_map]

            if not gold_fids:
                continue
            has_gold += 1
            gold_fid = gold_fids[0]   # use first evidence turn

            try:
                result = _mem.retrieve_facts(
                    pid, f"{pid}_diag", question,
                    top_n=10, threshold=0.0,
                    include_budget_info=True,
                    _gold_fid=gold_fid,
                )
            except Exception as e:
                print(f"  ERROR on pid={pid} gold={gold_fid}: {e}", file=sys.stderr)
                continue

            st = result.get("_stages", {})

            pool_pos    = st.get("pool_pos", -1)
            pool_size   = st.get("pool_size", 0)
            scored_pos  = st.get("scored_pos", -1)
            scored_size = st.get("scored_size", 0)
            gated_pos   = st.get("gated_pos", -1)
            gated_size  = st.get("gated_size", 0)
            mmr_pos     = st.get("mmr_pos", -1)
            in_top20    = 1 if st.get("in_mmr_top20") else 0
            ce_pos      = st.get("ce_pos", -1)
            final_pos   = st.get("final_pos", -1)

            # Accumulate stage stats
            for key, pos, size_key, size in [
                ("pool",   pool_pos,   "pool_size",   pool_size),
                ("scored", scored_pos, "scored_size", scored_size),
                ("gated",  gated_pos,  "gated_size",  gated_size),
            ]:
                if pos >= 0:
                    stages[key]["hit"] += 1
                    stages[key]["positions"].append(pos)
                else:
                    stages[key]["miss"] += 1

            if mmr_pos >= 0:
                stages["mmr"]["hit"] += 1
                stages["mmr"]["positions"].append(mmr_pos)
                stages["mmr"]["in_top20"] += in_top20
            else:
                stages["mmr"]["miss"] += 1

            if ce_pos >= 0:
                stages["ce"]["hit"] += 1
                stages["ce"]["positions"].append(ce_pos)
            else:
                stages["ce"]["miss"] += 1

            if final_pos >= 0:
                stages["final"]["hit"] += 1
            else:
                stages["final"]["miss"] += 1

            csv_rows.append(
                f"{total_qa},{pid},{evidence[0] if evidence else '?'},"
                f"{gold_fid},{pool_pos},{pool_size},{scored_pos},{scored_size},"
                f"{gated_pos},{gated_size},{mmr_pos},{in_top20},{ce_pos},{final_pos}"
            )

        if total_qa > MAX_QA:
            break

    elapsed = time.time() - t0
    OUT_CSV.write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"  Done in {elapsed:.1f}s — {has_gold} QA pairs with gold evidence\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    def pct(n, d): return f"{100*n/d:.1f}%" if d else "N/A"
    def avg(lst):  return f"{sum(lst)/len(lst):.1f}" if lst else "N/A"
    def p50(lst):
        if not lst: return "N/A"
        s = sorted(lst); return str(s[len(s)//2])

    print("=" * 65)
    print("  PIPELINE STAGE DIAGNOSTIC SUMMARY")
    print(f"  QA pairs evaluated : {has_gold} (of {total_qa} non-adversarial)")
    print("=" * 65)
    print(f"{'Stage':<20} {'Hit%':>6} {'Miss%':>6} {'Avg pos':>8} {'P50 pos':>8}  Interpretation")
    print("-" * 65)

    for label, key, note in [
        ("Pool (500-cap)",   "pool",   "gold fact not in A+B pool → raise limits"),
        ("Post-scoring",     "scored", "gold fact dropped by score filter"),
        ("Post-SM-2 gate",   "gated",  "gold gated by SM-2 interval"),
        ("Post-MMR",         "mmr",    "gold survives MMR or pushed to remaining"),
        ("Post-cross-enc",   "ce",     "gold after cross-encoder rerank"),
    ]:
        d = stages[key]
        h, m = d["hit"], d["miss"]
        total = h + m
        print(f"  {label:<18} {pct(h,total):>6} {pct(m,total):>6} {avg(d['positions']):>8} {p50(d['positions']):>8}  {note if m/(total or 1) > 0.1 else ''}")

    mmr_in = stages["mmr"]["in_top20"]
    mmr_hit = stages["mmr"]["hit"]
    print(f"\n  In MMR top-20 : {pct(mmr_in, mmr_hit)} of gold facts that survived MMR")
    print(f"  Final Recall  : {pct(stages['final']['hit'], has_gold)}")

    print("\n  KEY QUESTIONS:")
    pool_miss_pct = stages["pool"]["miss"] / max(has_gold, 1)
    mmr_demotion  = (stages["mmr"]["hit"] - stages["mmr"]["in_top20"]) / max(stages["mmr"]["hit"], 1)
    avg_pool_pos  = sum(stages["pool"]["positions"]) / max(len(stages["pool"]["positions"]), 1)

    if pool_miss_pct > 0.1:
        print(f"  ⚠ {pool_miss_pct:.0%} of gold facts missing from pool → raise _POOL_A_LIMIT or _POOL_B_LIMIT")
    else:
        print(f"  ✓ Pool coverage good ({pool_miss_pct:.0%} miss rate)")

    if avg_pool_pos > 150:
        print(f"  ⚠ Avg gold position in pool = {avg_pool_pos:.0f} → scoring weights may need tuning")
    else:
        print(f"  ✓ Avg gold pool position = {avg_pool_pos:.0f}")

    if mmr_demotion > 0.2:
        print(f"  ⚠ MMR demotes {mmr_demotion:.0%} of gold facts out of top-20 → lower _MMR_LAMBDA (current=0.6)")
    else:
        print(f"  ✓ MMR demotion rate acceptable ({mmr_demotion:.0%})")

    print(f"\n  Full CSV saved → {OUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    # Override DB_PATH so init_db() opens the eval DB instead of production.
    import memory as _mem_mod
    _mem_mod.DB_PATH = str(DB_B)
    main()
