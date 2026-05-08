path = r"C:\Users\Sheldon Antony\.config\preflight\test_mcp.py"
content = open(path, encoding="utf-8").read()

# Find the summary separator (the box-drawing line before "total = _PASS + _FAIL")
marker = "total = _PASS + _FAIL"
idx = content.rfind(marker)

new_tests = '''
# -- IMP7-1: memory_release soft-deletes a fact (RELEASE mutation written) --
print("\\n-- IMP7-1: memory_release soft-deletes a fact --")
IMP7_PROJ = "imp7_release_proj"
try:
    fid7 = _mem.store_fact(IMP7_PROJ, "imp7_s", "we use Redis for session caching", enrich=False)
    rel_result = _mem.memory_release(fid7, "imp7_s")
    _c7 = sqlite3.connect(_DB)
    sup_at = _c7.execute("SELECT superseded_at FROM facts WHERE id = ?", (fid7,)).fetchone()[0]
    mut_type = _c7.execute(
        "SELECT mutation_type FROM fact_mutations WHERE fact_id = ? AND mutation_type = 'RELEASE'",
        (fid7,)
    ).fetchone()
    _c7.close()
    check("IMP7-1: memory_release returns ok=True", rel_result.get("ok") is True,
          f"result: {rel_result}")
    check("IMP7-1: superseded_at is set", sup_at is not None, f"superseded_at={sup_at}")
    check("IMP7-1: RELEASE mutation written", mut_type is not None, f"mutation: {mut_type}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP7-2: released fact is invisible to retrieve_facts --
print("\\n-- IMP7-2: released fact is invisible to retrieval --")
try:
    facts_after = _mem.retrieve_facts(IMP7_PROJ, "imp7_s", "Redis session caching", top_n=5, threshold=0.0)
    redis_visible = any("Redis" in f for f in facts_after)
    check("IMP7-2: released fact not returned by retrieve_facts", not redis_visible,
          f"visible={redis_visible}, results={facts_after}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP7-3: memory_release on non-existent fact_id returns ok=False --
print("\\n-- IMP7-3: memory_release on bad fact_id returns error --")
try:
    bad_result = _mem.memory_release(99999999, "imp7_s")
    check("IMP7-3: bad fact_id returns ok=False", bad_result.get("ok") is False,
          f"result: {bad_result}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP7-4: memory_release on already-superseded fact returns ok=False --
print("\\n-- IMP7-4: double-release returns ok=False --")
try:
    double_result = _mem.memory_release(fid7, "imp7_s")
    check("IMP7-4: double-release returns ok=False", double_result.get("ok") is False,
          f"result: {double_result}")
except Exception:
    print("  ERROR:", traceback.format_exc())

'''

insert_pos = content.rfind("# ") 
# find the line with the box separator right before total = ...
sep_start = content.rfind("\\n# ", 0, idx)
content = content[:sep_start + 1] + new_tests + content[sep_start + 1:]
open(path, "w", encoding="utf-8").write(content)
print("Done, length:", len(content))
