import sqlite3
conn = sqlite3.connect("locomo_eval_H.db")
c = conn.cursor()
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)
for t in ["facts", "facts_fts", "facts_derived_fts"]:
    if t in tables:
        n = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} rows")
    else:
        print(f"  {t}: MISSING")
if "facts" in tables:
    print("Fact types:")
    for row in c.execute("SELECT fact_type, count(*) FROM facts GROUP BY fact_type ORDER BY 2 DESC").fetchall():
        print(f"  {row[0]}: {row[1]}")
conn.close()
