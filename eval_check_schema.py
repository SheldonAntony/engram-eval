import sqlite3
db = sqlite3.connect(r"C:\Users\Sheldon Antony\.config\preflight\locomo_eval_B.db")
# schema
schema = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'").fetchone()
print(schema[0][:500])
print()
# sample row
try:
    row = db.execute("SELECT id, project_id, session_id, fact_type, content FROM facts WHERE project_id LIKE 'locomo_%' LIMIT 1").fetchone()
    for col, val in zip(['id','project_id','session_id','fact_type','content'], row):
        print(col + ':', repr(val)[:120])
except Exception as e:
    print("Error:", e)
    # List columns
    cols = db.execute("PRAGMA table_info(facts)").fetchall()
    print("Columns:", [c[1] for c in cols])
db.close()
