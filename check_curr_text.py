import sqlite3
conn = sqlite3.connect('locomo_eval_H.db')
sql = "SELECT content FROM facts WHERE project_id='locomo_conv-26' AND superseded_at IS NULL AND fact_type NOT IN ('turn','llm_atomic') LIMIT 5"
rows = conn.execute(sql).fetchall()
for (c,) in rows:
    curr = next((ln[7:] for ln in c.split('\n') if ln.startswith('[curr] ')), None)
    print('FULL:', repr(c[:100]))
    print('CURR:', repr(curr[:80]) if curr else 'NONE')
    print()
