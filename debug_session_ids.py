import sqlite3, re, sys, os
sys.path.insert(0, os.path.expanduser('~/.config/preflight'))

db = sqlite3.connect(r"C:\Users\Sheldon Antony\.config\preflight\locomo_eval_B.db")

# Check window facts (with [curr] tag) - do they have session_id?
rows = db.execute("""
    SELECT id, session_id, fact_type, content FROM facts 
    WHERE content LIKE '%[curr]%'
    LIMIT 5
""").fetchall()
print("Window fact sample (with [curr]):")
for row in rows:
    print(f"  id={row[0]} session_id={row[1]!r} type={row[2]} content={row[3][:80]!r}")

print()
# Count window facts with NULL session_id
null_count = db.execute("""
    SELECT COUNT(*) FROM facts 
    WHERE content LIKE '%[curr]%' AND session_id IS NULL
""").fetchone()[0]
total_count = db.execute("""
    SELECT COUNT(*) FROM facts WHERE content LIKE '%[curr]%'
""").fetchone()[0]
print(f"Window facts with NULL session_id: {null_count} / {total_count}")

# Also check for "yesterday" in window facts
yest_rows = db.execute("""
    SELECT id, session_id, content FROM facts
    WHERE content LIKE '%yesterday%' AND content LIKE '%[curr]%'
    LIMIT 5
""").fetchall()
print()
print("Window facts with 'yesterday':")
for row in yest_rows:
    print(f"  id={row[0]} session_id={row[1]!r}")
    print(f"  content: {row[2][:120]!r}")

db.close()
