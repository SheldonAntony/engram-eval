import sqlite3
db = sqlite3.connect(r"C:\Users\Sheldon Antony\.config\preflight\locomo_eval_B.db")

rows = db.execute(
    "SELECT id, session_id, content FROM facts "
    "WHERE content LIKE '%pottery%' AND content LIKE '%[curr]%' AND content LIKE '%yesterday%'"
).fetchall()
print("Pottery + curr + yesterday:")
for r in rows:
    print(f"  id={r[0]} session={r[1]}")
    print(f"  content: {r[2][:200]!r}")
    print()

rows2 = db.execute(
    "SELECT id, session_id, content FROM facts "
    "WHERE session_id LIKE '%_s5' AND content LIKE '%pottery%' LIMIT 5"
).fetchall()
print("Session 5 pottery facts:")
for r in rows2:
    print(f"  id={r[0]} session={r[1]}")
    print(f"  content: {r[2][:150]!r}")

# Also: check what session Melanie's pottery signup turn belongs to
rows3 = db.execute(
    "SELECT id, session_id, content FROM facts "
    "WHERE content LIKE '%signed up for a pottery%' LIMIT 5"
).fetchall()
print()
print("'signed up for pottery' facts:")
for r in rows3:
    print(f"  id={r[0]} session={r[1]}")
    print(f"  content: {r[2][:150]!r}")

db.close()
