import sys, sqlite3
sys.path.insert(0, r'C:\Users\Sheldon Antony\.config\opencode')
import memory as m
from utils import cosine_similarity, embed_text

conn = sqlite3.connect(m.DB_PATH)
rows = conn.execute(
    'SELECT id, content, fact_type FROM facts WHERE project_id=? ORDER BY id',
    ('imp17b_dissimilar_proj',)
).fetchall()
print("imp17b facts:")
for r in rows:
    print(" ", r)
conn.close()

a = "database connection pool max 20"
b = "the cat sat on the mat outside"
sim = cosine_similarity(embed_text(a), embed_text(b))
print(f"\nSim({a!r}, {b!r}) = {sim:.4f}")
