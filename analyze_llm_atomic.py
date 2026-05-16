import sqlite3
conn = sqlite3.connect('locomo_eval_I.db')
c = conn.cursor()

print('=== Fact type counts ===')
c.execute('SELECT fact_type, COUNT(*) FROM facts GROUP BY fact_type')
for row in c.fetchall():
    print(f'  {row[0]}: {row[1]}')

print('\n=== Avg atomic facts per source turn ===')
c.execute('SELECT AVG(cnt) FROM (SELECT source_hash, COUNT(*) as cnt FROM facts WHERE fact_type=? GROUP BY source_hash)', ('llm_atomic',))
print(f'  {c.fetchone()[0]:.1f}')

print('\n=== llm_atomic with matched turn source_hash ===')
c.execute('SELECT COUNT(*) FROM facts f WHERE f.fact_type=? AND EXISTS (SELECT 1 FROM facts t WHERE t.fact_type=? AND t.source_hash=f.source_hash)', ('llm_atomic', 'turn'))
matched = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM facts WHERE fact_type=?', ('llm_atomic',))
total_atomic = c.fetchone()[0]
print(f'  {matched} / {total_atomic} ({100*matched/total_atomic:.1f}%)')

print('\n=== Sample llm_atomic facts ===')
c.execute('SELECT text FROM facts WHERE fact_type=? LIMIT 8', ('llm_atomic',))
for row in c.fetchall():
    print(f'  {row[0][:130]}')

print('\n=== llm_atomic importance/decay ===')
c.execute('SELECT importance, decay FROM facts WHERE fact_type=? LIMIT 1', ('llm_atomic',))
row = c.fetchone()
print(f'  importance={row[0]}, decay={row[1]}')

print('\n=== turn fact importance/decay ===')
c.execute('SELECT importance, decay FROM facts WHERE fact_type=? LIMIT 1', ('turn',))
row = c.fetchone()
print(f'  importance={row[0]}, decay={row[1]}')

conn.close()
