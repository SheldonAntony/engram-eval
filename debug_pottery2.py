"""Debug why temporal resolution fails for specific question."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/.config/opencode'))
sys.path.insert(0, os.path.expanduser('~/.config/preflight'))

import sqlite3, struct, json
import eval_locomo as ev

db_path = r"C:\Users\Sheldon Antony\.config\preflight\locomo_eval_B.db"
samples = ev.download_dataset()

# Find the sample with pottery question
target_q = "When did Melanie sign up for a pottery class"
for sample in samples:
    for qa in sample.get('qa', []):
        if target_q.lower() in qa.get('question', '').lower() and qa.get('category') != 5:
            sid_str = str(sample.get('sample_id', 0))
            pid = f"locomo_{sid_str}"
            print(f"Found in sample {sid_str}, pid={pid}")
            print(f"Q: {qa['question']}")
            print(f"GT: {qa['answer']}")
            print(f"Category: {qa['category']}")
            
            # Build session dates
            session_dates_map = {}
            for sn, ds, _ in ev.iter_sessions(sample.get('conversation', {})):
                session_dates_map[sn] = ds
            print(f"\nSession dates: {list(session_dates_map.items())[:8]}")
            
            # Query facts for this project
            from utils import embed_text, cosine_similarity
            q_emb = embed_text(qa['question'])
            
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                """SELECT id, content, embedding, session_id FROM facts
                   WHERE project_id = ? AND superseded_at IS NULL
                   AND fact_type != 'turn'
                   AND (valid_to IS NULL OR valid_to > unixepoch())""",
                (pid,)
            ).fetchall()
            
            fact_cache = []
            session_id_by_fid = {}
            for fid, content, blob, s_id in rows:
                if s_id:
                    session_id_by_fid[fid] = s_id
                if blob is None:
                    continue
                if isinstance(blob, (bytes, bytearray)):
                    n = len(blob) // 4
                    emb = list(struct.unpack(f"{n}f", blob))
                else:
                    try:
                        emb = json.loads(blob)
                    except:
                        continue
                fact_cache.append((fid, content, emb))
            
            # RRF ranking
            _RRF_K = 60
            n_facts = len(fact_cache)
            cos_ranked = sorted(fact_cache, key=lambda x: cosine_similarity(q_emb, x[2]), reverse=True)
            cos_rank = {fid: i for i, (fid, _, _e) in enumerate(cos_ranked)}
            
            # BM25
            bm25_rank = {}
            all_fids_set = {fid for fid, _, _e in fact_cache}
            safe = "".join(c if c.isalnum() or c.isspace() else " " for c in qa['question'])
            tokens = [t for t in safe.split() if len(t) > 2]
            fts_q = " OR ".join(f'"{t}"' for t in tokens)
            bm_rows = conn.execute(
                "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                (fts_q,)
            ).fetchall()
            rank = 0
            for (bfid,) in bm_rows:
                if bfid in all_fids_set:
                    bm25_rank[bfid] = rank
                    rank += 1
            
            rrf = {}
            for fid, _, _e in fact_cache:
                s = 1.0 / (_RRF_K + cos_rank.get(fid, n_facts))
                if fid in bm25_rank:
                    s += 1.0 / (_RRF_K + bm25_rank[fid])
                rrf[fid] = s
            
            sorted_fids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:5]
            content_by_fid = {fid: content for fid, content, _e in fact_cache}
            
            print(f"\nTop 5 retrieved facts:")
            fact_dates = []
            for i, fid in enumerate(sorted_fids):
                s_id = session_id_by_fid.get(fid, "")
                try:
                    sess_num = int(s_id.split("_s")[-1]) if s_id else 0
                    date_str = session_dates_map.get(sess_num, "")
                except:
                    date_str = ""
                fact_dates.append(date_str)
                print(f"  #{i+1} fid={fid} session={s_id} date={date_str!r}")
                print(f"       content: {content_by_fid[fid][:100]!r}")
            
            # Try temporal resolution
            facts = [content_by_fid[fid] for fid in sorted_fids]
            print("\nTemporal resolution attempt:")
            for fi, fact in enumerate(facts):
                date_str = fact_dates[fi]
                session_dt = ev._parse_session_dt(date_str) if date_str else None
                print(f"  Fact #{fi+1}: session_dt={session_dt}")
                import re
                for line in fact.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    line = re.sub(r'^\[(prev|curr|next)\]\s*', '', line)
                    line = re.sub(r'^\w[\w\s]*:\s*', '', line)
                    if not line or line.endswith('?'):
                        continue
                    resolved = ev._resolve_relative_date(line, session_dt)
                    if resolved:
                        print(f"    RESOLVED: '{line[:60]}' -> '{resolved}'")
            
            conn.close()
            break
