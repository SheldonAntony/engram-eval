path = r"C:\Users\Sheldon Antony\.config\preflight\test_mcp.py"
content = open(path, encoding="utf-8").read()
target = '_mem.store_fact(GRAPH_PROJ, "sess_g1", "we use SQLAlchemy as our ORM layer", "decision")'
if target in content:
    content = content.replace(
        '_mem.store_fact(GRAPH_PROJ, "sess_g1", "we use SQLAlchemy as our ORM layer", "decision")',
        '_mem.store_fact(GRAPH_PROJ, "sess_g1", "we use SQLAlchemy as our ORM layer", "decision", enrich=False)'
    ).replace(
        '_mem.store_fact(GRAPH_PROJ, "sess_g1", "N+1 query bug was caused by missing eager loading in SQLAlchemy", "finding")',
        '_mem.store_fact(GRAPH_PROJ, "sess_g1", "N+1 query bug was caused by missing eager loading in SQLAlchemy", "finding", enrich=False)'
    )
    open(path, "w", encoding="utf-8").write(content)
    print("Done")
else:
    print("NOT FOUND")
