"""ORIGIN-T HTTP API (HANDBOOK §7.7): read-only routes over reports/**.

    python -m dashboard.api              # serve on http://127.0.0.1:8000 (docs at /docs)
    python -m dashboard.api --openapi    # rewrite dashboard/api/openapi.json from the code

openapi.json is the contract the web UI generates its client from. v1 is frozen: add
routes or optional fields, never rename or remove; a breaking change is /v2.
"""
