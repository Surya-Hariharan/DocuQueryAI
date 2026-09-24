"""
Shared test setup.

config.py raises EnvironmentError at import time if required secrets are
missing — correct for the running app, but it means importing almost any
module in this codebase transitively requires a valid .env. Tests must not
depend on the real .env (with real secrets) being present, so dummy values
are set here, before any application module gets imported by a test file.

This is a safety boundary, not just a convenience: a real .env with real
production credentials sits at the repo root (see the security audit this
project has been through). DB_HOST/DB_PORT/DB_TABLE/DATABASE_URL are
force-overridden (not setdefault) rather than left to fall through to
whatever's in that file — a test suite must never be able to reach a real
database by accident just because a variable was missing here. This was
found the hard way: a test that mocks the DB layer but still triggers
FastAPI's startup lifespan (which calls init_connection_pool()) actually
attempted a live connection to a real Render.com Postgres host before this
fix, and only failed to connect because the dummy password happened to be
wrong — that is luck, not safety.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("BEARER_TOKEN", "test-bearer-token")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")

# Force-overridden, not setdefault: these must never resolve to a real,
# reachable database, regardless of what a local .env happens to contain.
os.environ["DB_HOST"] = "127.0.0.1"
os.environ["DB_PORT"] = "1"  # a port nothing legitimate listens on
os.environ["DB_TABLE"] = "test_document_chunks"
os.environ["DATABASE_URL"] = ""
