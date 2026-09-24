"""
Shared test setup.

config.py raises EnvironmentError at import time if required secrets are
missing — correct for the running app, but it means importing almost any
module in this codebase transitively requires a valid .env. Tests must not
depend on the real .env (with real secrets) being present, so dummy values
are set here, before any application module gets imported by a test file.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("BEARER_TOKEN", "test-bearer-token")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
