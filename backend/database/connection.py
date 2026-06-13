from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger("db.connection")

_pool: ThreadedConnectionPool | None = None

def _build_dsn() -> str:
    env = os.environ.get("DATABASE_URL")
    if env:
        return env

    try:
        import yaml
        cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if cfg_path.exists():
            cfg  = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            db   = cfg.get("database", {})
            host = db.get("host",     "localhost")
            port = db.get("port",     5432)
            name = db.get("name",     "ppe_db")
            user = db.get("user",     "postgres")
            pwd  = db.get("password", "")
            return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
    except Exception:
        pass

    raise RuntimeError(
        "Database configuration not found. "
        "Set the DATABASE_URL environment variable or configure database in config.yaml."
    )

def init_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    global _pool
    dsn = _build_dsn()
    _pool = ThreadedConnectionPool(min_conn, max_conn, dsn=dsn)
    logger.info("Connection pool initialized.")

def _get_conn():
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call init_pool() first.")
    return _pool.getconn()

def _put_conn(conn) -> None:
    if _pool:
        _pool.putconn(conn)

@contextmanager
def db_cursor():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

def init_db() -> None:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with db_cursor() as cur:
        cur.execute(sql)
    logger.info("Database schema ready.")
