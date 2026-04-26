# -*- coding: utf-8 -*-
"""
connection.py
=============
psycopg2 bagli havuzu ve konteks yoneticisi.

DSN onceligi:
  1. DATABASE_URL ortam degiskeni
  2. config.yaml -> database bloku
  3. Yerlesik varsayilanlar (localhost gelistirme)

Kullanim:
    from backend.database.connection import db_cursor, init_pool, init_db

    init_pool()   # uygulama baslarken bir kez
    init_db()     # tabloları olustur (idempotent)

    with db_cursor() as cur:
        cur.execute("SELECT 1")
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger("db.connection")

_pool: ThreadedConnectionPool | None = None

# ---------------------------------------------------------------------------
# DSN
# ---------------------------------------------------------------------------

def _build_dsn() -> str:
    env = os.environ.get("DATABASE_URL")
    if env:
        return env

    # config.yaml'dan oku
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            db  = cfg.get("database", {})
            host = db.get("host",     "localhost")
            port = db.get("port",     5432)
            name = db.get("name",     "ppe_db")
            user = db.get("user",     "postgres")
            pwd  = db.get("password", "postgres")
            return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
    except Exception:
        pass

    return "postgresql://postgres:postgres@localhost:5432/ppe_db"


# ---------------------------------------------------------------------------
# Havuz
# ---------------------------------------------------------------------------

def init_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    global _pool
    dsn = _build_dsn()
    _pool = ThreadedConnectionPool(min_conn, max_conn, dsn=dsn)
    logger.info("DB havuzu hazir.")


def _get_conn():
    if _pool is None:
        raise RuntimeError("DB havuzu baslatilmadi. once init_pool() cagir.")
    return _pool.getconn()


def _put_conn(conn) -> None:
    if _pool:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Konteks yoneticisi
# ---------------------------------------------------------------------------

@contextmanager
def db_cursor():
    """
    Commit/rollback otomatik. Hata durumunda rollback yapar.

    with db_cursor() as cur:
        cur.execute(...)
    """
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


# ---------------------------------------------------------------------------
# Tablo olusturma (idempotent)
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db() -> None:
    """schema.sql'i calistirir; tablolar zaten varsa atlar."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with db_cursor() as cur:
        cur.execute(sql)
    logger.info("DB semalari hazir.")
