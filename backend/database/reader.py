# -*- coding: utf-8 -*-
"""
reader.py
=========
PostgreSQL'den event verisini okur.
event_reader.py ile birebir ayni arayuzu saglar;
app.py'deki import'lar degismeden calismaya devam eder.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from .connection import db_cursor

logger = logging.getLogger("db.reader")


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _ts(val) -> str:
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _row_to_event(row) -> dict:
    # (event_id, event_status, updated_at, repeat_count,
    #  duration_sec, signature, llm_report, has_image)
    return {
        "event_id":      row[0],
        "event_status":  row[1],
        "timestamp":     _ts(row[2]),
        "repeat_count":  row[3],
        "duration_sec":  float(row[4]),
        "signature":     row[5] or {},
        "llm_report":    row[6],
        "has_image":     bool(row[7]),
    }


def _base_query() -> str:
    return """
        SELECT
            e.event_id,
            e.event_status,
            e.updated_at,
            e.repeat_count,
            e.duration_sec,
            e.signature,
            e.llm_report,
            EXISTS (
                SELECT 1 FROM event_timeline t
                WHERE t.event_id = e.event_id
                  AND t.image_filename IS NOT NULL
            ) AS has_image
        FROM events e
    """


# ---------------------------------------------------------------------------
# Genel sorgular
# ---------------------------------------------------------------------------

def get_all_events() -> list[dict]:
    with db_cursor() as cur:
        cur.execute(_base_query() + " ORDER BY e.updated_at DESC")
        return [_row_to_event(r) for r in cur.fetchall()]


def get_filtered_events(
    date_str:       str | None = None,
    violation_type: str | None = None,
    status:         str | None = None,
) -> list[dict]:
    conditions: list[str] = []
    params:     list      = []

    if date_str:
        conditions.append("DATE(e.updated_at) = %s")
        params.append(date_str)

    if violation_type:
        col = {
            "helmet": "e.helmet_violation",
            "vest":   "e.vest_violation",
            "mask":   "e.mask_violation",
            "fire":   "e.fire_detected",
        }.get(violation_type)
        if col:
            conditions.append(f"{col} = TRUE")

    if status:
        conditions.append("e.event_status = %s")
        params.append(status)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with db_cursor() as cur:
        cur.execute(
            _base_query() + f" {where} ORDER BY e.updated_at DESC",
            params,
        )
        return [_row_to_event(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Istatistikler
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    today = date.today().isoformat()

    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM events WHERE event_status IN ('new','active')"
        )
        active = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM events WHERE DATE(updated_at) = %s", (today,)
        )
        today_ct = cur.fetchone()[0]

        cur.execute(
            """
            SELECT
                SUM(helmet_violation::int),
                SUM(vest_violation::int),
                SUM(mask_violation::int),
                SUM(fire_detected::int)
            FROM events
            """
        )
        row = cur.fetchone()
        dist = {
            "helmet": int(row[0] or 0),
            "vest":   int(row[1] or 0),
            "mask":   int(row[2] or 0),
            "fire":   int(row[3] or 0),
        }

        # Son 5 event
        cur.execute(
            _base_query() + " ORDER BY e.updated_at DESC LIMIT 5"
        )
        recent = [_row_to_event(r) for r in cur.fetchall()]

    return {
        "total_events":     total,
        "active_alarms":    active,
        "today_violations": today_ct,
        "distribution":     dist,
        "recent":           recent,
    }


# ---------------------------------------------------------------------------
# Rapor verisi
# ---------------------------------------------------------------------------

def get_report_data(period: str, date_str: str | None = None) -> list[dict]:
    today = date.today()

    def _empty() -> dict:
        return {"helmet": 0, "vest": 0, "mask": 0, "fire": 0, "total": 0}

    if period == "daily":
        target = date.fromisoformat(date_str) if date_str else today
        buckets = {f"{h:02d}:00": _empty() for h in range(24)}

        with db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    TO_CHAR(updated_at AT TIME ZONE 'UTC', 'HH24') AS hr,
                    helmet_violation, vest_violation, mask_violation, fire_detected
                FROM events
                WHERE DATE(updated_at) = %s
                """,
                (target.isoformat(),),
            )
            for row in cur.fetchall():
                key = f"{row[0]}:00"
                if key not in buckets:
                    continue
                if row[1]: buckets[key]["helmet"] += 1
                if row[2]: buckets[key]["vest"]   += 1
                if row[3]: buckets[key]["mask"]   += 1
                if row[4]: buckets[key]["fire"]   += 1
                buckets[key]["total"] += 1

        return [{"label": k, **v} for k, v in sorted(buckets.items())]

    n_days = 7 if period == "weekly" else 30
    days   = [(today - timedelta(days=i)).isoformat() for i in range(n_days - 1, -1, -1)]
    buckets = {d: _empty() for d in days}

    with db_cursor() as cur:
        start = days[0]
        cur.execute(
            """
            SELECT
                DATE(updated_at)::text AS day,
                helmet_violation, vest_violation, mask_violation, fire_detected
            FROM events
            WHERE DATE(updated_at) >= %s
            """,
            (start,),
        )
        for row in cur.fetchall():
            key = row[0]
            if key not in buckets:
                continue
            if row[1]: buckets[key]["helmet"] += 1
            if row[2]: buckets[key]["vest"]   += 1
            if row[3]: buckets[key]["mask"]   += 1
            if row[4]: buckets[key]["fire"]   += 1
            buckets[key]["total"] += 1

    return [{"label": k, **v} for k, v in buckets.items()]


# ---------------------------------------------------------------------------
# Event detay
# ---------------------------------------------------------------------------

def get_event_timeline(event_id: str) -> list[dict]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT
                event_id, event_status, ts,
                repeat_count, duration_sec, change_reason,
                signature, llm_report, image_filename
            FROM event_timeline
            WHERE event_id = %s
            ORDER BY ts ASC, recorded_at ASC
            """,
            (event_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "event_id":       r[0],
            "event_status":   r[1],
            "timestamp":      _ts(r[2]),
            "repeat_count":   r[3],
            "duration_sec":   float(r[4]),
            "change_reason":  r[5] or "",
            "signature":      r[6] or {},
            "llm_report":     r[7],
            "image_filename": r[8],
        }
        for r in rows
    ]


def get_notes(event_id: str) -> list[dict]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT note_text, created_at
            FROM event_notes
            WHERE event_id = %s
            ORDER BY created_at ASC
            """,
            (event_id,),
        )
        return [
            {"text": r[0], "timestamp": _ts(r[1])}
            for r in cur.fetchall()
        ]


def save_note(event_id: str, text: str) -> dict:
    from .writer import write_note
    ts = write_note(event_id, text)
    return {"text": text, "timestamp": ts.isoformat()}
