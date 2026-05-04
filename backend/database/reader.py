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
    #  duration_sec, signature, llm_report, has_image, camera_id, zone, false_positive)
    return {
        "event_id":       row[0],
        "event_status":   row[1],
        "timestamp":      _ts(row[2]),
        "repeat_count":   row[3],
        "duration_sec":   float(row[4]),
        "signature":      row[5] or {},
        "llm_report":     row[6],
        "has_image":      bool(row[7]),
        "camera_id":      row[8],
        "zone":           row[9],
        "false_positive": bool(row[10]),
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
            ) AS has_image,
            e.camera_id,
            e.zone,
            e.false_positive
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
        cur.execute("SELECT COUNT(*) FROM events WHERE false_positive = FALSE")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM events WHERE event_status IN ('new','active') AND false_positive = FALSE"
        )
        active = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM events WHERE DATE(updated_at) = %s AND false_positive = FALSE", (today,)
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
            WHERE false_positive = FALSE
            """
        )
        row = cur.fetchone()
        dist = {
            "helmet": int(row[0] or 0),
            "vest":   int(row[1] or 0),
            "mask":   int(row[2] or 0),
            "fire":   int(row[3] or 0),
        }

        # Son 5 event (false positive olmayanlar)
        cur.execute(
            _base_query() + " WHERE e.false_positive = FALSE ORDER BY e.updated_at DESC LIMIT 5"
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
# Summary raporu için ham event verisi
# ---------------------------------------------------------------------------

def get_events_for_summary(start_date: str, end_date: str) -> list[dict]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT event_id, event_status, created_at, updated_at,
                   repeat_count, duration_sec,
                   helmet_violation, vest_violation, mask_violation, fire_detected,
                   camera_id, zone
            FROM events
            WHERE DATE(created_at) >= %s AND DATE(created_at) <= %s
              AND false_positive = FALSE
            ORDER BY created_at ASC
            """,
            (start_date, end_date),
        )
        rows = cur.fetchall()

    return [
        {
            "event_id":        r[0],
            "event_status":    r[1],
            "created_at":      _ts(r[2]),
            "updated_at":      _ts(r[3]),
            "repeat_count":    r[4],
            "duration_sec":    float(r[5]),
            "helmet_violation": r[6],
            "vest_violation":  r[7],
            "mask_violation":  r[8],
            "fire_detected":   r[9],
            "camera_id":       r[10],
            "zone":            r[11],
        }
        for r in rows
    ]


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
                WHERE DATE(updated_at) = %s AND false_positive = FALSE
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

    import calendar as _cal
    anchor = date.fromisoformat(date_str) if date_str else today

    if period == "weekly":
        # Pazartesi → Pazar (veya bugün)
        start_d = anchor - timedelta(days=anchor.weekday())
        end_d   = min(start_d + timedelta(days=6), today)
        days = [(start_d + timedelta(days=i)).isoformat()
                for i in range((end_d - start_d).days + 1)]
    else:
        # Ayın 1'i → son günü (veya bugün)
        start_d = anchor.replace(day=1)
        last    = _cal.monthrange(start_d.year, start_d.month)[1]
        end_d   = min(start_d.replace(day=last), today)
        days = [(start_d + timedelta(days=i)).isoformat()
                for i in range((end_d - start_d).days + 1)]

    buckets = {d: _empty() for d in days}

    with db_cursor() as cur:
        cur.execute(
            """
            SELECT
                DATE(updated_at)::text AS day,
                helmet_violation, vest_violation, mask_violation, fire_detected
            FROM events
            WHERE DATE(updated_at) BETWEEN %s AND %s AND false_positive = FALSE
            """,
            (days[0], days[-1]),
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


# ---------------------------------------------------------------------------
# Kaydedilmiş LLM raporları
# ---------------------------------------------------------------------------

def get_saved_reports(period: str | None = None, limit: int = 50) -> list[dict]:
    with db_cursor() as cur:
        if period:
            cur.execute(
                """
                SELECT id, period, report_date, generated_at, auto_generated
                FROM llm_reports
                WHERE period = %s
                ORDER BY report_date DESC LIMIT %s
                """,
                (period, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, period, report_date, generated_at, auto_generated
                FROM llm_reports
                ORDER BY report_date DESC LIMIT %s
                """,
                (limit,),
            )
        rows = cur.fetchall()
    return [
        {
            "id":            r[0],
            "period":        r[1],
            "report_date":   r[2],
            "generated_at":  _ts(r[3]),
            "auto_generated": bool(r[4]),
        }
        for r in rows
    ]


def get_saved_report(report_id: int) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT id, period, report_date, llm_text, generated_at, auto_generated
            FROM llm_reports WHERE id = %s
            """,
            (report_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id":            r[0],
        "period":        r[1],
        "report_date":   r[2],
        "llm_text":      r[3],
        "generated_at":  _ts(r[4]),
        "auto_generated": bool(r[5]),
    }
