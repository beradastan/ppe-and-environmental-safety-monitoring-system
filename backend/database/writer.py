# -*- coding: utf-8 -*-
"""
writer.py
=========
Olayları ve notları PostgreSQL'e yazar.

Ana fonksiyonlar:
    write_event(data, image_filename=None)
        — event JSON verisini events + event_timeline tablolarına yazar.
        — events tablosunda UPSERT yapar (en son durum her zaman güncel).
        — event_timeline'a yeni satır ekler (geçmiş korunur).

    write_note(event_id, note_text)
        — event_notes tablosuna not ekler.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from .connection import db_cursor

logger = logging.getLogger("db.writer")


def _parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.now()


def _sig_flags(sig: dict) -> tuple[bool, bool, bool, bool]:
    return (
        bool(sig.get("helmet_violation") or sig.get("helmet_missing_ids")),
        bool(sig.get("vest_violation")   or sig.get("vest_missing_ids")),
        bool(sig.get("mask_violation")   or sig.get("mask_missing_ids")),
        bool(sig.get("fire_detected")),
    )


def write_event(data: dict, image_filename: str | None = None) -> None:
    """
    Event JSON'unu DB'ye yazar.
    events: UPSERT (en son durum)
    event_timeline: INSERT (geçmis kayit)
    """
    event_id      = data.get("event_id", "")
    event_status  = data.get("event_status", "")
    ts            = _parse_ts(data.get("timestamp", ""))
    repeat_count  = int(data.get("repeat_count", 0))
    duration_sec  = float(data.get("duration_sec", 0.0))
    signature     = data.get("signature", {}) or {}
    llm_report    = data.get("llm_report")
    change_reason = data.get("change_reason", "")
    persons       = data.get("persons")
    camera_id     = data.get("camera_id")
    zone          = data.get("zone")

    h_viol, v_viol, m_viol, fire = _sig_flags(signature)
    sig_json     = json.dumps(signature, ensure_ascii=False)
    persons_json = json.dumps(persons, ensure_ascii=False) if persons is not None else None

    try:
        with db_cursor() as cur:
            # events: UPSERT — en son durum
            cur.execute(
                """
                INSERT INTO events (
                    event_id, event_status, created_at, updated_at,
                    repeat_count, duration_sec,
                    helmet_violation, vest_violation, mask_violation, fire_detected,
                    signature, llm_report, persons, camera_id, zone
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    event_status     = EXCLUDED.event_status,
                    updated_at       = EXCLUDED.updated_at,
                    repeat_count     = EXCLUDED.repeat_count,
                    duration_sec     = EXCLUDED.duration_sec,
                    helmet_violation = EXCLUDED.helmet_violation,
                    vest_violation   = EXCLUDED.vest_violation,
                    mask_violation   = EXCLUDED.mask_violation,
                    fire_detected    = EXCLUDED.fire_detected,
                    signature        = EXCLUDED.signature,
                    llm_report       = EXCLUDED.llm_report,
                    persons          = EXCLUDED.persons,
                    camera_id        = COALESCE(EXCLUDED.camera_id, events.camera_id),
                    zone             = COALESCE(EXCLUDED.zone, events.zone)
                """,
                (
                    event_id, event_status, ts, ts,
                    repeat_count, duration_sec,
                    h_viol, v_viol, m_viol, fire,
                    sig_json, llm_report, persons_json, camera_id, zone,
                ),
            )

            # event_timeline: her gecis ayri satir
            cur.execute(
                """
                INSERT INTO event_timeline (
                    event_id, event_status, ts,
                    repeat_count, duration_sec, change_reason,
                    signature, llm_report, image_filename, persons
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                """,
                (
                    event_id, event_status, ts,
                    repeat_count, duration_sec, change_reason,
                    sig_json, llm_report, image_filename, persons_json,
                ),
            )

        logger.debug("Written: %s [%s]", event_id, event_status)

    except Exception as exc:
        logger.error("write_event error: %s", exc, exc_info=True)


def close_event(event_id: str, repeat_count: int | None = None, duration_sec: float | None = None) -> None:
    """events tablosunda event_status='closed' olarak günceller (video sonu / pipeline kapanışı)."""
    now = datetime.now()
    try:
        with db_cursor() as cur:
            fields = ["event_status = %s", "updated_at = %s"]
            values: list = ["closed", now]
            if repeat_count is not None:
                fields.append("repeat_count = %s")
                values.append(repeat_count)
            if duration_sec is not None:
                fields.append("duration_sec = %s")
                values.append(duration_sec)
            values.append(event_id)
            cur.execute(
                f"UPDATE events SET {', '.join(fields)} WHERE event_id = %s",
                values,
            )
        logger.debug("Event closed: %s", event_id)
    except Exception as exc:
        logger.error("close_event error: %s", exc, exc_info=True)


def write_note(event_id: str, note_text: str) -> datetime:
    """
    event_notes tablosuna not ekler.
    Dondurur: created_at zamani.
    """
    now = datetime.now()
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_notes (event_id, note_text, created_at)
                VALUES (%s, %s, %s)
                """,
                (event_id, note_text, now),
            )
        logger.debug("Note written: %s", event_id)
    except Exception as exc:
        logger.error("write_note error: %s", exc, exc_info=True)
    return now


def resolve_event(event_id: str) -> None:
    """events tablosunda event_status='closed' olarak günceller (pipeline callback)."""
    now = datetime.now()
    try:
        with db_cursor() as cur:
            cur.execute(
                "UPDATE events SET event_status = %s, updated_at = %s WHERE event_id = %s",
                ("closed", now, event_id),
            )
        logger.debug("Event resolved: %s", event_id)
    except Exception as exc:
        logger.error("resolve_event error: %s", exc, exc_info=True)


def mark_false_positive(event_id: str) -> None:
    """Event'i yanlış tespit olarak işaretler; henüz kapatılmamışsa kapatır."""
    now = datetime.now()
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE events
                SET false_positive = TRUE,
                    event_status   = CASE WHEN event_status != 'closed' THEN 'closed' ELSE event_status END,
                    updated_at     = %s
                WHERE event_id = %s
                """,
                (now, event_id),
            )
        logger.debug("Marked as false positive: %s", event_id)
    except Exception as exc:
        logger.error("mark_false_positive error: %s", exc, exc_info=True)



def save_llm_report(
    period: str,
    report_date: str,
    llm_text: str,
    auto_generated: bool = False,
) -> None:
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_reports (period, report_date, llm_text, auto_generated)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (period, report_date) DO UPDATE
                    SET llm_text       = EXCLUDED.llm_text,
                        generated_at   = NOW(),
                        auto_generated = EXCLUDED.auto_generated
                """,
                (period, report_date, llm_text, auto_generated),
            )
        logger.debug("LLM report saved: %s %s", period, report_date)
    except Exception as exc:
        logger.error("save_llm_report error: %s", exc, exc_info=True)
