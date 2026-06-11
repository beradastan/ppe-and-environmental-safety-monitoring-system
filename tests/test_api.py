# -*- coding: utf-8 -*-
"""
tests/test_api.py — Backend API uçtan uca testi
================================================
Backend çalışırken (python -m backend.app) çalıştırılır.
Her endpoint'i test eder, sonuçları raporlar.

Kullanım:
    python tests/test_api.py
    python tests/test_api.py --host localhost --port 5050
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

import requests
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:5050"
PASS = "[OK]"
FAIL = "[XX]"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    sym = PASS if ok else FAIL
    print(f"  {sym}  {name}" + (f"  ->  {detail}" if detail else ""))
    results.append((name, ok, detail))


def get(path: str, **kwargs):
    return requests.get(BASE + path, timeout=10, **kwargs)


def post(path: str, body: dict = None, **kwargs):
    return requests.post(BASE + path, json=body or {}, timeout=10, **kwargs)


def section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


# ── 1. Genel sağlık ──────────────────────────────────────────────────────────

def test_health():
    section("1. Backend Bağlantısı")
    try:
        r = get("/api/stats")
        check("GET /api/stats erişilebilir", r.status_code == 200, f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        check("Backend erişilebilir", False, f"Bağlantı hatası — python -m backend.app çalışıyor mu?")
        print("\n  Backend'e ulaşılamıyor, testler durduruluyor.")
        sys.exit(1)


# ── 2. Config ────────────────────────────────────────────────────────────────

def test_config():
    section("2. Config API")
    r = get("/api/config")
    check("GET /api/config", r.status_code == 200)
    if r.status_code == 200:
        cfg = r.json()
        for key in ["use_helmet", "use_vest", "use_mask", "use_fire", "temporal_window"]:
            check(f"  config['{key}'] mevcut", key in cfg, str(cfg.get(key, "YOK")))

    # PUT test
    r2 = requests.put(BASE + "/api/config", json={"temporal_window": 25}, timeout=10)
    check("PUT /api/config (temporal_window=25)", r2.status_code == 200)
    # Geri al
    requests.put(BASE + "/api/config", json={"temporal_window": 20}, timeout=10)


# ── 3. Events ────────────────────────────────────────────────────────────────

def test_events():
    section("3. Events API")
    r = get("/api/events")
    check("GET /api/events (filtresiz)", r.status_code == 200)

    r2 = get("/api/events", params={"status": "closed"})
    check("GET /api/events?status=closed", r2.status_code == 200)
    if r2.status_code == 200:
        events = r2.json().get("events", [])
        check(f"  Closed event sayısı", True, f"{len(events)} adet")
        if events:
            eid = events[0]["event_id"]
            r3 = get(f"/api/events/{eid}")
            check(f"  GET /api/events/{eid} (timeline)", r3.status_code == 200)
            if r3.status_code == 200:
                data = r3.json()
                check("  timeline alanı mevcut", "timeline" in data)
                check("  notes alanı mevcut", "notes" in data)


# ── 4. Stats ─────────────────────────────────────────────────────────────────

def test_stats():
    section("4. Stats API")
    r = get("/api/stats")
    check("GET /api/stats", r.status_code == 200)
    if r.status_code == 200:
        d = r.json()
        for key in ["active_alarms", "today_violations", "total_events"]:
            check(f"  stats['{key}'] mevcut", key in d, str(d.get(key, "YOK")))


# ── 5. Reports ───────────────────────────────────────────────────────────────

def test_reports():
    section("5. Reports API")
    for period in ["daily", "weekly", "monthly"]:
        r = get("/api/reports", params={"period": period, "date": str(date.today())})
        check(f"GET /api/reports?period={period}", r.status_code == 200,
              f"HTTP {r.status_code}")

    r2 = get("/api/reports/summary", params={"period": "daily", "date": str(date.today())})
    check("GET /api/reports/summary", r2.status_code in (200, 503),
          f"HTTP {r2.status_code}" + (" (DB kapalı)" if r2.status_code == 503 else ""))

    r3 = get("/api/reports/saved")
    check("GET /api/reports/saved", r3.status_code == 200)
    if r3.status_code == 200:
        saved = r3.json().get("reports", [])
        check(f"  Kayıtlı rapor sayısı", True, f"{len(saved)} adet")


# ── 6. Pipeline durumu ───────────────────────────────────────────────────────

def test_pipeline():
    section("6. Pipeline Status API")
    r = get("/api/pipeline/status")
    check("GET /api/pipeline/status", r.status_code == 200)
    if r.status_code == 200:
        d = r.json()
        check("  'running' alanı mevcut", "running" in d, str(d.get("running")))
        check("  'mode' alanı mevcut", "mode" in d, str(d.get("mode", "—")))


# ── 7. Özet ─────────────────────────────────────────────────────────────────

def print_summary():
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"  SONUC: {passed}/{total} basarili" + (f", {failed} basarisiz" if failed else " -- tumü gecti"))
    print(f"{'='*50}")
    if failed:
        print("\n  Basarisiz testler:")
        for name, ok, detail in results:
            if not ok:
                print(f"    {FAIL}  {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    global BASE
    BASE = f"http://{args.host}:{args.port}"
    print(f"Backend: {BASE}")

    test_health()
    test_config()
    test_events()
    test_stats()
    test_reports()
    test_pipeline()
    print_summary()


if __name__ == "__main__":
    main()
