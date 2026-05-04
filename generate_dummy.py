# -*- coding: utf-8 -*-
"""
generate_dummy.py
=================
Gercekci PPE ihlal event'leri uretir ve DB'ye yazar.

Kullanim:
    python generate_dummy.py
    python generate_dummy.py --days 60 --count 300
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone

import psycopg2

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

CAMERAS = [
    {"id": "cam_01", "zone": "Giris/Cikis"},
    {"id": "cam_02", "zone": "Uretim Hatti A"},
    {"id": "cam_03", "zone": "Uretim Hatti B"},
    {"id": "cam_04", "zone": "Depo"},
]

CAM_WEIGHTS = {
    "cam_01": {"helmet": 0.55, "vest": 0.20, "mask": 0.10, "fire": 0.05, "multi": 0.10},
    "cam_02": {"helmet": 0.30, "vest": 0.20, "mask": 0.15, "fire": 0.12, "multi": 0.23},
    "cam_03": {"helmet": 0.28, "vest": 0.18, "mask": 0.12, "fire": 0.18, "multi": 0.24},
    "cam_04": {"helmet": 0.20, "vest": 0.40, "mask": 0.20, "fire": 0.08, "multi": 0.12},
}

LLM_TEMPLATES = [
    (
        "Kamera {cam_id} ({zone}) bolgesinde {ts} tarihinde guvenlik ihlali tespit edildi. "
        "{persons_desc}"
        "Olay {dur:.0f} saniye surdu ve {repeat} kez tekrarlandi. "
        "Ilgili personelin KKD kullanimi konusunda uyarilmasi ve denetim sikliginin artirilmasi onerilir."
    ),
    (
        "{zone} bolgesinde {persons_desc}"
        "Toplam {dur:.0f} saniyelik bu surecte {repeat} ihlal kaydedildi. "
        "Bolge sorumlusunun konu hakkinda bilgilendirilmesi ve KKD denetimleri siklestirilmalidir."
    ),
    (
        "Is guvenligi sistemi, {cam_id} kamerasinda {ts} itibariyla ihlal algilladi. "
        "{persons_desc}"
        "Sahada gorevli personelin is guvenligi prosedurlerine uyumu denetlenmeli, "
        "gerekirse ek egitim planlanmalidir. Ihlal suresi: {dur:.0f} sn, tekrar: {repeat}."
    ),
    (
        "{zone} -- {ts}: {persons_desc}"
        "Acil onlem alinarak ilgili personel uyarilmis, olay kayit altina alinmistir. "
        "Tekrar sayisi ({repeat}) dikkate alindiginda bolgede rutin denetim sikligi artirilmalidir."
    ),
]

FIRE_LLM = (
    "{zone} bolgesinde {ts} tarihinde yangin/duman belirtisi tespit edildi. "
    "Acil mudahale protokolu devreye alindi. Olay {dur:.0f} saniye surdu. "
    "Yangin sondurme ekipmanlarinin konumunun gozden gecirilmesi ve personelin tahliye "
    "prosedUrleri konusunda bilgilendirilmesi onerilir."
)

VIOLATION_NAMES = {
    "no_helmet": "baret",
    "no_vest":   "yelek",
    "no_mask":   "maske",
}

NOTES_POOL = [
    "Personel uyarildi, tekrar incelenecek.",
    "Vardiya amiri bilgilendirildi.",
    "Kamera acisi kontrol edildi, tespit dogru.",
    "Ilgili departmana yazili uyari gonderildi.",
    "Personel KKD egitimne yonlendirildi.",
    "Saha ziyareti planlandi.",
    "Tekrarlayan ihlal, disiplin sureci baslatildi.",
    "Ekipman eksikligi tespit edildi, temin edilecek.",
]


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _rnd_conf(present: bool) -> float:
    if present:
        return round(random.uniform(0.72, 0.97), 2)
    return round(random.uniform(0.00, 0.18), 2)


def _make_persons(violations_set: set) -> list:
    n = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    persons = []
    for i in range(n):
        tid = i + 1
        p_viols = list(violations_set) if i == 0 else [
            v for v in violations_set if random.random() < 0.5
        ]
        persons.append({
            "track_id":    tid,
            "violations":  p_viols,
            "helmet_conf": _rnd_conf("no_helmet" not in p_viols),
            "vest_conf":   _rnd_conf("no_vest"   not in p_viols),
            "mask_conf":   _rnd_conf("no_mask"   not in p_viols),
        })
    return persons


def _make_signature(violations_set: set, fire: bool, persons: list) -> dict:
    sig = {
        "helmet_violation": "no_helmet" in violations_set,
        "vest_violation":   "no_vest"   in violations_set,
        "mask_violation":   "no_mask"   in violations_set,
        "fire_detected":    fire,
    }
    for viol, key in [("no_helmet", "helmet_missing_ids"),
                      ("no_vest",   "vest_missing_ids"),
                      ("no_mask",   "mask_missing_ids")]:
        ids = [p["track_id"] for p in persons if viol in p["violations"]]
        if ids:
            sig[key] = ids
    return sig


def _pick_violations(cam_id: str) -> tuple:
    w = CAM_WEIGHTS[cam_id]
    choice = random.choices(
        ["helmet", "vest", "mask", "fire", "multi"],
        weights=[w["helmet"], w["vest"], w["mask"], w["fire"], w["multi"]],
    )[0]
    if choice == "helmet":
        return {"no_helmet"}, False, "ppe_violation"
    if choice == "vest":
        return {"no_vest"}, False, "ppe_violation"
    if choice == "mask":
        return {"no_mask"}, False, "ppe_violation"
    if choice == "fire":
        return set(), True, "fire_detected"
    opts = [
        ({"no_helmet", "no_vest"},            False, "multi_hazard"),
        ({"no_helmet", "no_mask"},            False, "multi_hazard"),
        ({"no_vest",   "no_mask"},            False, "multi_hazard"),
        ({"no_helmet", "no_vest", "no_mask"}, False, "multi_hazard"),
        ({"no_helmet"},                       True,  "multi_hazard"),
    ]
    viols, fire, etype = random.choice(opts)
    return viols, fire, etype


def _work_ts(base_date: datetime) -> datetime:
    hour = random.choices(
        list(range(24)),
        weights=[
            0.2, 0.1, 0.1, 0.1, 0.1, 0.3,
            0.8, 2.5, 4.0, 4.0, 3.5, 2.5,
            1.5, 3.0, 4.0, 4.0, 3.0, 2.0,
            1.0, 0.5, 0.3, 0.2, 0.2, 0.2,
        ],
    )[0]
    return base_date.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        tzinfo=timezone.utc,
    )


def _make_llm(etype: str, cam: dict, ts: datetime,
              dur: float, repeat: int, persons: list) -> str:
    ts_str = ts.strftime("%d.%m.%Y %H:%M")
    if etype == "fire_detected":
        return FIRE_LLM.format(zone=cam["zone"], ts=ts_str, dur=dur)
    parts = []
    for p in persons:
        if p["violations"]:
            names = ", ".join(VIOLATION_NAMES[v] for v in p["violations"] if v in VIOLATION_NAMES)
            parts.append(f"#{p['track_id']} numarali personelde {names} eksikligi tespit edildi.")
    persons_desc = " ".join(parts) + " " if parts else "PPE ihlali tespit edildi. "
    tmpl = random.choice(LLM_TEMPLATES)
    return tmpl.format(
        cam_id=cam["id"], zone=cam["zone"], ts=ts_str,
        dur=dur, repeat=repeat, persons_desc=persons_desc,
    )


# ---------------------------------------------------------------------------
# Ana uretici
# ---------------------------------------------------------------------------

def generate(days: int, count: int, dsn: str) -> None:
    conn = psycopg2.connect(dsn)
    cur  = conn.cursor()

    cur.execute("DELETE FROM events WHERE event_id LIKE 'evt_dummy_%'")
    cur.execute("DELETE FROM events WHERE camera_id IS NULL AND event_id != 'evt_0074'")
    print("Eski dummy kayitlar silindi.")

    now  = datetime.now(tz=timezone.utc)
    base = now - timedelta(days=days)

    def pick_day() -> datetime:
        d = base + timedelta(days=random.randint(0, days - 1))
        if d.weekday() >= 5 and random.random() < 0.60:
            d = base + timedelta(days=random.randint(0, days - 1))
        return d

    inserted = 0
    for _ in range(count):
        eid      = f"evt_dummy_{uuid.uuid4().hex[:8]}"
        cam      = random.choice(CAMERAS)
        viols, fire, etype = _pick_violations(cam["id"])

        persons  = _make_persons(viols) if not fire else []
        sig      = _make_signature(viols, fire, persons)

        dur    = round(random.uniform(8.0, 300.0), 2)
        repeat = random.choices(
            [1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
            weights=[20, 18, 14, 10, 8, 7, 6, 5, 4, 3],
        )[0]

        created_ts = _work_ts(pick_day())
        closed_ts  = created_ts + timedelta(seconds=dur + random.uniform(1, 8))

        llm          = _make_llm(etype, cam, created_ts, dur, repeat, persons)
        sig_json     = json.dumps(sig,     ensure_ascii=False)
        persons_json = json.dumps(persons, ensure_ascii=False)

        cur.execute(
            """
            INSERT INTO events (
                event_id, event_status, created_at, updated_at,
                repeat_count, duration_sec,
                helmet_violation, vest_violation, mask_violation, fire_detected,
                signature, llm_report, persons, camera_id, zone, false_positive
            ) VALUES (%s,'closed',%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,FALSE)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (
                eid, created_ts, closed_ts, repeat, dur,
                sig["helmet_violation"], sig["vest_violation"],
                sig["mask_violation"],   sig["fire_detected"],
                sig_json, llm, persons_json,
                cam["id"], cam["zone"],
            ),
        )

        # Timeline: new -> active -> closed
        mid_ts = created_ts + timedelta(seconds=dur * 0.4)
        for status, ts, reason, rep, d in [
            ("new",    created_ts, "initial_violation",  max(1, repeat // 3), 0.0),
            ("active", mid_ts,     "repeat_violation",   max(1, repeat // 2), 0.0),
            ("closed", closed_ts,  "violation_resolved", repeat,              dur),
        ]:
            cur.execute(
                """
                INSERT INTO event_timeline (
                    event_id, event_status, ts,
                    repeat_count, duration_sec, change_reason,
                    signature, llm_report, persons
                ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)
                """,
                (
                    eid, status, ts, rep, d, reason,
                    sig_json,
                    llm if status == "closed" else None,
                    persons_json,
                ),
            )

        if random.random() < 0.15:
            note_ts = closed_ts + timedelta(minutes=random.randint(2, 45))
            cur.execute(
                "INSERT INTO event_notes (event_id, note_text, created_at) VALUES (%s,%s,%s)",
                (eid, random.choice(NOTES_POOL), note_ts),
            )

        inserted += 1
        if inserted % 50 == 0:
            print(f"  {inserted}/{count}...")

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM events")
    total = cur.fetchone()[0]
    conn.close()
    print(f"\nTamamlandi: {inserted} event eklendi. DB toplam: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=45,  help="Kac gunluk gecmis")
    parser.add_argument("--count", type=int, default=220, help="Uretilecek event sayisi")
    parser.add_argument("--dsn",   default="postgresql://postgres:1234@localhost:5432/ppe_db")
    args = parser.parse_args()
    generate(args.days, args.count, args.dsn)
