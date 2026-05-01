import random
from datetime import datetime, timedelta
import uuid
import json
import logging

logging.basicConfig(level=logging.INFO)

from backend.database.connection import init_pool, init_db, db_cursor

_CAMERAS = [
    ("cam_01", "Giriş"),
    ("cam_02", "Üretim Hattı A"),
    ("cam_03", "Üretim Hattı B"),
    ("cam_04", "Depo"),
]

def generate_random_event(event_id, date):
    status = "closed"
    duration = random.uniform(10.0, 180.0)
    repeat_count = random.randint(1, 10)

    helmet = random.choices([True, False], weights=[0.25, 0.75])[0]
    vest = random.choices([True, False], weights=[0.15, 0.85])[0]
    mask = random.choices([True, False], weights=[0.35, 0.65])[0]
    fire = random.choices([True, False], weights=[0.05, 0.95])[0]

    if not (helmet or vest or mask or fire):
        helmet = True

    camera_id, zone = random.choice(_CAMERAS)
    signature = {"class": "Dummy", "confidence": random.uniform(0.5, 0.99)}
    persons = {
        "1": {
            "hardhat": "NO-Hardhat" if helmet else "Hardhat",
            "vest": "NO-Safety Vest" if vest else "Safety Vest",
            "mask": "NO-Mask" if mask else "Mask"
        }
    }
    llm_report = "Bu event gecmise yonelik otomatik uretilmistir."

    updated_time = date + timedelta(seconds=duration)

    return {
        "event_id": event_id,
        "event_status": status,
        "created_at": date,
        "updated_at": updated_time,
        "repeat_count": repeat_count,
        "duration_sec": duration,
        "helmet_violation": helmet,
        "vest_violation": vest,
        "mask_violation": mask,
        "fire_detected": fire,
        "camera_id": camera_id,
        "zone": zone,
        "signature": json.dumps(signature),
        "llm_report": llm_report,
        "persons": json.dumps(persons)
    }

def main():
    init_pool()
    init_db()

    now = datetime.now()
    records_to_insert = 150  # 150 events over 30 days

    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events WHERE llm_report = 'Bu event gecmise yonelik otomatik uretilmistir.'")
        if cur.fetchone()[0] > 0:
            print("Zaten dummy data var, oncekileri temizliyorum...")
            cur.execute("DELETE FROM events WHERE llm_report = 'Bu event gecmise yonelik otomatik uretilmistir.'")

        print(f"Toplam {records_to_insert} dummy kayit gecmis 30 gune dagitiliyor...")

        for _ in range(records_to_insert):
            days_ago = random.uniform(0, 30)
            date = now - timedelta(days=days_ago)
            evt = generate_random_event(f"evt_dummy_{uuid.uuid4().hex[:8]}", date)

            cur.execute("""
                INSERT INTO events (
                    event_id, event_status, created_at, updated_at, repeat_count, duration_sec,
                    helmet_violation, vest_violation, mask_violation, fire_detected,
                    camera_id, zone, signature, llm_report, persons
                ) VALUES (
                    %(event_id)s, %(event_status)s, %(created_at)s, %(updated_at)s, %(repeat_count)s, %(duration_sec)s,
                    %(helmet_violation)s, %(vest_violation)s, %(mask_violation)s, %(fire_detected)s,
                    %(camera_id)s, %(zone)s, %(signature)s, %(llm_report)s, %(persons)s
                )
            """, evt)

            cur.execute("""
                INSERT INTO event_timeline (
                    event_id, event_status, ts, repeat_count, duration_sec, change_reason, signature, llm_report, persons
                ) VALUES (
                    %(event_id)s, %(event_status)s, %(updated_at)s, %(repeat_count)s, %(duration_sec)s, 'Resolved automatically',
                    %(signature)s, %(llm_report)s, %(persons)s
                )
            """, evt)

    print("Olusturma tamamlandi!")

if __name__ == "__main__":
    main()
