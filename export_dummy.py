import json
import logging
from backend.database.connection import init_pool, init_db, db_cursor

logging.basicConfig(level=logging.ERROR)

def export_dummy_data():
    init_pool()
    with db_cursor() as cur:
        cur.execute('''SELECT event_id, event_status, created_at, updated_at, repeat_count, duration_sec,
                       helmet_violation, vest_violation, mask_violation, fire_detected,
                       camera_id, zone, signature, llm_report, persons
                       FROM events
                       WHERE llm_report = 'Bu event gecmise yonelik otomatik uretilmistir.'
                       ORDER BY created_at DESC''')
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        events = []
        for row in rows:
            evt = dict(zip(cols, row))
            evt['created_at'] = evt['created_at'].isoformat()
            evt['updated_at'] = evt['updated_at'].isoformat()
            events.append(evt)

    with open('dummy_events_export.json', 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4)

    print(f"Toplam {len(events)} kayit dummy_events_export.json dosyasina yazildi.")

if __name__ == "__main__":
    export_dummy_data()
