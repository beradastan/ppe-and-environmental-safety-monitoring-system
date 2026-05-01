import json
import os
import time
from backend.reports.services import EventAnalyticsService, ReportSummaryService, ReportRepository
from llm.safety_report_agent import SafetyReportAgent


DUMMY_EVENTS_PATH = "dummy_events_export.json"
MODEL_NAME = "qwen3:8b"
FACILITY_NAME = "Fabrika"


def load_events() -> list:
    if not os.path.exists(DUMMY_EVENTS_PATH):
        raise FileNotFoundError(
            f"{DUMMY_EVENTS_PATH} bulunamadı. "
            "Önce generate_dummy.py ve export_dummy.py çalıştır."
        )
    with open(DUMMY_EVENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_report_flow():
    events = load_events()
    print(f"Yüklendi: {len(events)} event\n")

    analytics = EventAnalyticsService(events)
    summary_svc = ReportSummaryService(analytics)
    repo = ReportRepository()
    agent = SafetyReportAgent(model_name=MODEL_NAME, facility_name=FACILITY_NAME)

    reports = [
        ("daily",   lambda: summary_svc.generate_daily_summary("2026-04-30")),
        ("weekly",  lambda: summary_svc.generate_weekly_summary("2026-04-24", "2026-04-30")),
        ("monthly", lambda: summary_svc.generate_monthly_summary("2026-04-01", "2026-04-30")),
    ]

    for report_type, build_summary in reports:
        print(f"{'='*60}")
        print(f"  {report_type.upper()} RAPORU")
        print(f"{'='*60}")

        summary = build_summary()

        # Risk özeti ekrana bas
        risk = summary.get("risk_summary", {})
        comparison = summary.get("comparison", {})
        print(f"  Toplam olay    : {summary.get('total_events', 0)}")
        print(f"  Risk           : {risk.get('risk_level')} (raw={risk.get('risk_score')}, normalized={risk.get('normalized_score')})")
        if comparison.get("trend") != "no_data":
            print(f"  Trend          : {comparison.get('trend')} ({comparison.get('change_percent')}%)")
        loc = summary.get("location_breakdown", [])
        if loc:
            top = loc[0]
            print(f"  Top lokasyon   : {top['zone']} ({top['camera_id']}) — {top['event_count']} olay")
        print()

        # LLM'e gönder
        print(f"  → {MODEL_NAME} çağrılıyor...")
        t0 = time.time()
        llm_text = agent.generate_report(summary)
        elapsed = round(time.time() - t0, 1)
        print(f"  ✓ Yanıt alındı ({elapsed}s)\n")

        print(llm_text)
        print()

        # Kaydet
        out_path = f"{report_type}_report_output.json"
        repo.save_report({"summary": summary, "llm_text": llm_text}, out_path)
        print(f"  → Kaydedildi: {out_path}\n")


if __name__ == "__main__":
    run_report_flow()
