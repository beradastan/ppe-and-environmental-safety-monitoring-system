import json
import os
from datetime import datetime, timedelta
from backend.reports.services import EventAnalyticsService, ReportSummaryService, ReportRepository

def main():
    dummy_events_path = "dummy_events_export.json"
    if not os.path.exists(dummy_events_path):
        print(f"Error: {dummy_events_path} not found.")
        return

    with open(dummy_events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    analytics_service = EventAnalyticsService(events)
    summary_service = ReportSummaryService(analytics_service)
    repo = ReportRepository()

    # Define the dates based on the dummy data
    # the latest date in dummy data is "2026-04-30"
    end_date_str = "2026-04-30"

    # daily
    daily_summary = summary_service.generate_daily_summary(end_date_str)
    repo.save_report(daily_summary, "daily_report_summary.json")
    print("Created daily_report_summary.json")

    # weekly
    start_date_weekly = "2026-04-24" # 7 days ending on 04-30
    weekly_summary = summary_service.generate_weekly_summary(start_date_weekly, end_date_str)
    repo.save_report(weekly_summary, "weekly_report_summary.json")
    print("Created weekly_report_summary.json")

    # monthly
    start_date_monthly = "2026-04-01"
    monthly_summary = summary_service.generate_monthly_summary(start_date_monthly, end_date_str)
    repo.save_report(monthly_summary, "monthly_report_summary.json")
    print("Created monthly_report_summary.json")

if __name__ == "__main__":
    main()

