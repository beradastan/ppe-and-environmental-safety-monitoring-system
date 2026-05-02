import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

class EventAnalyticsService:
    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events

    def filter_events(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        filtered = []
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return filtered

        for event in self.events:
            try:
                event_dt = datetime.fromisoformat(event.get("created_at")).date()
                if start_dt <= event_dt <= end_dt:
                    filtered.append(event)
            except (ValueError, TypeError):
                continue
        return filtered

    def calculate_risk_score(self, violation_counts: Dict[str, int], total_events: int) -> int:
        score = (
            violation_counts.get("helmet_violation", 0) * 3 +
            violation_counts.get("vest_violation", 0) * 2 +
            violation_counts.get("mask_violation", 0) * 1 +
            violation_counts.get("fire_detected", 0) * 10
        )
        return score

    def determine_risk_level(self, score: int) -> str:
        if score < 20: return "low"
        if score < 50: return "medium"
        if score < 100: return "high"
        return "critical"

    def normalize_risk_score(self, raw_score: int) -> float:
        """Maps raw score to 0-100 preserving risk level semantics:
        low (0-19) → 0-25, medium (20-49) → 25-50, high (50-99) → 50-75, critical (100+) → 75-100"""
        if raw_score < 20:
            return round(raw_score / 20 * 25, 1)
        elif raw_score < 50:
            return round(25 + (raw_score - 20) / 30 * 25, 1)
        elif raw_score < 100:
            return round(50 + (raw_score - 50) / 50 * 25, 1)
        else:
            return min(100.0, round(75 + (raw_score - 100) / 20, 1))


class ReportSummaryService:
    def __init__(self, analytics_service: EventAnalyticsService):
        self.analytics = analytics_service

    def _compute_location_breakdown(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        loc_dist: Dict[str, Dict] = {}
        for e in events:
            cam = e.get("camera_id") or "unknown"
            zone = e.get("zone") or "Bilinmiyor"
            if cam not in loc_dist:
                loc_dist[cam] = {"camera_id": cam, "zone": zone, "event_count": 0}
            loc_dist[cam]["event_count"] += 1
        return sorted(loc_dist.values(), key=lambda x: x["event_count"], reverse=True)

    def _compute_comparison(self, current_count: int, prev_start: str, prev_end: str) -> Dict[str, Any]:
        prev_events = self.analytics.filter_events(prev_start, prev_end)
        prev_count = len(prev_events)
        if prev_count == 0:
            return {"previous_period_events": 0, "change_percent": None, "trend": "no_data"}
        change_pct = round((current_count - prev_count) / prev_count * 100, 1)
        if change_pct > 5:
            trend = "increasing"
        elif change_pct < -5:
            trend = "decreasing"
        else:
            trend = "stable"
        return {"previous_period_events": prev_count, "change_percent": change_pct, "trend": trend}

    def _aggregate_data(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_events = len(events)
        if total_events == 0:
            return self._empty_aggregation()

        violation_counts = {"helmet_violation": 0, "vest_violation": 0, "mask_violation": 0, "fire_detected": 0}
        total_dur = 0
        max_dur = 0
        min_dur = float('inf')
        total_repeat = 0
        multi_violation_count = 0

        for e in events:
            v_count = 0
            if e.get("helmet_violation"):
                violation_counts["helmet_violation"] += 1
                v_count += 1
            if e.get("vest_violation"):
                violation_counts["vest_violation"] += 1
                v_count += 1
            if e.get("mask_violation"):
                violation_counts["mask_violation"] += 1
                v_count += 1
            if e.get("fire_detected"):
                violation_counts["fire_detected"] += 1
                v_count += 1

            if v_count > 1:
                multi_violation_count += 1

            dur = e.get("duration_sec", 0)
            total_dur += dur
            if dur > max_dur: max_dur = dur
            if dur < min_dur: min_dur = dur

            total_repeat += e.get("repeat_count", 0)

        risk_score = self.analytics.calculate_risk_score(violation_counts, total_events)
        risk_level = self.analytics.determine_risk_level(risk_score)
        normalized_score = self.analytics.normalize_risk_score(risk_score)

        return {
            "total_events": total_events,
            "violation_counts": violation_counts,
            "multi_violation_event_count": multi_violation_count,
            "location_breakdown": self._compute_location_breakdown(events),
            "duration_summary": {
                "average_duration_sec": round(total_dur / total_events, 2),
                "max_duration_sec": round(max_dur, 2),
                "min_duration_sec": round(min_dur, 2) if min_dur != float('inf') else 0
            },
            "repeat_summary": {
                "total_repeat_count": total_repeat,
                "average_repeat_count": round(total_repeat / total_events, 2)
            },
            "risk_summary": {
                "risk_score": risk_score,
                "normalized_score": normalized_score,
                "risk_level": risk_level
            }
        }

    def _empty_aggregation(self) -> Dict[str, Any]:
        return {
            "total_events": 0,
            "violation_counts": {
                "helmet_violation": 0,
                "vest_violation": 0,
                "mask_violation": 0,
                "fire_detected": 0
            },
            "multi_violation_event_count": 0,
            "location_breakdown": [],
            "duration_summary": {
                "average_duration_sec": 0,
                "max_duration_sec": 0,
                "min_duration_sec": 0
            },
            "repeat_summary": {
                "total_repeat_count": 0,
                "average_repeat_count": 0
            },
            "risk_summary": {
                "risk_score": 0,
                "normalized_score": 0.0,
                "risk_level": "low"
            }
        }

    def generate_daily_summary(self, target_date: str) -> Dict[str, Any]:
        events = self.analytics.filter_events(target_date, target_date)
        agg = self._aggregate_data(events)

        hourly_dist = {}
        for e in events:
            try:
                dt = datetime.fromisoformat(e.get("created_at"))
                hour_str = f"{dt.hour:02d}:00"
                hourly_dist[hour_str] = hourly_dist.get(hour_str, 0) + 1
            except (ValueError, TypeError):\n                pass
        hourly_list = [{"hour": k, "event_count": v} for k, v in sorted(hourly_dist.items())]

        prev_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        comparison = self._compute_comparison(agg["total_events"], prev_date, prev_date)

        return {
            "report_type": "daily",
            "period": {"start_date": target_date, "end_date": target_date},
            "total_events": agg["total_events"],
            "violation_counts": agg["violation_counts"],
            "multi_violation_event_count": agg["multi_violation_event_count"],
            "location_breakdown": agg["location_breakdown"],
            "repeat_summary": agg["repeat_summary"],
            "duration_summary": agg["duration_summary"],
            "hourly_distribution": hourly_list,
            "comparison": comparison,
            "risk_summary": agg["risk_summary"]
        }

    def generate_weekly_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        events = self.analytics.filter_events(start_date, end_date)
        agg = self._aggregate_data(events)

        daily_dist = {}
        for e in events:
            try:
                dt_str = datetime.fromisoformat(e.get("created_at")).strftime("%Y-%m-%d")
                daily_dist[dt_str] = daily_dist.get(dt_str, 0) + 1
            except (ValueError, TypeError):\n                pass
        daily_list = [{"date": k, "event_count": v} for k, v in sorted(daily_dist.items())]
        most_active_day = max(daily_list, key=lambda x: x["event_count"], default={"date": "N/A", "event_count": 0})

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        prev_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        prev_end = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        comparison = self._compute_comparison(agg["total_events"], prev_start, prev_end)

        return {
            "report_type": "weekly",
            "period": {"start_date": start_date, "end_date": end_date},
            "total_events": agg["total_events"],
            "violation_counts": agg["violation_counts"],
            "daily_breakdown": daily_list,
            "most_active_day": most_active_day,
            "multi_violation_event_count": agg["multi_violation_event_count"],
            "location_breakdown": agg["location_breakdown"],
            "duration_summary": agg["duration_summary"],
            "repeat_summary": agg["repeat_summary"],
            "comparison": comparison,
            "risk_summary": agg["risk_summary"]
        }

    def generate_monthly_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        events = self.analytics.filter_events(start_date, end_date)
        agg = self._aggregate_data(events)

        counts = agg["violation_counts"]
        top_violation = max(counts, key=counts.get) if agg["total_events"] > 0 else "N/A"

        weekly_breakdown_dict = {}
        for e in events:
            try:
                dt = datetime.fromisoformat(e.get("created_at")).date()
                year = dt.isocalendar()[0]
                week = dt.isocalendar()[1]
                week_str = f"{year}-W{week}"
                weekly_breakdown_dict[week_str] = weekly_breakdown_dict.get(week_str, 0) + 1
            except (ValueError, TypeError):\n                pass
        weekly_breakdown = [{"week": k, "event_count": v} for k, v in sorted(weekly_breakdown_dict.items())]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        period_days = (end_dt - start_dt).days + 1
        prev_end_dt = start_dt - timedelta(days=1)
        prev_start_dt = prev_end_dt - timedelta(days=period_days - 1)
        comparison = self._compute_comparison(
            agg["total_events"],
            prev_start_dt.strftime("%Y-%m-%d"),
            prev_end_dt.strftime("%Y-%m-%d")
        )

        return {
            "report_type": "monthly",
            "period": {"start_date": start_date, "end_date": end_date},
            "total_events": agg["total_events"],
            "violation_counts": agg["violation_counts"],
            "weekly_breakdown": weekly_breakdown,
            "top_violation_type": top_violation,
            "multi_violation_event_count": agg["multi_violation_event_count"],
            "location_breakdown": agg["location_breakdown"],
            "duration_summary": agg["duration_summary"],
            "repeat_summary": agg["repeat_summary"],
            "comparison": comparison,
            "risk_summary": agg["risk_summary"]
        }

class ReportRepository:
    def save_report(self, report_data: Dict[str, Any], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
