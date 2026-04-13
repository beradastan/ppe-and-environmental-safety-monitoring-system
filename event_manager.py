# -*- coding: utf-8 -*-
"""
event_manager.py
================
Frame-bazlı alarm sonuçlarını event-bazlı mantığa dönüştür.
Aynı alarm tekrar tekrar oluştuğunda yeni event sayma.
Alarm başladığında, değiştiğinde, sona erdiğinde event üret.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime


@dataclass
class AlarmSignature:
    """Bir andaki alarm durumunun hash'i"""
    helmet_violation_count: int
    vest_violation_count: int
    fire_detected: bool
    fire_confidence: float

    def __hash__(self):
        return hash((self.helmet_violation_count, self.vest_violation_count,
                    self.fire_detected, round(self.fire_confidence, 2)))

    def __eq__(self, other):
        if not isinstance(other, AlarmSignature):
            return False
        # Tolerans: count farkları <= 1
        h_ok = abs(self.helmet_violation_count - other.helmet_violation_count) <= 1
        v_ok = abs(self.vest_violation_count - other.vest_violation_count) <= 1
        f_ok = self.fire_detected == other.fire_detected
        c_ok = abs(self.fire_confidence - other.fire_confidence) <= 0.05
        return h_ok and v_ok and f_ok and c_ok


@dataclass
class ActiveEvent:
    """Aktif alarm event'i"""
    event_id: str
    start_time: float
    last_seen: float
    alarm_signature: AlarmSignature
    repeat_count: int = 1
    status: str = "active"  # new, update, active, resolved
    change_reason: str = "initial_alarm"

    @property
    def duration_sec(self) -> float:
        return self.last_seen - self.start_time

    def is_same_alarm(self, new_signature: AlarmSignature) -> bool:
        """Yeni signature aynı alarm mı?"""
        return self.alarm_signature == new_signature

    def is_alarm_changed(self, new_signature: AlarmSignature) -> bool:
        """Aynı alarm mı ama biraz değişmiş mi?"""
        if not self.is_same_alarm(new_signature):
            return False
        # Detaylar değişti mi?
        h_changed = self.alarm_signature.helmet_violation_count != new_signature.helmet_violation_count
        v_changed = self.alarm_signature.vest_violation_count != new_signature.vest_violation_count
        c_changed = abs(self.alarm_signature.fire_confidence - new_signature.fire_confidence) > 0.05
        return h_changed or v_changed or c_changed


class EventManager:
    """
    Frame-bazlı CNN sonuçlarını event-bazlı alarmlara dönüştür.

    Kullanım:
        manager = EventManager(timeout_sec=10, event_id_prefix="evt")

        event_info = manager.process_frame(helmet_result, vest_result, fire_result)
        if event_info["event_status"] in ["new", "update", "resolved"]:
            # Kaydet
            pass
    """

    def __init__(self, timeout_sec: float = 10.0, event_id_prefix: str = "evt"):
        self.timeout_sec = timeout_sec
        self.event_id_prefix = event_id_prefix
        self.event_counter = 0

        self.active_event: Optional[ActiveEvent] = None
        self.last_frame_had_alarm = False

    def _get_next_event_id(self) -> str:
        """Event ID üret"""
        self.event_counter += 1
        return f"{self.event_id_prefix}_{self.event_counter:04d}"

    def _extract_signature(self, helmet_result: Dict, vest_result: Dict,
                          fire_result: Dict) -> AlarmSignature:
        """CNN sonuçlarından alarm signature üret"""
        h_viol = helmet_result.get("warning_count", 0)
        v_viol = vest_result.get("warning_count", 0)
        f_detect = fire_result.get("detection_count", 0) > 0
        f_conf = 0.0

        if f_detect and fire_result.get("detections"):
            f_conf = fire_result["detections"][0].get("confidence", 0.0)

        return AlarmSignature(
            helmet_violation_count=h_viol,
            vest_violation_count=v_viol,
            fire_detected=f_detect,
            fire_confidence=f_conf
        )

    def _has_any_alarm(self, signature: AlarmSignature) -> bool:
        """Alarm var mı?"""
        return (signature.helmet_violation_count > 0 or
                signature.vest_violation_count > 0 or
                signature.fire_detected)

    def process_frame(self, helmet_result: Dict, vest_result: Dict,
                     fire_result: Dict) -> Dict:
        """
        Frame işle ve event bilgisini döndür.

        Returns:
            {
                "event_id": str,
                "event_status": str,  # "no_alarm", "new", "update", "active", "resolved"
                "alarm_signature": AlarmSignature,
                "repeat_count": int,
                "duration_sec": float,
                "change_reason": str,
                "should_save": bool,
                ...other fields...
            }
        """
        current_time = time.time()
        signature = self._extract_signature(helmet_result, vest_result, fire_result)
        has_alarm = self._has_any_alarm(signature)

        result = {
            "event_id": None,
            "event_status": "no_alarm",
            "alarm_signature": signature,
            "repeat_count": 0,
            "duration_sec": 0.0,
            "change_reason": None,
            "should_save": False,
        }

        # ── DURUM 1: Aktif event yok ────────────────────────────
        if self.active_event is None:
            if has_alarm:
                # Yeni alarm başladı
                self.active_event = ActiveEvent(
                    event_id=self._get_next_event_id(),
                    start_time=current_time,
                    last_seen=current_time,
                    alarm_signature=signature,
                    repeat_count=1,
                    status="new",
                    change_reason="initial_alarm"
                )
                result.update({
                    "event_id": self.active_event.event_id,
                    "event_status": "new",
                    "repeat_count": 1,
                    "duration_sec": 0.0,
                    "change_reason": "initial_alarm",
                    "should_save": True,
                })
                self.last_frame_had_alarm = True
            else:
                # Alarm yok, event yok → hiçbir şey yapma
                self.last_frame_had_alarm = False

        # ── DURUM 2: Aktif event var ────────────────────────────
        else:
            if has_alarm:
                # Alarm hala var
                if self.active_event.is_same_alarm(signature):
                    # Aynı alarm, frame tekrarı
                    self.active_event.repeat_count += 1
                    self.active_event.last_seen = current_time
                    self.active_event.status = "active"
                    result.update({
                        "event_id": self.active_event.event_id,
                        "event_status": "active",
                        "repeat_count": self.active_event.repeat_count,
                        "duration_sec": self.active_event.duration_sec,
                        "change_reason": "repeat",
                        "should_save": False,
                    })
                    self.last_frame_had_alarm = True

                elif self.active_event.is_alarm_changed(signature):
                    # Aynı alarm ama detaylar değişti
                    old_signature = self.active_event.alarm_signature
                    self.active_event.alarm_signature = signature
                    self.active_event.repeat_count += 1
                    self.active_event.last_seen = current_time
                    self.active_event.status = "update"

                    # Değişim sebebini belirle
                    h_diff = signature.helmet_violation_count - old_signature.helmet_violation_count
                    v_diff = signature.vest_violation_count - old_signature.vest_violation_count

                    if h_diff > 0:
                        change_reason = f"helmet_violation_increased_{h_diff}"
                    elif h_diff < 0:
                        change_reason = f"helmet_violation_decreased_{-h_diff}"
                    elif v_diff > 0:
                        change_reason = f"vest_violation_increased_{v_diff}"
                    elif v_diff < 0:
                        change_reason = f"vest_violation_decreased_{-v_diff}"
                    else:
                        change_reason = "fire_severity_changed"

                    self.active_event.change_reason = change_reason
                    result.update({
                        "event_id": self.active_event.event_id,
                        "event_status": "update",
                        "repeat_count": self.active_event.repeat_count,
                        "duration_sec": self.active_event.duration_sec,
                        "change_reason": change_reason,
                        "should_save": True,
                    })
                    self.last_frame_had_alarm = True

                else:
                    # Tamamen yeni alarm (eski event kapatılacak)
                    # Eski event'i kayıt ettirme (caller yapacak)
                    # Yeni event başlat
                    self.active_event = ActiveEvent(
                        event_id=self._get_next_event_id(),
                        start_time=current_time,
                        last_seen=current_time,
                        alarm_signature=signature,
                        repeat_count=1,
                        status="new",
                        change_reason="new_alarm_started"
                    )
                    result.update({
                        "event_id": self.active_event.event_id,
                        "event_status": "new",
                        "repeat_count": 1,
                        "duration_sec": 0.0,
                        "change_reason": "new_alarm_started",
                        "should_save": True,
                    })
                    self.last_frame_had_alarm = True

            else:
                # Alarm bitmiş, event var
                # Timeout kontrol et
                if current_time - self.active_event.last_seen > self.timeout_sec:
                    # Event sona ermiş
                    event_copy = self.active_event
                    event_copy.status = "resolved"
                    result.update({
                        "event_id": event_copy.event_id,
                        "event_status": "resolved",
                        "repeat_count": event_copy.repeat_count,
                        "duration_sec": event_copy.duration_sec,
                        "change_reason": "alarm_resolved",
                        "should_save": True,
                        "resolved_event": event_copy,
                    })
                    self.active_event = None
                    self.last_frame_had_alarm = False
                else:
                    # Henüz timeout olmadı, bekleme durumunda
                    result.update({
                        "event_id": self.active_event.event_id,
                        "event_status": "active",
                        "repeat_count": self.active_event.repeat_count,
                        "duration_sec": self.active_event.duration_sec,
                        "change_reason": "waiting_for_timeout",
                        "should_save": False,
                    })

        return result

    def get_active_event(self) -> Optional[ActiveEvent]:
        """Aktif event'i döndür"""
        return self.active_event

    def force_resolve_event(self) -> Optional[ActiveEvent]:
        """Aktif event'i zorla kapat (video sonunda)"""
        if self.active_event:
            self.active_event.status = "resolved"
            event_copy = self.active_event
            self.active_event = None
            return event_copy
        return None


