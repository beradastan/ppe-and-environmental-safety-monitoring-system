# -*- coding: utf-8 -*-
"""
event_manager.py
================
İhlal tabanlı event state machine.

Kayıt mantığı:
  - Sadece "new" (ihlal kesinleşti) kaydedilir ve frontend'e gönderilir.
  - "resolved" tamamen dahili — dosyaya/socket'e yansımaz.

Kişi kameradan çıkınca:
  - Son history_frames frame'deki tespitlerin çoğunluğuna bakılır.
  - Çoğunluk ihlalliyse → exit_grace_sec kadar ihlal aktif tutulur.
  - Çoğunluk temizse → hemen silinir.

process_frame() çıktısı:
    {
        "event_id"    : str | None,
        "event_status": "idle" | "new" | "active" | "resolved",
        "repeat_count": int,
        "duration_sec": float,
        "should_save" : bool,   # sadece "new"'da True
        "change_reason": str,
        "signature": {
            "helmet_violation": bool,
            "vest_violation"  : bool,
            "mask_violation"  : bool,
            "fire_detected"   : bool,
        },
        "person_violations": [
            {"track_id": int, "violations": [...], "duration_sec": float},
        ],
    }
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Eşik sabitleri
# ---------------------------------------------------------------------------

DEFAULT_NEW_CONFIRM_SEC:        float = 3.0
DEFAULT_RESOLVED_CONFIRM_SEC:   float = 5.0
DEFAULT_FIRE_CONFIRM_FRAMES:    int   = 2
DEFAULT_FIRE_CLEAR_FRAMES:      int   = 2
DEFAULT_HISTORY_FRAMES:         int   = 8
DEFAULT_EXIT_GRACE_SEC:         float = 2.0
DEFAULT_CONFIRM_GAP_TOLERANCE:  float = 1.0  # onay süresindeki tespit boşluğu toleransı (s)


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------

@dataclass
class PersonViolationRecord:
    track_id:   int
    violations: list[str]
    since:      float


@dataclass(frozen=True)
class EventSignature:
    helmet_violation: bool
    vest_violation:   bool
    mask_violation:   bool
    fire_detected:    bool

    @property
    def has_violation(self) -> bool:
        return (
            self.helmet_violation
            or self.vest_violation
            or self.mask_violation
            or self.fire_detected
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "helmet_violation": self.helmet_violation,
            "vest_violation"  : self.vest_violation,
            "mask_violation"  : self.mask_violation,
            "fire_detected"   : self.fire_detected,
        }


@dataclass
class ActiveEvent:
    event_id:     str
    start_time:   float
    last_seen:    float
    signature:    EventSignature
    repeat_count: int = 1

    @property
    def duration_sec(self) -> float:
        return self.last_seen - self.start_time


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class PersonEventManager:
    """
    İhlal tabanlı event yöneticisi.

    State machine: idle → new → active → resolved → idle
    Sadece "new" kaydedilir. "resolved" dahili temizliktir.
    """

    def __init__(
        self,
        new_confirm_sec:           float = DEFAULT_NEW_CONFIRM_SEC,
        resolved_confirm_sec:      float = DEFAULT_RESOLVED_CONFIRM_SEC,
        fire_confirm_frames:       int   = DEFAULT_FIRE_CONFIRM_FRAMES,
        fire_clear_frames:         int   = DEFAULT_FIRE_CLEAR_FRAMES,
        event_id_prefix:           str   = "evt",
        check_helmet:              bool  = True,
        check_vest:                bool  = True,
        check_mask:                bool  = False,
        history_frames:            int   = DEFAULT_HISTORY_FRAMES,
        exit_grace_sec:            float = DEFAULT_EXIT_GRACE_SEC,
        confirm_gap_tolerance:     float = DEFAULT_CONFIRM_GAP_TOLERANCE,
        persist_violations_on_exit: bool = False,
    ) -> None:
        self.persist_violations_on_exit = persist_violations_on_exit
        self.new_confirm_sec        = new_confirm_sec
        self.resolved_confirm_sec   = resolved_confirm_sec
        self.fire_confirm_frames    = fire_confirm_frames
        self.fire_clear_frames      = fire_clear_frames
        self.event_id_prefix        = event_id_prefix
        self.check_helmet           = check_helmet
        self.check_vest             = check_vest
        self.check_mask             = check_mask
        self.history_frames         = history_frames
        self.exit_grace_sec         = exit_grace_sec
        self.confirm_gap_tolerance  = confirm_gap_tolerance

        self._counter: int               = 0
        self._active:  ActiveEvent | None = None

        self._violation_since:      float | None = None
        self._last_violation_seen:  float | None = None  # onay süresinde son ihlal zamanı
        self._clear_since:          float | None = None

        self._fire_pending:   int  = 0
        self._fire_clear_cnt: int  = 0
        self._fire_confirmed: bool = False

        # Per-person violation state
        self._person_states:     dict[int, PersonViolationRecord] = {}
        # Per-person detection history (son N frame'de ihlal var mıydı)
        self._person_history:    dict[int, deque] = {}
        # Kameradan çıkan kişi → çıkış zamanı (grace period)
        self._person_exit_times: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        persons_with_ppe: list[dict[str, Any]],
        fire_raw: bool = False,
    ) -> dict[str, Any]:
        self._update_person_states(persons_with_ppe)
        self._fire_confirmed = self._update_fire(fire_raw)
        sig = self._build_signature(self._fire_confirmed)

        if self._active is None:
            result = self._handle_idle(sig)
        else:
            result = self._handle_active(sig)

        result["person_violations"] = self._get_person_violations()
        return result

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(self, sig: EventSignature) -> dict[str, Any]:
        now = time.monotonic()

        if not sig.has_violation:
            if self._violation_since is not None:
                # Referans repo mantığı: brief tespit boşluklarında sayacı sıfırlama.
                # Son ihlalden bu yana confirm_gap_tolerance saniyeyi geçmediyse bekle.
                last_seen = self._last_violation_seen or self._violation_since
                if now - last_seen > self.confirm_gap_tolerance:
                    self._violation_since     = None
                    self._last_violation_seen = None
            return self._out(None, "idle", 0, 0.0, False, "no_violation", sig)

        self._last_violation_seen = now
        if self._violation_since is None:
            self._violation_since = now

        elapsed = now - self._violation_since
        if elapsed >= self.new_confirm_sec:
            self._violation_since     = None
            self._last_violation_seen = None
            return self._open_event(sig)

        return self._out(
            None, "idle", 0, 0.0, False,
            f"confirming {elapsed:.1f}s/{self.new_confirm_sec:.1f}s", sig,
        )

    def _handle_active(self, sig: EventSignature) -> dict[str, Any]:
        ev  = self._active
        now = time.monotonic()

        if not sig.has_violation:
            if self._clear_since is None:
                self._clear_since = now
            elapsed = now - self._clear_since
            if elapsed >= self.resolved_confirm_sec:
                return self._resolve(ev, sig)
            ev.last_seen    = now
            ev.repeat_count += 1
            return self._out(
                ev.event_id, "active", ev.repeat_count, ev.duration_sec, False,
                f"confirming_resolved {elapsed:.1f}s/{self.resolved_confirm_sec:.1f}s",
                ev.signature,
            )

        self._clear_since   = None
        ev.last_seen        = now
        ev.repeat_count    += 1
        ev.signature        = sig
        return self._out(ev.event_id, "active", ev.repeat_count, ev.duration_sec, False, "ongoing", sig)

    # ------------------------------------------------------------------
    # Event lifecycle
    # ------------------------------------------------------------------

    def _open_event(self, sig: EventSignature) -> dict[str, Any]:
        self._counter += 1
        now = time.monotonic()
        self._active = ActiveEvent(
            event_id     = f"{self.event_id_prefix}_{self._counter:04d}",
            start_time   = now,
            last_seen    = now,
            signature    = sig,
            repeat_count = 1,
        )
        self._clear_since = None
        return self._out(self._active.event_id, "new", 1, 0.0, True, "initial_violation", sig)

    def _resolve(self, ev: ActiveEvent, sig: EventSignature) -> dict[str, Any]:
        self._active      = None
        self._clear_since = None
        # should_save=False — resolved frontend'e/diske yansımaz
        return self._out(ev.event_id, "resolved", ev.repeat_count, ev.duration_sec, False, "alarm_cleared", sig)

    # ------------------------------------------------------------------
    # Per-person violation state + exit history
    # ------------------------------------------------------------------

    def _update_person_states(self, persons_with_ppe: list[dict[str, Any]]) -> None:
        now = time.monotonic()
        current_ids: set[int] = set()

        for p in persons_with_ppe:
            tid = p.get("track_id")
            if tid is None:
                continue
            current_ids.add(tid)

            # Tekrar göründüyse exit grace'i iptal et
            self._person_exit_times.pop(tid, None)

            # Geçmiş güncelle
            if tid not in self._person_history:
                self._person_history[tid] = deque(maxlen=self.history_frames)
            viols = p.get("violations", [])
            self._person_history[tid].append(bool(viols))

            # Violation state güncelle
            if viols:
                if tid not in self._person_states:
                    self._person_states[tid] = PersonViolationRecord(tid, viols, now)
                else:
                    self._person_states[tid].violations = viols
            else:
                self._person_states.pop(tid, None)

        # Kameradan çıkan kişiler
        for tid in list(self._person_history.keys()):
            if tid in current_ids:
                continue

            if tid not in self._person_exit_times:
                # Yeni çıkış — sadece geçmişe bak (son frame gürültüsünü yok say)
                history = self._person_history[tid]
                violation_ratio = sum(history) / len(history) if history else 0.0

                if violation_ratio >= 0.5:
                    if self.persist_violations_on_exit:
                        # Kişi ihlalle çıktı → kaydı sil değil, sonsuza koru
                        # (aynı track_id uyumlu dönene kadar ya da event kapanana kadar)
                        pass
                    else:
                        # Geçmişin çoğunluğu ihlalliydi → grace period başlat
                        self._person_exit_times[tid] = now
                else:
                    # Geçmişin çoğunluğu temizdi → hemen sil
                    self._person_states.pop(tid, None)
                    del self._person_history[tid]
                continue

            # Grace period doldu mu?
            if now - self._person_exit_times[tid] >= self.exit_grace_sec:
                self._person_states.pop(tid, None)
                self._person_history.pop(tid, None)
                del self._person_exit_times[tid]

    def _get_person_violations(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        # Grace period'dakiler dahil — signature ile tutarlılık için
        return [
            {
                "track_id":     rec.track_id,
                "violations":   rec.violations,
                "duration_sec": round(now - rec.since, 1),
            }
            for rec in self._person_states.values()
        ]

    # ------------------------------------------------------------------
    # Yangın debounce
    # ------------------------------------------------------------------

    def _update_fire(self, raw: bool) -> bool:
        if raw:
            self._fire_pending   += 1
            self._fire_clear_cnt  = 0
            if self._fire_pending >= self.fire_confirm_frames:
                self._fire_confirmed = True
        else:
            self._fire_clear_cnt += 1
            self._fire_pending    = 0
            if self._fire_clear_cnt >= self.fire_clear_frames:
                self._fire_confirmed = False
        return self._fire_confirmed

    # ------------------------------------------------------------------
    # Signature — _person_states'ten türetilir
    # ------------------------------------------------------------------

    def _build_signature(self, fire_detected: bool) -> EventSignature:
        """
        Signature doğrudan _person_states'ten üretilir (exit grace dahil).
        Bool-based: kaç kişi değil, hangi ihlal tipi var.
        """
        helmet = self.check_helmet and any(
            "no_helmet" in rec.violations for rec in self._person_states.values()
        )
        vest = self.check_vest and any(
            "no_vest" in rec.violations for rec in self._person_states.values()
        )
        mask = self.check_mask and any(
            "no_mask" in rec.violations for rec in self._person_states.values()
        )
        return EventSignature(bool(helmet), bool(vest), bool(mask), fire_detected)

    @staticmethod
    def _out(
        event_id:      str | None,
        event_status:  str,
        repeat_count:  int,
        duration_sec:  float,
        should_save:   bool,
        change_reason: str,
        sig:           EventSignature,
    ) -> dict[str, Any]:
        return {
            "event_id"    : event_id,
            "event_status": event_status,
            "repeat_count": repeat_count,
            "duration_sec": round(duration_sec, 1),
            "should_save" : should_save,
            "change_reason": change_reason,
            "signature"   : sig.to_dict(),
        }
