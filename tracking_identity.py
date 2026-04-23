# -*- coding: utf-8 -*-
"""
tracking_identity.py
====================
Kararlı kişi kimliği katmanı — ByteTrack ID fragmentasyonunu yumuşatır.

ByteTrack kısa oklüzyon sonrası aynı kişiye yeni raw_tid atar.
Bu modül kaybolan raw_tid'leri belirli bir süre belleğe alır ve
yeni gelen raw_tid'i spatial benzerliğe göre eski kararlı kimliğe bağlar.

Kullanım:
    from tracking_identity import TrackReattacher

    reattacher = TrackReattacher()

    # Her frame'de:
    stable_map = reattacher.update(raw_detections)
    # raw_detections: [{"tid": int, "box": [x1,y1,x2,y2]}, ...]
    # stable_map: {raw_tid: stable_pid}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

REATTACH_WINDOW_SEC: float = 2.0
MAX_CENTER_DIST_PX:  int   = 80
MIN_SIZE_RATIO:      float = 0.50


# ---------------------------------------------------------------------------
# Yardımcı geometri fonksiyonları
# ---------------------------------------------------------------------------

def bbox_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_size(box: list[int]) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1])) ** 0.5


def center_dist(c1: tuple, c2: tuple) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# Internal aliases for backwards compat
_center = bbox_center
_size   = bbox_size
_dist   = center_dist


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------

@dataclass
class _MemoryEntry:
    stable_pid: int
    last_box:   list[int]
    last_seen:  float


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class TrackReattacher:
    """
    raw_tid → stable_pid eşlemesi.

    Her frame'de update() çağrılır; dönen dict'ten raw_tid → stable_pid alınır.
    Aynı stable_pid, kişinin video boyunca koruduğu kimliğidir.
    """

    def __init__(
        self,
        reattach_window_sec: float = REATTACH_WINDOW_SEC,
        max_center_dist_px:  int   = MAX_CENTER_DIST_PX,
        min_size_ratio:      float = MIN_SIZE_RATIO,
    ) -> None:
        self._window    = reattach_window_sec
        self._max_dist  = max_center_dist_px
        self._min_ratio = min_size_ratio

        self._next_pid: int = 1
        self._active:  dict[int, int]          = {}   # raw_tid → stable_pid
        self._memory:  dict[int, _MemoryEntry] = {}   # stable_pid → entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_dets: list[dict]) -> dict[int, int]:
        """
        Parametre:
            raw_dets: [{"tid": int, "box": [x1,y1,x2,y2]}, ...]
        Döner:
            {raw_tid: stable_pid} — tüm aktif raw_tid'ler için
        """
        now = time.monotonic()
        current_raw_ids = {d["tid"] for d in raw_dets}

        # 1. Kayıp track'ları belleğe al
        lost = set(self._active) - current_raw_ids
        for raw_tid in lost:
            stable_pid = self._active.pop(raw_tid)
            if stable_pid not in self._memory:
                self._memory[stable_pid] = _MemoryEntry(
                    stable_pid=stable_pid,
                    last_box=[0, 0, 0, 0],
                    last_seen=now,
                )
            self._memory[stable_pid].last_seen = now

        # 2. Aktif track'ların son konumlarını güncelle
        for det in raw_dets:
            raw_tid = det["tid"]
            if raw_tid in self._active:
                stable_pid = self._active[raw_tid]
                if stable_pid in self._memory:
                    self._memory[stable_pid].last_box = list(det["box"])

        # 3. Yeni raw_tid'leri reattach veya yeni stable_pid ver
        for det in raw_dets:
            raw_tid = det["tid"]
            if raw_tid in self._active:
                continue
            matched_pid = self._try_reattach(det["box"], now)
            if matched_pid is not None:
                self._active[raw_tid] = matched_pid
                del self._memory[matched_pid]
            else:
                self._active[raw_tid] = self._next_pid
                self._next_pid += 1

        # 4. Süresi dolmuş memory'yi temizle
        expired = [pid for pid, e in self._memory.items()
                   if now - e.last_seen >= self._window]
        for pid in expired:
            del self._memory[pid]

        return dict(self._active)

    def stats(self) -> dict:
        return {
            "active_tracks":  len(self._active),
            "memory_entries": len(self._memory),
            "next_pid":       self._next_pid,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _try_reattach(self, box: list[int], now: float) -> int | None:
        cx, cy = bbox_center(box)
        sz     = bbox_size(box)
        best_pid:  int | None = None
        best_dist: float      = float("inf")

        for pid, entry in self._memory.items():
            if now - entry.last_seen >= self._window:
                continue
            ex, ey = bbox_center(entry.last_box)
            es     = bbox_size(entry.last_box)
            dist   = center_dist((cx, cy), (ex, ey))
            if dist >= self._max_dist:
                continue
            ratio = min(sz, es) / max(sz, es)
            if ratio < self._min_ratio:
                continue
            if dist < best_dist:
                best_dist = dist
                best_pid  = pid

        return best_pid
