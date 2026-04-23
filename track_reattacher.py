# -*- coding: utf-8 -*-
"""
track_reattacher.py
===================
ByteTrack ID fragmentasyonunu yumuşatan kararlı kişi kimliği katmanı.

ByteTrack kısa oklüzyon sonrası aynı kişiye yeni raw_tid atar.
Bu modül, kaybolan raw_tid'leri belirli bir süre belleğe alır ve
yeni gelen raw_tid'i spatial benzerliğe göre eski kararlı kimliğe bağlar.

Kullanım:
    from track_reattacher import TrackReattacher

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

REATTACH_WINDOW_SEC: float = 2.0   # bu kadar süre sonra kaybolan track silinir
MAX_CENTER_DIST_PX:  int   = 80    # merkez mesafesi eşiği: bu px'in altındaysa eşleştir
MIN_SIZE_RATIO:      float = 0.50  # bbox boyutu oranı: çok farklı boyut → eşleştirme


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _size(box: list[int]) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1])) ** 0.5


def _dist(cx1: tuple, cx2: tuple) -> float:
    return ((cx1[0] - cx2[0]) ** 2 + (cx1[1] - cx2[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

@dataclass
class _MemoryEntry:
    stable_pid: int
    last_box:   list[int]
    last_seen:  float


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
        # raw_tid → stable_pid (aktif track'lar)
        self._active:  dict[int, int]         = {}
        # stable_pid → MemoryEntry (kayıp track'lar, reattach penceresi içinde)
        self._memory:  dict[int, _MemoryEntry] = {}

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

        # 1. Kayıp track'ları belleğe al (aktif → memory)
        lost = set(self._active) - current_raw_ids
        for raw_tid in lost:
            stable_pid = self._active.pop(raw_tid)
            # Önceki last_box'u belleğe yaz (varsa)
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

        # 3. Yeni (görülmemiş) raw_tid'leri reattach veya yeni stable_pid ver
        new_raw_ids = current_raw_ids - set(self._active)
        for det in raw_dets:
            raw_tid = det["tid"]
            if raw_tid in self._active:
                continue  # zaten aktif
            matched_pid = self._try_reattach(det["box"], now)
            if matched_pid is not None:
                self._active[raw_tid] = matched_pid
                del self._memory[matched_pid]
            else:
                new_pid = self._next_pid
                self._next_pid += 1
                self._active[raw_tid] = new_pid

        # 4. Süresi dolmuş memory'yi temizle
        expired = [pid for pid, entry in self._memory.items()
                   if now - entry.last_seen >= self._window]
        for pid in expired:
            del self._memory[pid]

        return dict(self._active)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _try_reattach(self, box: list[int], now: float) -> int | None:
        """Bellek içinde en yakın uygun eski stable_pid'i döndür; yoksa None."""
        cx, cy = _center(box)
        sz     = _size(box)
        best_pid:  int | None = None
        best_dist: float      = float("inf")

        for pid, entry in self._memory.items():
            if now - entry.last_seen >= self._window:
                continue
            ex, ey = _center(entry.last_box)
            es     = _size(entry.last_box)
            dist   = _dist((cx, cy), (ex, ey))
            if dist >= self._max_dist:
                continue
            ratio = min(sz, es) / max(sz, es)
            if ratio < self._min_ratio:
                continue
            if dist < best_dist:
                best_dist = dist
                best_pid  = pid

        return best_pid

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "active_tracks": len(self._active),
            "memory_entries": len(self._memory),
            "next_pid": self._next_pid,
        }
