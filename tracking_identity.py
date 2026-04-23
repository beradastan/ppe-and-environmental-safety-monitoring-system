# -*- coding: utf-8 -*-
"""
tracking_identity.py
====================
Multi-signal scoring kararlı kişi kimliği katmanı.

Ham ByteTrack ID'lerini (raw_tid) kararlı kimliğe (stable_pid) eşler.
80 px sabit eşik yerine şunları birlikte değerlendirir:
  - merkez yakınlığı   (dinamik eşik: kişi boyuna orantılı)
  - bbox alan benzerliği
  - bbox aspect ratio benzerliği
  - zaman benzerliği
  - PPE imzası benzerliği
  - hız tahmini (velocity prediction)

Kullanım:
    from tracking_identity import TrackReattacher

    reattacher = TrackReattacher(max_gap_frames=60)

    # Her frame'de:
    stable_map = reattacher.update(detections)
    # detections: [{"tid": int, "box": [x1,y1,x2,y2],
    #               "ppe_signature": dict (opsiyonel)}, ...]
    # stable_map : {raw_tid: stable_pid}
"""
from __future__ import annotations

import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Varsayılan parametreler
# ---------------------------------------------------------------------------

REATTACH_WINDOW_SEC   = 2.0    # saniye bazlı bellek penceresi
MAX_GAP_FRAMES        = 60     # ~2 s @ 30 fps
DISTANCE_SCALE        = 0.35   # dinamik eşik: max(min_dist, scale * bbox_yüksekliği)
MIN_DISTANCE_PX       = 40.0
MIN_AREA_SIMILARITY   = 0.45
MIN_ASPECT_SIMILARITY = 0.60
MIN_REATTACH_SCORE    = 0.70

_DEFAULT_WEIGHTS = {
    "center": 0.40,
    "area":   0.25,
    "aspect": 0.15,
    "time":   0.15,
    "ppe":    0.05,
}


# ---------------------------------------------------------------------------
# Geometri yardımcıları
# ---------------------------------------------------------------------------

def bbox_center(box: list) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_area(box: list) -> float:
    return max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))


def bbox_aspect(box: list) -> float:
    """Genişlik / yükseklik oranı."""
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    return w / h


def bbox_size(box: list) -> float:
    """Geriye dönük uyumluluk için alan'ın karekökü."""
    return bbox_area(box) ** 0.5


def center_dist(c1: tuple, c2: tuple) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# internal legacy aliases
_center = bbox_center
_size   = bbox_size
_dist   = center_dist


def _ppe_similarity(sig_a: dict, sig_b: dict) -> float:
    """
    İki PPE imzasının benzerlik skoru [0, 1].
    Bilinmeyen ('unknown') etiketler atlanır; veri yoksa 0.5 (nötr) döner.
    """
    if not sig_a or not sig_b:
        return 0.5
    known_pairs = [
        (sig_a[k], sig_b[k])
        for k in set(sig_a) & set(sig_b)
        if sig_a[k] != "unknown" and sig_b[k] != "unknown"
    ]
    if not known_pairs:
        return 0.5
    matches = sum(1 for a, b in known_pairs if a == b)
    return matches / len(known_pairs)


# ---------------------------------------------------------------------------
# Bellek girişi
# ---------------------------------------------------------------------------

@dataclass
class _MemoryEntry:
    stable_pid:      int
    last_bbox:       list
    last_center:     tuple
    prev_center:     object          # tuple | None — hız tahmini için
    last_area:       float
    last_aspect:     float
    last_seen_frame: int
    last_seen_time:  float
    ppe_signature:   dict


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class TrackReattacher:
    """
    Çok sinyalli skor tabanlı raw_tid → stable_pid eşleyici.

    Sabit mesafe eşiği yerine ağırlıklı skor üretir; alan, aspect ratio,
    zaman ve PPE imzası birlikte değerlendirilir.  Hız tahmini sayesinde
    kısa oklüzyonlarda konumu öngörür.
    """

    def __init__(
        self,
        reattach_window_sec   : float = REATTACH_WINDOW_SEC,
        max_gap_frames        : int   = MAX_GAP_FRAMES,
        distance_scale        : float = DISTANCE_SCALE,
        min_distance_px       : float = MIN_DISTANCE_PX,
        min_area_similarity   : float = MIN_AREA_SIMILARITY,
        min_aspect_similarity : float = MIN_ASPECT_SIMILARITY,
        min_reattach_score    : float = MIN_REATTACH_SCORE,
        score_weights         : dict  = None,
    ) -> None:
        self._window      = reattach_window_sec
        self._max_gap     = max_gap_frames
        self._dist_scale  = distance_scale
        self._min_dist    = min_distance_px
        self._min_area    = min_area_similarity
        self._min_aspect  = min_aspect_similarity
        self._min_score   = min_reattach_score
        self._weights     = score_weights or dict(_DEFAULT_WEIGHTS)

        self._next_pid    : int = 1
        self._frame       : int = 0

        # raw_tid → stable_pid (aktif track'lar)
        self._active     : dict[int, int]          = {}
        # raw_tid → konum bilgisi (aktif track'ların son durumu)
        self._active_pos : dict[int, dict]         = {}
        # stable_pid → bellek girişi (kayıp ama hâlâ eşleştirilebilir)
        self._memory     : dict[int, _MemoryEntry] = {}

        self._stats = {
            "new_stable_count"     : 0,
            "reattached_count"     : 0,
            "failed_reattach_count": 0,
            "total_reattach_score" : 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_dets: list[dict]) -> dict[int, int]:
        """
        Parametre:
            raw_dets: [{"tid": int, "box": [x1,y1,x2,y2],
                         "ppe_signature": dict (opsiyonel)}, ...]
        Döner:
            {raw_tid: stable_pid}
        """
        now   = time.monotonic()
        frame = self._frame
        self._frame += 1

        current_raw_ids = {d["tid"] for d in raw_dets}

        # 1. Kaybolan track'ları belleğe al
        for raw_tid in list(set(self._active) - current_raw_ids):
            stable_pid = self._active.pop(raw_tid)
            pos        = self._active_pos.pop(raw_tid, {})
            self._memory[stable_pid] = _MemoryEntry(
                stable_pid      = stable_pid,
                last_bbox       = pos.get("bbox",        [0, 0, 0, 0]),
                last_center     = pos.get("center",      (0.0, 0.0)),
                prev_center     = pos.get("prev_center"),
                last_area       = pos.get("area",        1.0),
                last_aspect     = pos.get("aspect",      1.0),
                last_seen_frame = pos.get("frame",       frame),
                last_seen_time  = pos.get("time",        now),
                ppe_signature   = pos.get("ppe_sig",     {}),
            )

        # 2. Aktif track'ların konumunu güncelle (center → prev_center kayması)
        for det in raw_dets:
            raw_tid = det["tid"]
            if raw_tid not in self._active:
                continue
            box    = det["box"]
            center = bbox_center(box)
            old    = self._active_pos.get(raw_tid, {})
            self._active_pos[raw_tid] = {
                "bbox":        list(box),
                "center":      center,
                "prev_center": old.get("center"),   # hız tahmini için kaydır
                "area":        bbox_area(box),
                "aspect":      bbox_aspect(box),
                "frame":       frame,
                "time":        now,
                "ppe_sig":     det.get("ppe_signature", {}),
            }

        # 3. Yeni raw_tid'lere stable_pid ver
        for det in raw_dets:
            raw_tid = det["tid"]
            if raw_tid in self._active:
                continue
            box     = det["box"]
            ppe_sig = det.get("ppe_signature", {})
            matched_pid, score, had_candidate = self._try_reattach(
                box, frame, now, ppe_sig
            )
            if matched_pid is not None:
                mem = self._memory.pop(matched_pid)
                self._active[raw_tid]     = matched_pid
                self._active_pos[raw_tid] = {
                    "bbox":        list(box),
                    "center":      bbox_center(box),
                    "prev_center": mem.last_center,   # hız sürekliliği
                    "area":        bbox_area(box),
                    "aspect":      bbox_aspect(box),
                    "frame":       frame,
                    "time":        now,
                    "ppe_sig":     ppe_sig,
                }
                self._stats["reattached_count"]     += 1
                self._stats["total_reattach_score"] += score
            else:
                if had_candidate:
                    self._stats["failed_reattach_count"] += 1
                new_pid = self._next_pid
                self._next_pid += 1
                self._active[raw_tid]     = new_pid
                self._active_pos[raw_tid] = {
                    "bbox":        list(box),
                    "center":      bbox_center(box),
                    "prev_center": None,
                    "area":        bbox_area(box),
                    "aspect":      bbox_aspect(box),
                    "frame":       frame,
                    "time":        now,
                    "ppe_sig":     ppe_sig,
                }
                self._stats["new_stable_count"] += 1

        # 4. Süresi dolmuş bellek girişlerini temizle
        expired = [pid for pid, e in self._memory.items()
                   if now - e.last_seen_time >= self._window]
        for pid in expired:
            del self._memory[pid]

        return dict(self._active)

    def stats(self) -> dict:
        rc    = self._stats["reattached_count"]
        score = self._stats["total_reattach_score"]
        return {
            "active_tracks"         : len(self._active),
            "memory_entries"        : len(self._memory),
            "next_pid"              : self._next_pid,
            "new_stable_count"      : self._stats["new_stable_count"],
            "reattached_count"      : rc,
            "failed_reattach_count" : self._stats["failed_reattach_count"],
            "avg_reattach_score"    : round(score / max(1, rc), 3),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _try_reattach(
        self, box: list, frame: int, now: float, ppe_sig: dict
    ) -> tuple:
        """
        Döner: (matched_pid | None, best_score, had_candidate: bool)
        """
        new_center = bbox_center(box)
        new_area   = bbox_area(box)
        new_aspect = bbox_aspect(box)
        h          = max(1, box[3] - box[1])
        dyn_max    = max(self._min_dist, self._dist_scale * h)

        best_pid   = None
        best_score = 0.0
        had_cand   = False

        for pid, entry in self._memory.items():
            # ---- Hard gates ----
            frame_gap = frame - entry.last_seen_frame
            if frame_gap > self._max_gap:
                continue
            if now - entry.last_seen_time >= self._window:
                continue

            area_sim = (min(new_area, entry.last_area)
                        / max(new_area, entry.last_area))
            if area_sim < self._min_area:
                continue

            aspect_sim = (min(new_aspect, entry.last_aspect)
                          / max(new_aspect, entry.last_aspect))
            if aspect_sim < self._min_aspect:
                continue

            # ---- Hız tahmini ----
            if entry.prev_center is not None:
                vx   = entry.last_center[0] - entry.prev_center[0]
                vy   = entry.last_center[1] - entry.prev_center[1]
                pred = (entry.last_center[0] + vx * frame_gap,
                        entry.last_center[1] + vy * frame_gap)
            else:
                pred = entry.last_center

            dist = center_dist(new_center, pred)
            if dist >= dyn_max:
                continue

            had_cand = True

            # ---- Skor bileşenleri ----
            center_sim = max(0.0, 1.0 - dist / dyn_max)
            time_sim   = max(0.0, 1.0 - frame_gap / max(1, self._max_gap))
            ppe_sim    = _ppe_similarity(ppe_sig, entry.ppe_signature)

            w     = self._weights
            score = (w["center"] * center_sim +
                     w["area"]   * area_sim   +
                     w["aspect"] * aspect_sim +
                     w["time"]   * time_sim   +
                     w["ppe"]    * ppe_sim)

            if score > best_score:
                best_score = score
                best_pid   = pid

        if best_pid is not None and best_score >= self._min_score:
            return best_pid, best_score, had_cand

        return None, best_score, had_cand
