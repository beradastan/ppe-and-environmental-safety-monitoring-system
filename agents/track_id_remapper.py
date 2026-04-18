# -*- coding: utf-8 -*-
"""
track_id_remapper.py
====================
ByteTrack occlusion sonrası ID switch düzeltici.

Problem:
    İki kişi yan yana geçerken ByteTrack biri "kaybedebilir". Kişi yeniden
    görününce ByteTrack ona yeni bir ID atar (örn. 5 → 17). Bu:
      - Alarm geçmişinin bozulmasına
      - Per-kişi ihlal sürelerinin sıfırlanmasına
      - Kişinin PPE durumunun "yeni kişi" gibi değerlendirmesine
    yol açar.

Çözüm:
    Kaybolan track'in son bilinen bbox'ı ve zamanı "lost pool"da saklanır.
    Yeni bir track_id ortaya çıktığında, bbox merkezi "lost pool"daki
    yakın bir entry ile eşleştirilir (merkez mesafesi + boyut benzerliği).
    Eşleşme varsa yeni ID eski ID olarak remap edilir.

Entegrasyon:
    PersonTrackingAgent.detect() çıktısını bu sınıfın update() metoduna
    geçirmek yeterlidir. Dışarıdan görülen arayüz değişmez; sadece
    track_id'ler stabil kalır.

Kısıtlamalar:
    - Kameradan uzak ve küçük görünen kişilerde (bbox < MIN_AREA_PX)
      remap devre dışı bırakılır (gürültüye karşı koruma).
    - Aynı anda iki yeni track, aynı lost entry ile eşleşirse daha yakın
      olan kazanır (greedy matching).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any


class TrackIDRemapper:
    """
    Kısa süreli occlusion sonrası ByteTrack ID switch'lerini düzeltir.

    Parametreler
    ------------
    max_center_dist : float
        Lost track merkezi ile yeni track merkezi arasındaki maksimum
        piksel mesafesi. Bunu ekranın ~%10'u kadar ayarlamak iyi başlangıç
        noktasıdır (1280 px genişlik → ~128 px).
    max_height_ratio : float
        İki bbox yüksekliğinin birbirine oranı. Ratio > bu eşik ise
        muhtemelen farklı mesafedeki iki kişidir → remap yok.
    max_lost_sec : float
        Lost track'in havuzda tutulacağı maksimum süre.
        ByteTrack track_buffer süresiyle örtüşmeli (track_buffer / fps).
    min_area_px : float
        Çok küçük bbox'lar (gürültü) için remap devre dışı.
    """

    def __init__(
        self,
        max_center_dist: float = 120.0,
        max_height_ratio: float = 1.6,
        max_lost_sec: float = 3.0,
        min_area_px: float = 1000.0,
    ) -> None:
        self.max_center_dist  = max_center_dist
        self.max_height_ratio = max_height_ratio
        self.max_lost_sec     = max_lost_sec
        self.min_area_px      = min_area_px

        # canonical_id → {"bbox": [x1,y1,x2,y2], "time": float}
        self._lost: dict[int, dict] = {}

        # raw ByteTrack id → canonical (stabil) id
        self._id_map: dict[int, int] = {}

        # canonical_id → son bilinen bbox (aktif track'ler için)
        self._last_bbox: dict[int, list] = {}

        self.logger = logging.getLogger("TrackIDRemapper")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        PersonTrackingAgent.detect() çıktısını alır; track_id'leri
        stabil hale getirerek aynı formatta döndürür.

        Args:
            tracked: [{"track_id": int, "bbox": [...], ...}, ...]

        Returns:
            Aynı liste ama track_id'ler remap edilmiş olabilir.
        """
        now = time.monotonic()
        self._expire_lost(now)

        result: list[dict[str, Any]] = []
        claimed_canonical: set[int] = set()

        for p in tracked:
            raw_id = p["track_id"]
            bbox   = p["bbox"]

            canonical = self._id_map.get(raw_id)

            if canonical is None:
                # ByteTrack'in ilk kez gördüğü ya da yeni atadığı bir ID
                canonical = self._try_recover(bbox, now, claimed_canonical)
                if canonical is not None:
                    # Başarıyla eski kimliğe bağlandı
                    self.logger.debug(
                        f"ID recovery: raw={raw_id} → canonical={canonical}  "
                        f"(bbox merkezi eşleşti)"
                    )
                    del self._lost[canonical]
                else:
                    # Gerçekten yeni bir kişi
                    canonical = raw_id

                self._id_map[raw_id] = canonical

            claimed_canonical.add(canonical)
            self._last_bbox[canonical] = bbox
            result.append({**p, "track_id": canonical})

        # Artık görünmeyen canonical ID'leri lost pool'a al
        active_canonical = claimed_canonical
        for cid in list(self._last_bbox.keys()):
            if cid not in active_canonical:
                self._lost[cid] = {
                    "bbox": self._last_bbox.pop(cid),
                    "time": now,
                }

        # Uzun süredir ne aktif ne lost olan raw id'leri temizle
        self._cleanup_id_map(active_canonical)

        return result

    def reset(self) -> None:
        """Yeni video açılırken state'i temizle."""
        self._lost.clear()
        self._id_map.clear()
        self._last_bbox.clear()

    # ------------------------------------------------------------------
    # İç metodlar
    # ------------------------------------------------------------------

    def _expire_lost(self, now: float) -> None:
        """Süresi dolan lost entry'leri temizle."""
        expired = [
            cid for cid, rec in self._lost.items()
            if now - rec["time"] > self.max_lost_sec
        ]
        for cid in expired:
            del self._lost[cid]

    def _try_recover(
        self,
        bbox: list,
        now: float,
        claimed: set[int],
    ) -> int | None:
        """
        Yeni bbox için lost pool'dan en iyi eşleşmeyi döndürür.
        Eşleşme bulunamazsa None döner.
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < self.min_area_px:
            return None  # Çok küçük — gürültüye karşı koruma

        best_id:    int | None = None
        best_dist:  float      = self.max_center_dist

        cx_new = (bbox[0] + bbox[2]) / 2
        cy_new = (bbox[1] + bbox[3]) / 2
        h_new  = bbox[3] - bbox[1]

        for cid, rec in self._lost.items():
            if cid in claimed:
                continue

            lb = rec["bbox"]
            cx_l = (lb[0] + lb[2]) / 2
            cy_l = (lb[1] + lb[3]) / 2
            h_l  = lb[3] - lb[1]

            # Boyut benzerliği kontrolü (çok farklı boylar → farklı kişi)
            if h_l > 0 and h_new > 0:
                ratio = max(h_new, h_l) / min(h_new, h_l)
                if ratio > self.max_height_ratio:
                    continue

            dist = math.hypot(cx_new - cx_l, cy_new - cy_l)
            if dist < best_dist:
                best_dist = dist
                best_id   = cid

        return best_id

    def _cleanup_id_map(self, active_canonical: set[int]) -> None:
        """
        raw → canonical mapping tablosundan artık işe yaramaz
        kayıtları temizler (bellek sızıntısını önler).
        """
        live_canonical = active_canonical | set(self._lost.keys())
        stale = [
            raw for raw, cid in self._id_map.items()
            if cid not in live_canonical
        ]
        for raw in stale:
            del self._id_map[raw]
