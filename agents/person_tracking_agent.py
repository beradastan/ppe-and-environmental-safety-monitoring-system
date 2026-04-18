# -*- coding: utf-8 -*-
"""
PersonTrackingAgent
===================
YOLO11n + ByteTrack ile kişi tespiti ve stabil takibi.

Sorumluluk:
  - Frame içindeki kişileri tespit eder (COCO person sınıfı)
  - ByteTrack ile kareler arasında stabil track_id atar
  - PPE ile hiç ilgilenmez; sadece tracking çıktısı üretir

Örnek çıktı:
    [
        {"track_id": 7,  "bbox": [100, 50, 220, 410], "confidence": 0.92, "is_confirmed": True},
        {"track_id": 12, "bbox": [300, 60, 430, 420], "confidence": 0.88, "is_confirmed": True},
    ]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class PersonTrackingAgent:
    """
    YOLO11n + ByteTrack tabanlı kişi takip ajanı.

    Ultralytics'in dahili track() API'sini kullanır;
    bu sayede ByteTrack yönetimi tamamen Ultralytics'e bırakılır
    ve kareler arasında stabil ID elde edilir.
    """

    # COCO veri setinde "person" sınıfı
    PERSON_CLASS_ID: int = 0

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.40,
        iou: float = 0.50,
        device: str = "cpu",
        min_hits: int = 3,
        remap_ids: bool = True,
        remap_max_center_dist: float = 120.0,
        remap_max_lost_sec: float = 3.0,
    ) -> None:
        """
        Args:
            model_path             : YOLO11n model dosyası (yolo11n.pt proje kökünde olmalı).
            tracker                : Ultralytics tracker konfigürasyonu ("bytetrack.yaml").
            confidence             : Tespit güven eşiği.
            iou                    : NMS IoU eşiği.
            device                 : "cpu" veya "cuda".
            min_hits               : Track ID'nin "confirmed" sayılması için gereken minimum frame sayısı.
            remap_ids              : Occlusion sonrası ID switch recovery'yi etkinleştir.
            remap_max_center_dist  : Recovery için maksimum bbox merkez mesafesi (piksel).
            remap_max_lost_sec     : Kaybolan track'i recovery havuzunda tutma süresi (saniye).
        """
        self.tracker = tracker
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.min_hits = min_hits
        self.logger = logging.getLogger("PersonTrackingAgent")

        model_file = Path(model_path)
        if not model_file.exists():
            self.logger.warning(
                f"Model dosyası bulunamadı: {model_path!r}. "
                "Ultralytics otomatik indirmeyi deneyecek."
            )

        self.model: YOLO = YOLO(str(model_path))

        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.device = "cpu"  # fallback

        # Track başına hit sayacı: track_id → kaç frame görüldü
        self._hit_counts: dict[int, int] = {}

        # ID switch recovery
        self._remapper = None
        if remap_ids:
            from agents.track_id_remapper import TrackIDRemapper
            self._remapper = TrackIDRemapper(
                max_center_dist=remap_max_center_dist,
                max_lost_sec=remap_max_lost_sec,
            )

        self.logger.info(
            f"PersonTrackingAgent hazır | model={model_path} tracker={tracker}"
        )

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Frame içindeki kişileri tespit et ve takip et.

        Args:
            frame: BGR formatında NumPy array (OpenCV frame).

        Returns:
            Tracked kişi listesi. Her eleman:
            {
                "track_id"   : int   – ByteTrack tarafından atanan stabil ID,
                "bbox"       : [x1, y1, x2, y2]  – piksel koordinatları,
                "confidence" : float – tespit güveni,
                "is_confirmed": bool – min_hits eşiğine ulaştı mı,
            }
        """
        try:
            results = self.model.track(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                classes=[self.PERSON_CLASS_ID],
                tracker=self.tracker,
                persist=True,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.logger.error(f"Tracking hatası: {exc}")
            return []

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        tracked: list[dict[str, Any]] = []

        for box in boxes:
            # ByteTrack henüz ID atamadıysa atla
            if box.id is None:
                continue

            track_id: int = int(box.id[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf: float = float(box.conf[0])

            # Hit sayacını güncelle ve onay durumunu belirle
            self._hit_counts[track_id] = self._hit_counts.get(track_id, 0) + 1
            is_confirmed: bool = self._hit_counts[track_id] >= self.min_hits

            tracked.append(
                {
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "is_confirmed": is_confirmed,
                }
            )

        if self._remapper is not None:
            tracked = self._remapper.update(tracked)

        return tracked

    def reset(self) -> None:
        """
        Tracker state'ini sıfırla.
        Yeni bir video dosyası açılırken çağrılmalıdır.
        """
        self._hit_counts.clear()
        if self._remapper is not None:
            self._remapper.reset()
        # Ultralytics dahili tracker buffer'ını temizle
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None
        self.logger.debug("PersonTrackingAgent sıfırlandı.")
