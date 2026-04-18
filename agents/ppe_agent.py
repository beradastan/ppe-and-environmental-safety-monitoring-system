# -*- coding: utf-8 -*-
"""
PPEAgent
========
Hansung-Cho/yolov8-ppe-detection modeliyle PPE tespiti.

Sorumluluk:
  - Frame içinde Hardhat / No-Hardhat ve Safety Vest / No-Safety Vest tespiti yapar.
  - Kişi bazlı eşleştirme YAPMAZ; sadece ham PPE bounding box'larını üretir.
  - Model sınıf isimlerini başlangıçta otomatik parse eder (farklı checkpoint
    isimlendirmelerine karşı dayanıklı).

Örnek çıktı:
    [
        {"label": "Hardhat",        "bbox": [...], "confidence": 0.88,
         "category": "helmet", "status": "present"},
        {"label": "No-Safety Vest", "bbox": [...], "confidence": 0.76,
         "category": "vest",   "status": "missing"},
    ]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Sınıf ismi → (PPE kategorisi, durum) normalizasyon tablosu.
# Hansung modeli genellikle şu isimler kullanır ama küçük farklar olabilir.
# ---------------------------------------------------------------------------
_LABEL_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    # ---- Helmet / Hardhat ----
    "hardhat":            ("helmet", "present"),
    "helmet":             ("helmet", "present"),
    # ---- No-Helmet ----
    "no-hardhat":         ("helmet", "missing"),
    "no hardhat":         ("helmet", "missing"),
    "nohardhat":          ("helmet", "missing"),
    "no-helmet":          ("helmet", "missing"),
    "no helmet":          ("helmet", "missing"),
    "nohelmet":           ("helmet", "missing"),
    # ---- Safety Vest ----
    "safety vest":        ("vest", "present"),
    "safetyvest":         ("vest", "present"),
    "vest":               ("vest", "present"),
    # ---- No-Vest ----
    "no-safety vest":     ("vest", "missing"),
    "no safety vest":     ("vest", "missing"),
    "nosafetyvest":       ("vest", "missing"),
    "no-vest":            ("vest", "missing"),
    "no vest":            ("vest", "missing"),
    "novest":             ("vest", "missing"),
    # ---- Mask (VoxDroid) ----
    "mask":               ("mask", "present"),
    # ---- No-Mask (VoxDroid) ----
    "no-mask":            ("mask", "missing"),
    "no mask":            ("mask", "missing"),
    "nomask":             ("mask", "missing"),
}


def _classify_label(label: str) -> tuple[str, str] | None:
    """
    Ham model etiketi 'label'ı (category, status) çiftine çevirir.
    Eşleşme bulunamazsa None döner (Person, Safety Cone gibi ilgisiz sınıflar).
    """
    return _LABEL_CATEGORY_MAP.get(label.strip().lower())


class PPEAgent:
    """
    VoxDroid Construction Site Safety PPE Detection Agent.

    detect()      → sadece PPE item'larını döndürür (eski API korundu).
    detect_full() → tek geçişte hem Person bbox'larını hem PPE item'larını döndürür.
    """

    def __init__(
        self,
        model_path: str = "models/voxdroid_200epoch_best.pt",
        confidence: float = 0.25,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path : Hansung PPE model dosyası yolu.
            confidence : Tespit güven eşiği.
            device     : "cpu" veya "cuda".

        Raises:
            FileNotFoundError: Model dosyası yoksa.
        """
        self.confidence = confidence
        self.device = device
        self.logger = logging.getLogger("PPEAgent")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"PPE model dosyası bulunamadı: {model_path!r}\n"
                "  → models/ dizininde voxdroid_200epoch_best.pt dosyasının bulunduğundan emin ol."
            )

        self.model: YOLO = YOLO(str(model_path))

        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.device = "cpu"

        # Model sınıf isimlerinden PPE ve Person sınıf ID'lerini tespit et
        self._ppe_class_ids: list[int] = []
        self._class_meta: dict[int, tuple[str, str]] = {}  # id → (category, status)
        self._person_class_ids: list[int] = []
        self._build_class_index()

        self.logger.info(
            f"PPEAgent hazır | model={model_path} | "
            f"PPE sınıfları: {[self.model.names[i] for i in self._ppe_class_ids]} | "
            f"Person sınıfları: {[self.model.names[i] for i in self._person_class_ids]}"
        )

    # ------------------------------------------------------------------
    # İç yardımcılar
    # ------------------------------------------------------------------

    def _build_class_index(self) -> None:
        """
        Model.names sözlüğünü tarayarak PPE ve Person sınıflarını indeksler.
        """
        for class_id, name in self.model.names.items():
            if name.strip().lower() == "person":
                self._person_class_ids.append(class_id)
                continue
            mapping = _classify_label(name)
            if mapping is not None:
                self._ppe_class_ids.append(class_id)
                self._class_meta[class_id] = mapping

        if not self._ppe_class_ids:
            self.logger.warning(
                "Modelde bilinen PPE sınıfları bulunamadı! "
                f"Mevcut sınıflar: {list(self.model.names.values())}\n"
                "  → _LABEL_CATEGORY_MAP tablosuna bu isimleri eklemeyi düşün."
            )

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Frame içindeki PPE item'larını tespit et.

        Args:
            frame: BGR formatında NumPy array.

        Returns:
            PPE tespit listesi. Her eleman:
            {
                "label"      : str   – modelin orijinal sınıf adı,
                "bbox"       : [x1, y1, x2, y2],
                "confidence" : float,
                "category"   : str   – "helmet" veya "vest",
                "status"     : str   – "present" (giyiyor) veya "missing" (giymemiş),
            }
        """
        try:
            results = self.model(
                frame,
                conf=self.confidence,
                classes=self._ppe_class_ids if self._ppe_class_ids else None,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.logger.error(f"PPE tespit hatası: {exc}")
            return []

        if not results or results[0].boxes is None:
            return []

        return self._parse_boxes(results[0].boxes)

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """
        Birden fazla crop'u tek GPU forward pass'te işler.

        Args:
            crops: BGR crop listesi (her biri bir kişiye ait).

        Returns:
            Her crop için detect() formatında PPE tespit listesi.
        """
        if not crops:
            return []
        try:
            results = self.model(
                crops,
                conf=self.confidence,
                classes=self._ppe_class_ids if self._ppe_class_ids else None,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.logger.error(f"Batch PPE tespit hatası: {exc}")
            return [[] for _ in crops]

        return [
            self._parse_boxes(r.boxes) if r.boxes is not None else []
            for r in results
        ]

    def _parse_boxes(self, boxes: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id: int = int(box.cls[0])
            label: str = self.model.names.get(class_id, "unknown")
            conf: float = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            mapping = self._class_meta.get(class_id) or _classify_label(label)
            if mapping is None:
                continue

            category, status = mapping
            detections.append({
                "label"     : label,
                "bbox"      : [x1, y1, x2, y2],
                "confidence": conf,
                "category"  : category,
                "status"    : status,
            })
        return detections

    def detect_full(self, frame: np.ndarray) -> dict[str, list]:
        """
        Tek model geçişinde hem Person bbox'larını hem PPE item'larını döndürür.

        Returns:
            {
                "persons": [
                    {"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float, "is_confirmed": True},
                    ...
                ],
                "ppe": [  # detect() ile aynı format
                    {"label": str, "bbox": [...], "confidence": float, "category": str, "status": str},
                    ...
                ],
            }
        """
        all_class_ids = self._ppe_class_ids + self._person_class_ids

        try:
            results = self.model(
                frame,
                conf=self.confidence,
                classes=all_class_ids if all_class_ids else None,
                verbose=False,
            )
        except Exception as exc:
            self.logger.error(f"detect_full hatası: {exc}")
            return {"persons": [], "ppe": []}

        if not results or results[0].boxes is None:
            return {"persons": [], "ppe": []}

        persons: list[dict] = []
        ppe: list[dict] = []
        person_idx = 0

        for box in results[0].boxes:
            class_id: int = int(box.cls[0])
            label: str = self.model.names.get(class_id, "unknown")
            conf: float = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            if class_id in self._person_class_ids:
                persons.append({
                    "track_id": person_idx,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "is_confirmed": True,
                })
                person_idx += 1
                continue

            mapping = self._class_meta.get(class_id) or _classify_label(label)
            if mapping is None:
                continue

            category, status = mapping
            ppe.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "category": category,
                "status": status,
            })

        return {"persons": persons, "ppe": ppe}
