# -*- coding: utf-8 -*-
"""
CropPPEMatcher
==============
Her takip edilen kişi için bbox crop'u alıp PPE tespiti yapar.
IoA eşleştirmesi yoktur: her crop'un PPE tespitleri kesin olarak o kişiye aittir.

Çıktı formatı PPEPersonMatcher ile aynıdır:
    [
        {
            "track_id"     : int | None,
            "person_bbox"  : [x1, y1, x2, y2],   # orijinal frame koordinatları
            "helmet_status": "present" | "missing" | "unknown",
            "vest_status"  : "present" | "missing" | "unknown",
            "mask_status"  : "present" | "missing" | "unknown",
            "violations"   : ["no_helmet", "no_vest", "no_mask"],
            "ppe_dets"     : [...]   # full-frame koordinatlarına dönüştürülmüş PPE bbox'ları
        },
        ...
    ]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agents.ppe_agent import PPEAgent


class CropPPEMatcher:

    def __init__(
        self,
        crop_pad:     float = 0.40,   # bbox genişliği/yüksekliğine göre padding oranı
        check_helmet: bool  = True,
        check_vest:   bool  = True,
        check_mask:   bool  = False,
    ) -> None:
        self.crop_pad     = crop_pad
        self.check_helmet = check_helmet
        self.check_vest   = check_vest
        self.check_mask   = check_mask
        self.logger = logging.getLogger("CropPPEMatcher")

    def match(
        self,
        frame:    np.ndarray,
        persons:  list[dict[str, Any]],
        ppe_agent: "PPEAgent",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Args:
            frame     : Tam BGR frame.
            persons   : PersonTrackingAgent çıktısı (confirmed track'ler).
            ppe_agent : PPEAgent instance'ı.

        Returns:
            (person_results, all_ppe_dets_full_frame)
            - person_results    : event_manager.process_frame() için hazır liste.
            - all_ppe_dets_full_frame : görselleştirme için full-frame bbox'lı PPE listesi.
        """
        fh, fw = frame.shape[:2]
        person_results:  list[dict[str, Any]] = []
        all_ppe_dets:    list[dict[str, Any]] = []

        # 1. Tüm kişiler için crop'ları ve offset'leri hazırla
        crops:   list[np.ndarray]      = []
        offsets: list[tuple[int, int]] = []   # (cx1, cy1) — full-frame'e geri dönüşüm için
        valid_persons: list[dict[str, Any]] = []

        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            bw = x2 - x1
            bh = y2 - y1

            pad_x = bw * self.crop_pad
            pad_y = bh * self.crop_pad
            cx1 = max(0,  int(x1 - pad_x))
            cy1 = max(0,  int(y1 - pad_y))
            cx2 = min(fw, int(x2 + pad_x))
            cy2 = min(fh, int(y2 + pad_y))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                self.logger.warning(f"Boş crop: track_id={person.get('track_id')}")
                continue

            crops.append(crop)
            offsets.append((cx1, cy1))
            valid_persons.append(person)

        if not crops:
            return [], []

        # 2. Tek GPU forward pass — tüm crop'lar batch olarak işlenir
        batch_dets = ppe_agent.detect_batch(crops)

        # 3. Sonuçları kişilerle eşleştir
        for person, (cx1, cy1), ppe_dets in zip(valid_persons, offsets, batch_dets):
            helmet_status = "unknown"
            vest_status   = "unknown"
            mask_status   = "unknown"

            for det in ppe_dets:
                cat    = det["category"]
                status = det["status"]

                if cat == "helmet":
                    if helmet_status == "unknown" or status == "present":
                        helmet_status = status
                elif cat == "vest":
                    if vest_status == "unknown" or status == "present":
                        vest_status = status
                elif cat == "mask":
                    if mask_status == "unknown" or status == "present":
                        mask_status = status

                dx1, dy1, dx2, dy2 = det["bbox"]
                all_ppe_dets.append({
                    **det,
                    "bbox": [dx1 + cx1, dy1 + cy1, dx2 + cx1, dy2 + cy1],
                })

            violations: list[str] = []
            if self.check_helmet and helmet_status == "missing":
                violations.append("no_helmet")
            if self.check_vest and vest_status == "missing":
                violations.append("no_vest")
            if self.check_mask and mask_status == "missing":
                violations.append("no_mask")

            person_results.append({
                "track_id"     : person.get("track_id"),
                "person_bbox"  : person["bbox"],
                "helmet_status": helmet_status,
                "vest_status"  : vest_status,
                "mask_status"  : mask_status,
                "violations"   : violations,
            })

        return person_results, all_ppe_dets
