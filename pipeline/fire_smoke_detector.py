from __future__ import annotations

from collections import deque

class FireSmokeDetector:

    def __init__(
        self,
        fire_min_area:     float = 0.01,
        fire_growth_window: int  = 10,
        fire_growth_factor: float = 1.5,
    ) -> None:
        self.fire_min_area      = fire_min_area
        self.fire_growth_window = fire_growth_window
        self.fire_growth_factor = fire_growth_factor

        self._fire_area_history:  deque[float] = deque(maxlen=fire_growth_window * 2)
        self._smoke_area_history: deque[float] = deque(maxlen=fire_growth_window * 2)

    def update(
        self,
        fire_res,
        frame_area: int,
    ) -> tuple[bool, float, bool, float]:
        fire_conf_max  = 0.0
        smoke_conf_max = 0.0
        max_fire_area  = 0.0
        max_smoke_area = 0.0

        if fire_res.boxes:
            for box in fire_res.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                name = fire_res.names[cid]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                if name == "fire":
                    max_fire_area  = max(max_fire_area, area_ratio)
                    fire_conf_max  = max(fire_conf_max, conf)
                elif name == "smoke":
                    max_smoke_area = max(max_smoke_area, area_ratio)
                    smoke_conf_max = max(smoke_conf_max, conf)

        self._fire_area_history.append(max_fire_area)
        self._smoke_area_history.append(max_smoke_area)

        fire_raw  = fire_conf_max  > 0 and (max_fire_area  >= self.fire_min_area or self._is_growing(self._fire_area_history))
        smoke_raw = smoke_conf_max > 0 and (max_smoke_area >= self.fire_min_area or self._is_growing(self._smoke_area_history))

        return fire_raw, fire_conf_max, smoke_raw, smoke_conf_max

    def _is_growing(self, history: deque[float]) -> bool:
        if len(history) < self.fire_growth_window:
            return False
        half      = self.fire_growth_window // 2
        hist      = list(history)
        older_avg = sum(hist[-self.fire_growth_window:-half]) / half
        newer_avg = sum(hist[-half:]) / half
        return older_avg > 0 and (newer_avg / older_avg) >= self.fire_growth_factor
