from __future__ import annotations

import cv2

class CameraMonitor:

    def __init__(
        self,
        freeze_frames: int   = 60,
        dark_frames:   int   = 60,
        freeze_diff:   float = 0.002,
        dark_thresh:   float = 0.03,
    ) -> None:
        self.freeze_frames = freeze_frames
        self.dark_frames   = dark_frames
        self.freeze_diff   = freeze_diff
        self.dark_thresh   = dark_thresh

        self._prev_gray  = None
        self._freeze_cnt = 0
        self._dark_cnt   = 0
        self._status     = "online"

    @property
    def status(self) -> str:
        return self._status

    def update(self, frame) -> str | None:
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0] / 255.0

        if brightness < self.dark_thresh:
            self._dark_cnt   += 1
            self._freeze_cnt  = 0
        else:
            self._dark_cnt = 0
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                if cv2.mean(diff)[0] / 255.0 < self.freeze_diff:
                    self._freeze_cnt += 1
                else:
                    self._freeze_cnt = 0

        self._prev_gray = gray

        new_status = (
            "dark"   if self._dark_cnt   >= self.dark_frames   else
            "frozen" if self._freeze_cnt >= self.freeze_frames  else
            "online"
        )

        if new_status != self._status:
            self._status = new_status
            return new_status
        return None
