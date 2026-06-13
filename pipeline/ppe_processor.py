from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from pipeline.config import (
    CROP_MASK_IMGSZ, IMGSZ, MIN_TRACK_FRAMES,
    PPE_INFER_EVERY, SCENE_PPE_INFER_EVERY, STATES_CLEANUP_EVERY,
    COLOR_UNKNOWN,
)
from pipeline.ppe_detector import (
    best_scene, collect_dets, compliance_color,
    crop_ok, crop_ppe, crop_to_frame,
    global_assign_ppe, is_region_too_small,
    scene_dets, validate_ppe_scored, vote,
)
from pipeline.visualizer import draw_box, draw_ppe_annotations

@dataclass
class PPEModels:
    helmet: object
    vest:   object
    mask:   object
    h_ids:  list[int]
    v_ids:  list[int]
    m_ids:  list[int]

    @property
    def h_ids_set(self) -> set[int]: return set(self.h_ids)
    @property
    def v_ids_set(self) -> set[int]: return set(self.v_ids)
    @property
    def m_ids_set(self) -> set[int]: return set(self.m_ids)

@dataclass
class PPEThresholds:
    crop_helmet:   float
    crop_vest:     float
    crop_mask:     float
    scene_helmet:  float
    scene_vest:    float
    scene_mask:    float
    device:        str
    half:          bool

class PPEProcessor:

    def __init__(
        self,
        mode:            str,
        models:          PPEModels,
        thresholds:      PPEThresholds,
        temporal_window: int,
    ) -> None:
        self.mode       = mode
        self.models     = models
        self.thresholds = thresholds

        self.states = defaultdict(lambda: {
            "hardhat":     deque(maxlen=temporal_window),
            "vest":        deque(maxlen=temporal_window),
            "mask":        deque(maxlen=temporal_window),
            "frame_count": 0,
        })
        self._seen_stable_pids: set[int] = set()

    def process_frame(
        self,
        frame,
        draw_frame,
        boxes,
        stable_map:        dict[int, int],
        all_persons_frame: list[dict],
        frame_idx:         int,
        display:           bool,
        fh:                int,
        fw:                int,
    ) -> list[dict]:
        self._cleanup_stale_states(frame_idx)

        if self.mode == "crop":
            return self._crop_mode(
                frame, draw_frame, boxes, stable_map, all_persons_frame,
                frame_idx, display, fh, fw,
            )
        return self._scene_mode(
            frame, draw_frame, boxes, stable_map, frame_idx, display,
        )

    def _crop_mode(
        self,
        frame, draw_frame, boxes, stable_map, all_persons_frame,
        frame_idx, display, fh, fw,
    ) -> list[dict]:
        t = self.thresholds
        m = self.models
        _do_ppe = (frame_idx % PPE_INFER_EVERY == 0)

        h_cands: list[dict]       = []
        v_cands: list[dict]       = []
        m_cands: list[dict]       = []
        h_batch: list[tuple]      = []
        v_batch: list[tuple]      = []
        m_batch: list[tuple]      = []
        person_coords: dict[int, tuple] = {}

        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id   = int(tid)
            stable_pid = stable_map.get(track_id, track_id)
            self._seen_stable_pids.add(stable_pid)
            self.states[stable_pid]["frame_count"] += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            person_coords[stable_pid] = (x1, y1, x2, y2)

            if _do_ppe:
                hcrop, hox, hoy = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                if crop_ok(hcrop) and not is_region_too_small(x1, y1, x2, y2, "helmet"):
                    h_batch.append((stable_pid, track_id, hcrop, hox, hoy))

                vcrop, vox, voy = crop_ppe(frame, x1, y1, x2, y2, "vest")
                if crop_ok(vcrop):
                    v_batch.append((stable_pid, track_id, vcrop, vox, voy))

                mcrop, mox, moy = crop_ppe(frame, x1, y1, x2, y2, "mask")
                if crop_ok(mcrop) and not is_region_too_small(x1, y1, x2, y2, "mask"):
                    m_batch.append((stable_pid, track_id, mcrop, mox, moy))

        if h_batch:
            h_results = m.helmet.predict(
                [b[2] for b in h_batch], classes=m.h_ids, imgsz=IMGSZ,
                conf=t.crop_helmet, device=t.device, half=t.half, verbose=False,
            )
            for (stable_pid, track_id, _, hox, hoy), hres in zip(h_batch, h_results):
                hdets = collect_dets(m.helmet, hres, m.h_ids, t.crop_helmet)
                if hdets:
                    best = max(hdets, key=lambda d: d["conf"])
                    hbbox_f = list(crop_to_frame(best["bbox"], hox, hoy, fh, fw))
                    vlbl, h_own, h_nb, h_reason = validate_ppe_scored(
                        hbbox_f, best["label"], track_id, all_persons_frame, "helmet")
                    h_cands.append({"tid": stable_pid, "bbox_f": hbbox_f,
                                    "label": vlbl, "raw_label": best["label"],
                                    "conf": best["conf"], "own_score": h_own,
                                    "neighbor_pen": h_nb, "reason": h_reason})

        if v_batch:
            v_results = m.vest.predict(
                [b[2] for b in v_batch], classes=m.v_ids, imgsz=IMGSZ,
                conf=t.crop_vest, device=t.device, half=t.half, verbose=False,
            )
            for (stable_pid, track_id, _, vox, voy), vres in zip(v_batch, v_results):
                vdets = collect_dets(m.vest, vres, m.v_ids, t.crop_vest)
                if vdets:
                    best = max(vdets, key=lambda d: d["conf"])
                    vbbox_f = list(crop_to_frame(best["bbox"], vox, voy, fh, fw))
                    vlbl, v_own, v_nb, v_reason = validate_ppe_scored(
                        vbbox_f, best["label"], track_id, all_persons_frame, "vest")
                    v_cands.append({"tid": stable_pid, "bbox_f": vbbox_f,
                                    "label": vlbl, "raw_label": best["label"],
                                    "conf": best["conf"], "own_score": v_own,
                                    "neighbor_pen": v_nb, "reason": v_reason})

        if m_batch:
            m_results = m.mask.predict(
                [b[2] for b in m_batch], classes=m.m_ids, imgsz=CROP_MASK_IMGSZ,
                conf=t.crop_mask, device=t.device, half=t.half, verbose=False,
            )
            for (stable_pid, track_id, _, mox, moy), mres in zip(m_batch, m_results):
                mdets = collect_dets(m.mask, mres, m.m_ids, t.crop_mask)
                if mdets:
                    best = max(mdets, key=lambda d: d["conf"])
                    mbbox_f = list(crop_to_frame(best["bbox"], mox, moy, fh, fw))
                    vlbl, m_own, m_nb, m_reason = validate_ppe_scored(
                        mbbox_f, best["label"], track_id, all_persons_frame, "mask")
                    m_cands.append({"tid": stable_pid, "bbox_f": mbbox_f,
                                    "label": vlbl, "raw_label": best["label"],
                                    "conf": best["conf"], "own_score": m_own,
                                    "neighbor_pen": m_nb, "reason": m_reason})

        h_assigned = global_assign_ppe(h_cands)
        v_assigned = global_assign_ppe(v_cands)
        m_assigned = global_assign_ppe(m_cands)

        persons_with_ppe: list[dict] = []
        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id   = int(tid)
            stable_pid = stable_map.get(track_id, track_id)
            x1, y1, x2, y2 = person_coords[stable_pid]

            hcand = h_assigned.get(stable_pid)
            hconf = hcand["conf"]   if hcand else 0.0
            hbbox = hcand["bbox_f"] if hcand else None
            if _do_ppe:
                self.states[stable_pid]["hardhat"].append(hcand["label"] if hcand else "unknown")
            hvote = vote(self.states[stable_pid]["hardhat"], min_known=3)

            vcand = v_assigned.get(stable_pid)
            vconf = vcand["conf"]   if vcand else 0.0
            vbbox = vcand["bbox_f"] if vcand else None
            if _do_ppe:
                self.states[stable_pid]["vest"].append(vcand["label"] if vcand else "unknown")
            vvote = vote(self.states[stable_pid]["vest"], min_known=2)

            mcand = m_assigned.get(stable_pid)
            mconf = mcand["conf"]   if mcand else 0.0
            mbbox = mcand["bbox_f"] if mcand else None
            if _do_ppe:
                self.states[stable_pid]["mask"].append(mcand["label"] if mcand else "unknown")
            mvote = vote(self.states[stable_pid]["mask"], min_known=1)

            if self.states[stable_pid]["frame_count"] < MIN_TRACK_FRAMES:
                draw_box(draw_frame, x1, y1, x2, y2, "...", COLOR_UNKNOWN)
                continue

            color, viols = compliance_color(hvote, vvote, mvote)
            persons_with_ppe.append(
                self._build_person_record(stable_pid, viols, hvote, hconf, vvote, vconf, mvote, mconf)
            )
            draw_box(draw_frame, x1, y1, x2, y2, f"ID{stable_pid}", color)
            if display:
                draw_ppe_annotations(draw_frame, hbbox, hvote, hconf, vbbox, vvote, vconf, mbbox, mvote, mconf)

        return persons_with_ppe

    def _scene_mode(
        self,
        frame, draw_frame, boxes, stable_map, frame_idx, display,
    ) -> list[dict]:
        t = self.thresholds
        m = self.models
        _do_scene_ppe = (frame_idx % SCENE_PPE_INFER_EVERY == 0)

        if _do_scene_ppe:
            scene_h = scene_dets(m.helmet, frame, m.h_ids_set, t.scene_helmet, t.device, t.half)
            scene_v = scene_dets(m.vest,   frame, m.v_ids_set, t.scene_vest,   t.device, t.half)
            scene_m = scene_dets(m.mask,   frame, m.m_ids_set, t.scene_mask,   t.device, t.half)
        else:
            scene_h = scene_v = scene_m = []

        persons_with_ppe: list[dict] = []
        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id   = int(tid)
            stable_pid = stable_map.get(track_id, track_id)
            self._seen_stable_pids.add(stable_pid)
            self.states[stable_pid]["frame_count"] += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            person_box = [x1, y1, x2, y2]

            if _do_scene_ppe:
                hlabel, hconf, hbbox = best_scene(scene_h, person_box)
                vlabel, vconf, vbbox = best_scene(scene_v, person_box)
                mlabel, mconf, mbbox = best_scene(scene_m, person_box)
                self.states[stable_pid]["hardhat"].append(hlabel)
                self.states[stable_pid]["vest"].append(vlabel)
                self.states[stable_pid]["mask"].append(mlabel)
            else:
                hconf = vconf = mconf = 0.0
                hbbox = vbbox = mbbox = None

            hvote = vote(self.states[stable_pid]["hardhat"], min_known=2)
            vvote = vote(self.states[stable_pid]["vest"],    min_known=2)
            mvote = vote(self.states[stable_pid]["mask"],    min_known=1)

            color, viols = compliance_color(hvote, vvote, mvote)
            persons_with_ppe.append(
                self._build_person_record(stable_pid, viols, hvote, hconf, vvote, vconf, mvote, mconf)
            )
            draw_box(draw_frame, x1, y1, x2, y2, f"ID{stable_pid}", color)
            if display:
                draw_ppe_annotations(draw_frame, hbbox, hvote, hconf, vbbox, vvote, vconf, mbbox, mvote, mconf)

        return persons_with_ppe

    @staticmethod
    def _build_person_record(
        stable_pid: int, viols: list[str],
        hvote: str, hconf: float,
        vvote: str, vconf: float,
        mvote: str, mconf: float,
    ) -> dict:
        return {
            "track_id":      stable_pid,
            "violations":    viols,
            "helmet_status": "ok" if hvote == "Hardhat"     else "violation" if hvote == "NO-Hardhat"     else "unknown",
            "vest_status":   "ok" if vvote == "Safety Vest" else "violation" if vvote == "NO-Safety Vest" else "unknown",
            "mask_status":   "ok" if mvote == "Mask"        else "violation" if mvote == "NO-Mask"        else "unknown",
            "helmet_conf":   round(hconf, 2),
            "vest_conf":     round(vconf, 2),
            "mask_conf":     round(mconf, 2),
        }

    def _cleanup_stale_states(self, frame_idx: int) -> None:
        if frame_idx % STATES_CLEANUP_EVERY != 0 or not self._seen_stable_pids:
            return
        stale = set(self.states.keys()) - self._seen_stable_pids
        for pid in stale:
            del self.states[pid]
        if stale:
            print(f"  [CLEANUP] {len(stale)} stale stable_pid entries removed. Remaining: {len(self.states)}")
        self._seen_stable_pids.clear()
