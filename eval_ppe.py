# -*- coding: utf-8 -*-
"""
eval_ppe.py  —  Scene-tabanlı PPE pipeline doğruluk testi
4 test videosu için ground truth ile karşılaştırır, TP/TN/FP/FN hesaplar.

Kullanım:
    python eval_ppe.py
"""
from __future__ import annotations

import sys
import os
import cv2
import yaml
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config'den model yolları ve cihaz
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        with open("config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_cfg     = _load_cfg()
_models  = _cfg.get("models", {})
_DEVICE  = "cpu" if not __import__("torch").cuda.is_available() else _models.get("device", "cuda")
if _DEVICE == "auto":
    _DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

PERSON_MODEL_PATH = _models.get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")
HELMET_MODEL_PATH = _models.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
VEST_MODEL_PATH   = _models.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
MASK_MODEL_PATH   = _models.get("mask_model",   "models/mask_agent_scene_200ep_best.pt")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

HELMET_CONF  = 0.15
VEST_CONF    = 0.20
MASK_CONF    = 0.20
PERSON_CONF  = 0.20
IMGSZ        = 1024
TEMPORAL_WIN = 30
INSIDE_FRAC_THR = 0.30
TRACKER      = "bytetrack.yaml"
SAMPLE_EVERY = 3

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask", "NO-Mask"]

# ---------------------------------------------------------------------------
# Ground truth — kişi bazlı
# n_main     : videodaki ana kişi sayısı
# violations : her PPE tipinde ihlal yapan kişi sayısı
# ---------------------------------------------------------------------------

VIDEOS = [
    ("nohat",  "test/nohat_test.mp4",  {
        "n_main": 5,
        "violations": {"helmet": 2, "vest": 0, "mask": 5},
    }),
    ("novest", "test/novest_test.mp4", {
        "n_main": 2,
        "violations": {"helmet": 0, "vest": 1, "mask": 2},
    }),
    ("noppe",  "test/noppe_test.mp4",  {
        "n_main": 4,
        "violations": {"helmet": 4, "vest": 4, "mask": 4},
    }),
    ("mask",   "test/mask_test.mp4",   {
        "n_main": 1,
        "violations": {"helmet": 1, "vest": 0, "mask": 0},
    }),
    # --- Yeni dış test videoları ---
    # Intel demo: sarı baret+yelek = tam PPE, siyah kıyafetli = ihlalci.
    # Maske değerlendirilmiyor (pre-COVID, kimse maske takmıyor).
    ("intel",  "test/intel_safety_full.mp4", {
        "n_main": 3,
        "violations": {"helmet": 1, "vest": 1, "mask": None},
    }),
    # KARAM tanıtım: 1 işçi başta tam PPE yok, sonra beyaz baret takıyor.
    # Maske: Hindistan fabrikası, işçi maske takmıyor → ihlal sayılır.
    ("karam",  "test/github_hardhat.mp4", {
        "n_main": 1,
        "violations": {"helmet": 1, "vest": 1, "mask": 1},
    }),
]

# ---------------------------------------------------------------------------
# Scene-based detection yardımcıları (run_live_video.py ile aynı mantık)
# ---------------------------------------------------------------------------

def class_ids(model: YOLO, names: list[str]) -> list[int]:
    n2id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in n2id]
    if missing:
        raise ValueError(f"Eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [n2id[n] for n in names]


def _inside_frac(ppe_box: list, person_box: list) -> float:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area  = (bx2 - bx1) * (by2 - by1)
    return inter / area if area > 0 else 0.0


def _scene_dets(
    model: YOLO,
    frame,
    allowed_ids: set[int],
    min_conf: float,
) -> list[tuple[str, float, list]]:
    res = model.predict(frame, imgsz=IMGSZ, conf=min_conf, device=_DEVICE, verbose=False)[0]
    if not res.boxes:
        return []
    return [
        (str(model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist())
        for b in res.boxes
        if int(b.cls[0]) in allowed_ids
    ]


def _best_scene(
    dets: list[tuple[str, float, list]],
    person_box: list,
) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for lbl, c, bbox in dets:
        if _inside_frac(bbox, person_box) >= INSIDE_FRAC_THR:
            if best is None or c > best[1]:
                best = (lbl, c)
    return best if best else ("unknown", 0.0)


def vote(q: deque, min_known: int = 3, ratio: float = 0.4) -> str:
    if not q:
        return "unknown"
    top, _ = Counter(q).most_common(1)[0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    top2, cnt = Counter(known).most_common(1)[0]
    return top2 if cnt / len(known) >= ratio else "unknown"


# ---------------------------------------------------------------------------
# Tek video çalıştır
# ---------------------------------------------------------------------------

def run_video(
    name: str,
    path: str,
    gt: dict,
    models: tuple,
    tracker: str = TRACKER,
    track_every_frame: bool = True,
) -> tuple[dict[str, tuple], float, int]:
    """Videoyu işler, per-PPE (tp,fp,fn,tn) tuple döndürür."""
    person_model, helmet_model, vest_model, mask_model = models

    n_main     = gt["n_main"]
    violations = gt["violations"]

    p_ids   = class_ids(person_model, ["Person"])
    h_ids_s = set(class_ids(helmet_model, HELMET_CLASSES))
    v_ids_s = set(class_ids(vest_model,   VEST_CLASSES))
    m_ids_s = set(class_ids(mask_model,   MASK_CLASSES))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [HATA] Açılamadı: {path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # None → o PPE tipi bu video için değerlendirilmez
    skip = {ppe for ppe, v in violations.items() if v is None}

    print(f"\n{'─'*60}")
    print(f"  {name.upper()}  —  {Path(path).name}  ({total_frames}f @ {fps:.0f}fps)")
    viol_str = "  ".join(
        f"{p}={'SKIP' if violations[p] is None else violations[p]}"
        for p in ("helmet", "vest", "mask")
    )
    print(f"  GT ({n_main} kişi):  {viol_str}")
    print(f"  tracker={tracker}  track_every_frame={track_every_frame}")
    print(f"{'─'*60}")

    states: dict[int, dict] = defaultdict(lambda: {
        "hardhat": deque(maxlen=TEMPORAL_WIN),
        "vest":    deque(maxlen=TEMPORAL_WIN),
        "mask":    deque(maxlen=TEMPORAL_WIN),
        "obs":     0,
    })

    frame_idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if track_every_frame:
            # Tracking her karede, PPE inference her SAMPLE_EVERY karede
            p_res = person_model.track(
                frame, classes=p_ids, tracker=tracker, persist=True,
                imgsz=IMGSZ, conf=PERSON_CONF, device=_DEVICE, verbose=False,
            )[0]
            boxes = p_res.boxes
            if boxes is None or boxes.id is None:
                continue
            if frame_idx % SAMPLE_EVERY != 0:
                continue
        else:
            # Tracking ve PPE inference birlikte her SAMPLE_EVERY karede
            if frame_idx % SAMPLE_EVERY != 0:
                continue
            p_res = person_model.track(
                frame, classes=p_ids, tracker=tracker, persist=True,
                imgsz=IMGSZ, conf=PERSON_CONF, device=_DEVICE, verbose=False,
            )[0]
            boxes = p_res.boxes
            if boxes is None or boxes.id is None:
                continue

        # --- PPE scene-based inference ---
        h_dets = _scene_dets(helmet_model, frame, h_ids_s, HELMET_CONF)
        v_dets = _scene_dets(vest_model,   frame, v_ids_s, VEST_CONF)
        m_dets = _scene_dets(mask_model,   frame, m_ids_s, MASK_CONF)

        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id   = int(tid)
            person_box = list(map(int, box.tolist()))

            hlabel, _ = _best_scene(h_dets, person_box)
            vlabel, _ = _best_scene(v_dets, person_box)
            mlabel, _ = _best_scene(m_dets, person_box)

            states[track_id]["hardhat"].append(hlabel)
            states[track_id]["vest"].append(vlabel)
            states[track_id]["mask"].append(mlabel)
            states[track_id]["obs"] += 1

    t_end = time.time()
    elapsed = t_end - t_start
    cap.release()

    # --- Per-person sonuçlar ---
    person_results: dict[int, tuple[str, str, str, int]] = {}
    for tid, st in sorted(states.items()):
        obs = st["obs"]
        if obs < 2:
            continue
        hv = vote(st["hardhat"], min_known=2, ratio=0.4)
        vv = vote(st["vest"],    min_known=2, ratio=0.4)
        mv = vote(st["mask"],    min_known=1, ratio=0.4)

        h_flag = "VIOLATION" if hv == "NO-Hardhat"     else ("ok" if hv == "Hardhat"      else "unknown")
        v_flag = "VIOLATION" if vv == "NO-Safety Vest" else ("ok" if vv == "Safety Vest"  else "unknown")
        m_flag = "VIOLATION" if mv == "NO-Mask"        else ("ok" if mv == "Mask"         else "unknown")
        person_results[tid] = (h_flag, v_flag, m_flag, obs)

    # --- Top-N seçimi ---
    sorted_persons = sorted(person_results.items(), key=lambda x: x[1][3], reverse=True)
    main_persons   = sorted_persons[:n_main]
    if len(main_persons) < n_main:
        print(f"  [UYARI] Yalnızca {len(main_persons)}/{n_main} ana kişi bulundu.")

    # --- Tespit sayıları ---
    d_v: dict[str, int] = {"helmet": 0, "vest": 0, "mask": 0}
    for _tid, (hf, vf, mf, _obs) in main_persons:
        if hf == "VIOLATION": d_v["helmet"] += 1
        if vf == "VIOLATION": d_v["vest"]   += 1
        if mf == "VIOLATION": d_v["mask"]   += 1

    # --- Kişi tablosu ---
    main_ids = {pid for pid, _ in main_persons}
    print(f"\n  {'ID':>4}  {'Obs':>5}  {'Helmet':>16}  {'Vest':>16}  {'Mask':>16}  Ana?")
    print(f"  {'─'*4}  {'─'*5}  {'─'*16}  {'─'*16}  {'─'*16}  {'─'*4}")
    for tid, (hf, vf, mf, obs) in sorted(person_results.items(), key=lambda x: x[1][3], reverse=True):
        tag = " <-" if tid in main_ids else ""
        h_dist = dict(Counter(states[tid]["hardhat"]))
        v_dist = dict(Counter(states[tid]["vest"]))
        m_dist = dict(Counter(states[tid]["mask"]))
        print(f"  {tid:>4}  {obs:>5}  {hf:>16}  {vf:>16}  {mf:>16}  {tag}")
        print(f"         H:{h_dist}  V:{v_dist}  M:{m_dist}")

    # --- Per-PPE TP/FP/FN/TN ---
    results: dict[str, tuple] = {}
    print(f"\n  {'PPE':<8}  {'GT_ihlal':>8}  {'Tespit':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
    print(f"  {'─'*60}")
    for ppe in ("helmet", "vest", "mask"):
        e_v  = violations[ppe]
        if e_v is None:
            print(f"  {ppe:<8}  {'SKIP':>8}  {'SKIP':>8}  {'SKIP':>4}  {'SKIP':>4}  {'SKIP':>4}  {'SKIP':>4}")
            continue
            
        e_ok = n_main - e_v
        det  = d_v[ppe]

        tp = min(e_v, det)
        fn = e_v - tp
        fp = max(0, det - e_v)
        tn = max(0, e_ok - fp)

        flags = []
        if fn: flags.append(f"FN={fn}")
        if fp: flags.append(f"FP={fp}")
        flag_str = "  <- " + ", ".join(flags) if flags else ""
        print(f"  {ppe:<8}  {e_v:>8}  {det:>8}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}{flag_str}")
        results[ppe] = (tp, fp, fn, tn)

    fps_val = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"  [Performans] {frame_idx} kare | {elapsed:.2f} saniye | {fps_val:.1f} FPS")
    print(f"  Toplam kişi (>=3 obs): {len(person_results)}  |  Ana kişi: {len(main_persons)}/{n_main}")
    return results, elapsed, frame_idx


# ---------------------------------------------------------------------------
# Ana fonksiyon + genel metrikler
# ---------------------------------------------------------------------------

def main(tracker: str = TRACKER, track_every_frame: bool = True) -> None:
    print("Modeller yükleniyor...")
    person_model = YOLO(PERSON_MODEL_PATH)
    helmet_model = YOLO(HELMET_MODEL_PATH)
    vest_model   = YOLO(VEST_MODEL_PATH)
    mask_model   = YOLO(MASK_MODEL_PATH)
    models = (person_model, helmet_model, vest_model, mask_model)
    print(f"  Hazır. Cihaz: {_DEVICE}\n")
    print(f"  Tracker: {tracker}  |  track_every_frame: {track_every_frame}")

    all_results: dict[str, dict[str, str]] = {}
    tot_elapsed = 0.0
    tot_frames = 0

    for name, path, gt in VIDEOS:
        if not Path(path).exists():
            print(f"\n[ATLA] {name}: {path} bulunamadı.")
            continue
        res, el, fr = run_video(name, path, gt, models,
                                      tracker=tracker,
                                      track_every_frame=track_every_frame)
        all_results[name] = res
        tot_elapsed += el
        tot_frames += fr

    if not all_results:
        print("\nHiç sonuç yok.")
        return

    # --- Genel metrikler (kişi-PPE bazlı) ---
    tp_tot = tn_tot = fp_tot = fn_tot = 0
    ppe_totals: dict[str, list] = {"helmet": [], "vest": [], "mask": []}
    for vid_res in all_results.values():
        for ppe, (tp, fp, fn, tn) in vid_res.items():
            tp_tot += tp; fp_tot += fp; fn_tot += fn; tn_tot += tn
            ppe_totals[ppe].append((tp, fp, fn, tn))

    total = tp_tot + tn_tot + fp_tot + fn_tot
    acc   = (tp_tot + tn_tot) / total if total else 0.0
    prec  = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec   = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n{'='*60}")
    print("GENEL OZET  (kisi-PPE bazli)")
    print(f"{'='*60}")
    print(f"TP={tp_tot}  TN={tn_tot}  FP={fp_tot}  FN={fn_tot}  (toplam kisi-PPE={total})")
    print(f"Accuracy : {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall   : {rec:.1%}")
    print(f"F1       : {f1:.1%}")

    print(f"\n  {'PPE':<8}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'Recall':>7}  {'Prec':>7}")
    print(f"  {'─'*50}")
    for ppe in ("helmet", "vest", "mask"):
        rows = ppe_totals[ppe]
        p_tp = sum(r[0] for r in rows); p_fp = sum(r[1] for r in rows)
        p_fn = sum(r[2] for r in rows); p_tn = sum(r[3] for r in rows)
        p_rec  = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0.0
        p_prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0.0
        print(f"  {ppe:<8}  {p_tp:>4}  {p_tn:>4}  {p_fp:>4}  {p_fn:>4}  {p_rec:>7.1%}  {p_prec:>7.1%}")

    if fn_tot or fp_tot:
        print(f"\nKayip/hatali tespitler:")
        for vname, vid_res in all_results.items():
            for ppe, (tp, fp, fn, tn) in vid_res.items():
                if fn: print(f"  {vname:<10}  {ppe:<8}  FN={fn}  (kacirilan ihlal)")
                if fp: print(f"  {vname:<10}  {ppe:<8}  FP={fp}  (yanlis alarm)")
    else:
        print("\nTum kisi-PPE tespitler dogru!")

    top_fps = tot_frames / tot_elapsed if tot_elapsed > 0 else 0.0
    print(f"\n[GENEL PERFORMANS] Toplam {tot_frames} kare | {tot_elapsed:.2f} saniye | Ortalama {top_fps:.1f} FPS")
    print(f"\n{'='*60}")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="PPE eval — scene-based")
    p.add_argument("--mask-model",         default=None, help="Mask model yolu")
    p.add_argument("--helmet-model",       default=None, help="Helmet model yolu")
    p.add_argument("--vest-model",         default=None, help="Vest model yolu")
    p.add_argument("--tracker",            default=TRACKER,
                   help="Tracker config (varsayılan: botsort.yaml)")
    p.add_argument("--no-track-every-frame", action="store_true",
                   help="Tracking'i de her SAMPLE_EVERY'de çalıştır (eski davranış)")
    p.add_argument("--compare", action="store_true",
                   help="Yeni vs eski ayarları karşılaştır (2x çalıştır)")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.mask_model:
        MASK_MODEL_PATH = _args.mask_model
    if _args.helmet_model:
        HELMET_MODEL_PATH = _args.helmet_model
    if _args.vest_model:
        VEST_MODEL_PATH = _args.vest_model

    if _args.compare:
        configs = [
            ("YENİ  (botsort  + track-every-frame)", "botsort.yaml",    True),
            ("ESKİ  (bytetrack + track-every-N)",    "bytetrack.yaml",  False),
        ]
        for label, trk, tef in configs:
            print(f"\n{'█'*60}")
            print(f"  KARŞILAŞTIRMA — {label}")
            print(f"{'█'*60}")
            main(tracker=trk, track_every_frame=tef)
    else:
        main(tracker=_args.tracker,
             track_every_frame=not _args.no_track_every_frame)
