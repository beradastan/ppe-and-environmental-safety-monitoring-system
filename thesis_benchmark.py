# -*- coding: utf-8 -*-
"""
thesis_benchmark.py
===================
Tez için crop-based ve scene-based detection modlarını karşılaştırır.
Backend, veritabanı veya kayıt işlemi gerektirmez; sadece inference yapar.

Kullanim:
    python thesis_benchmark.py

Çıktı:
    benchmark_results.json  — tezde kullanılacak sayısal değerler
    benchmark_results.txt   — okunabilir özet rapor
"""
from __future__ import annotations

import json
import time
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import torch
import yaml

os.chdir(Path(__file__).parent)
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────

TEST_VIDEOS = [
    {"name": "nohat_test",  "path": "test/nohat_test.mp4",  "ground_truth": {"helmet": "violation", "vest": "ok",        "mask": "violation"}},
    {"name": "novest_test", "path": "test/novest_test.mp4", "ground_truth": {"helmet": "ok",        "vest": "violation", "mask": "violation"}},
    {"name": "noppe_test",  "path": "test/noppe_test.mp4",  "ground_truth": {"helmet": "violation", "vest": "violation", "mask": "violation"}},
    {"name": "mask_test",   "path": "test/mask_test.mp4",   "ground_truth": {"helmet": "violation", "vest": "ok",        "mask": "ok"}},
]

TEMPORAL_WIN   = 20
INSIDE_FRAC    = 0.40
PPE_INFER_EVERY = 4
FIRE_INFER_EVERY = 5
MIN_CROP_PX    = 40
IMGSZ          = 640

# Confidence eşikleri — crop
CROP_HELMET_CONF = 0.15
CROP_VEST_CONF   = 0.35
CROP_MASK_CONF   = 0.10

# Confidence eşikleri — scene
SCENE_HELMET_CONF = 0.20
SCENE_VEST_CONF   = 0.30
SCENE_MASK_CONF   = 0.25

FIRE_CONF   = 0.75
PERSON_CONF = 0.25

# Model yolları (config.yaml'dan okunur)
def _load_cfg():
    try:
        with open("config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_CFG = _load_cfg()
_MC  = _CFG.get("models", {})
_CC  = _MC.get("crop",  {})
_SC  = _MC.get("scene", {})

PERSON_MODEL_PATH = _MC.get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")
FIRE_MODEL_PATH   = _MC.get("fire_model",   "models/bera/fire_smoke_other_agent_final_best.pt")

CROP_HELMET_PATH = _CC.get("helmet_model", "models/bera/crophelmet_agent_final_best.pt")
CROP_VEST_PATH   = _CC.get("vest_model",   "models/bera/cropvest_agent_final_best.pt")
CROP_MASK_PATH   = _CC.get("mask_model",   "models/bera/cropmask_agent_final_best.pt")

SCENE_HELMET_PATH = _SC.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
SCENE_VEST_PATH   = _SC.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
SCENE_MASK_PATH   = _SC.get("mask_model",   "models/mask_agent_scene_200ep_yolov8m_best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

def vote(q: deque) -> str:
    if not q:
        return "unknown"
    counts = Counter(q)
    top, _ = counts.most_common(1)[0]
    return top


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


def crop_ppe(frame, x1, y1, x2, y2, ppe: str):
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if ppe == "helmet":
        cx1 = max(0, x1 - int(pw * 0.10)); cy1 = max(0, y1 - int(ph * 0.15))
        cx2 = min(fw, x2 + int(pw * 0.10)); cy2 = min(fh, y1 + int(ph * 0.40))
    elif ppe == "vest":
        cx1 = max(0, x1 - int(pw * 0.15)); cy1 = max(0, y1 + int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15)); cy2 = min(fh, y1 + int(ph * 0.90))
    else:  # mask
        cx1 = max(0, x1 - int(pw * 0.15)); cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15)); cy2 = min(fh, y1 + int(ph * 0.45))
    crop = frame[cy1:cy2, cx1:cx2]
    return crop


def predict_crop_ppe(model: YOLO, crop, conf: float, pos_cls: str, neg_cls: str):
    if crop is None or crop.size == 0:
        return "unknown"
    h, w = crop.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return "unknown"
    res = model.predict(crop, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
    if not res.boxes:
        return neg_cls  # crop içinde PPE göremedi → eksik
    names = model.names
    best_label, best_conf = "unknown", 0.0
    for b in res.boxes:
        lbl = str(names[int(b.cls[0])])
        c   = float(b.conf[0])
        if c > best_conf:
            best_conf = c
            best_label = lbl
    return best_label


def predict_scene_ppe(model: YOLO, frame, person_box: list, conf: float, pos_cls: str, neg_cls: str):
    res = model.predict(frame, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
    if not res.boxes:
        return "unknown"
    names = model.names
    best_label, best_conf = "unknown", 0.0
    for b in res.boxes:
        lbl  = str(names[int(b.cls[0])])
        c    = float(b.conf[0])
        bbox = b.xyxy[0].tolist()
        if _inside_frac(bbox, person_box) >= INSIDE_FRAC and c > best_conf:
            best_conf  = c
            best_label = lbl
    return best_label if best_label != "unknown" else "unknown"


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def gpu_max_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Modelleri yükle
# ──────────────────────────────────────────────────────────────────────────────

def load_models(mode: str):
    print(f"\n[MODEL] {mode.upper()} modeli yükleniyor...")
    torch.cuda.reset_peak_memory_stats()

    person_model = YOLO(PERSON_MODEL_PATH)
    fire_model   = YOLO(FIRE_MODEL_PATH)

    if mode == "crop":
        helmet_model = YOLO(CROP_HELMET_PATH)
        vest_model   = YOLO(CROP_VEST_PATH)
        mask_model   = YOLO(CROP_MASK_PATH)
    else:
        helmet_model = YOLO(SCENE_HELMET_PATH)
        vest_model   = YOLO(SCENE_VEST_PATH)
        mask_model   = YOLO(SCENE_MASK_PATH)

    # Warm-up (GPU başlatma gecikmesini hesaptan çıkar)
    dummy = torch.zeros(1, 3, 640, 640).to(DEVICE)
    for m in [person_model, fire_model, helmet_model, vest_model, mask_model]:
        m.predict(source=dummy.cpu().numpy().transpose(0, 2, 3, 1)[0],
                  imgsz=IMGSZ, device=DEVICE, verbose=False)
    torch.cuda.reset_peak_memory_stats()

    mem_after_load = gpu_mem_mb()
    print(f"  GPU bellek (yükleme sonrası): {mem_after_load:.0f} MB")

    return person_model, fire_model, helmet_model, vest_model, mask_model


# ──────────────────────────────────────────────────────────────────────────────
# Tek video benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_video(video_path: str, mode: str,
              person_model, fire_model,
              helmet_model, vest_model, mask_model) -> dict:

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Temporal state
    helmet_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    vest_hist:   dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    mask_hist:   dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))

    # Violation sayaçları
    violation_counts = {"helmet": 0, "vest": 0, "mask": 0, "fire": 0}
    # Kişi bazlı ihlal takibi (toplam unique kişi sayısını bulmak için)
    person_violations: dict[int, set] = defaultdict(set)

    frame_times: list[float] = []
    peak_gpu_mb_list: list[float] = []

    frame_idx = 0
    fire_cache: list = []  # son fire sonucu (her 5 frame'de bir güncellenir)

    torch.cuda.reset_peak_memory_stats()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Kişi tespiti + ByteTrack (her frame)
        person_res = person_model.track(
            frame, imgsz=IMGSZ, conf=PERSON_CONF,
            device=DEVICE, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )[0]

        persons = []
        if person_res.boxes is not None and person_res.boxes.id is not None:
            for box, tid in zip(person_res.boxes.xyxy, person_res.boxes.id):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                persons.append({"tid": int(tid), "box": [x1, y1, x2, y2]})

        # ── Yangın tespiti (her 5 frame)
        if frame_idx % FIRE_INFER_EVERY == 0:
            fire_res = fire_model.predict(
                frame, imgsz=IMGSZ, conf=FIRE_CONF, device=DEVICE, verbose=False
            )[0]
            fire_cache = []
            if fire_res.boxes:
                for b in fire_res.boxes:
                    lbl = str(fire_model.names[int(b.cls[0])])
                    if lbl == "fire":
                        fire_cache.append(float(b.conf[0]))
            if fire_cache:
                violation_counts["fire"] += 1

        # ── PPE tespiti (her 4 frame)
        if frame_idx % PPE_INFER_EVERY == 0 and persons:
            if mode == "crop":
                for p in persons:
                    x1, y1, x2, y2 = p["box"]
                    tid = p["tid"]

                    h_crop = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                    v_crop = crop_ppe(frame, x1, y1, x2, y2, "vest")
                    m_crop = crop_ppe(frame, x1, y1, x2, y2, "mask")

                    h_lbl = predict_crop_ppe(helmet_model, h_crop, CROP_HELMET_CONF, "Hardhat", "NO-Hardhat")
                    v_lbl = predict_crop_ppe(vest_model,   v_crop, CROP_VEST_CONF,   "Safety Vest", "NO-Safety Vest")
                    m_lbl = predict_crop_ppe(mask_model,   m_crop, CROP_MASK_CONF,   "Mask", "NO-Mask")

                    if h_lbl != "unknown": helmet_hist[tid].append(h_lbl)
                    if v_lbl != "unknown": vest_hist[tid].append(v_lbl)
                    if m_lbl != "unknown": mask_hist[tid].append(m_lbl)

            else:  # scene
                # Tüm PPE modellerini tam kare üzerinde bir kez çalıştır
                helmet_dets = []
                vest_dets   = []
                mask_dets   = []

                hres = helmet_model.predict(frame, imgsz=IMGSZ, conf=SCENE_HELMET_CONF, device=DEVICE, verbose=False)[0]
                vres = vest_model.predict(  frame, imgsz=IMGSZ, conf=SCENE_VEST_CONF,   device=DEVICE, verbose=False)[0]
                mres = mask_model.predict(  frame, imgsz=IMGSZ, conf=SCENE_MASK_CONF,   device=DEVICE, verbose=False)[0]

                if hres.boxes:
                    helmet_dets = [(str(helmet_model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist()) for b in hres.boxes]
                if vres.boxes:
                    vest_dets   = [(str(vest_model.names[int(b.cls[0])]),   float(b.conf[0]), b.xyxy[0].tolist()) for b in vres.boxes]
                if mres.boxes:
                    mask_dets   = [(str(mask_model.names[int(b.cls[0])]),   float(b.conf[0]), b.xyxy[0].tolist()) for b in mres.boxes]

                for p in persons:
                    pbox = p["box"]
                    tid  = p["tid"]

                    # helmet
                    best_h = "unknown"
                    for lbl, c, bbox in helmet_dets:
                        if _inside_frac(bbox, pbox) >= INSIDE_FRAC:
                            best_h = lbl; break
                    # vest
                    best_v = "unknown"
                    for lbl, c, bbox in vest_dets:
                        if _inside_frac(bbox, pbox) >= INSIDE_FRAC:
                            best_v = lbl; break
                    # mask
                    best_m = "unknown"
                    for lbl, c, bbox in mask_dets:
                        if _inside_frac(bbox, pbox) >= INSIDE_FRAC:
                            best_m = lbl; break

                    if best_h != "unknown": helmet_hist[tid].append(best_h)
                    if best_v != "unknown": vest_hist[tid].append(best_v)
                    if best_m != "unknown": mask_hist[tid].append(best_m)

        # ── Temporal voting ile ihlal sayımı (her frame)
        for p in persons:
            tid = p["tid"]
            h_vote = vote(helmet_hist[tid])
            v_vote = vote(vest_hist[tid])
            m_vote = vote(mask_hist[tid])
            if h_vote == "NO-Hardhat":
                violation_counts["helmet"] += 1
                person_violations[tid].add("helmet")
            if v_vote == "NO-Safety Vest":
                violation_counts["vest"] += 1
                person_violations[tid].add("vest")
            if m_vote == "NO-Mask":
                violation_counts["mask"] += 1
                person_violations[tid].add("mask")

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        peak_gpu_mb_list.append(gpu_max_mem_mb())
        frame_idx += 1

    cap.release()

    # İstatistikler
    avg_fps  = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    peak_gpu = max(peak_gpu_mb_list) if peak_gpu_mb_list else 0
    total_persons = len(helmet_hist)  # benzersiz track ID sayısı

    # İhlali olan kişi sayıları
    persons_with_helmet_viol = sum(1 for vs in person_violations.values() if "helmet" in vs)
    persons_with_vest_viol   = sum(1 for vs in person_violations.values() if "vest"   in vs)
    persons_with_mask_viol   = sum(1 for vs in person_violations.values() if "mask"   in vs)

    return {
        "total_frames":    frame_idx,
        "avg_fps":         round(avg_fps, 2),
        "min_fps":         round(1.0 / max(frame_times), 2) if frame_times else 0,
        "max_fps":         round(1.0 / min(frame_times), 2) if frame_times else 0,
        "peak_gpu_mb":     round(peak_gpu, 0),
        "total_persons_tracked": total_persons,
        "persons_with_helmet_violation": persons_with_helmet_viol,
        "persons_with_vest_violation":   persons_with_vest_viol,
        "persons_with_mask_violation":   persons_with_mask_viol,
        "fire_frames_detected": violation_counts["fire"],
        "violation_frame_counts": violation_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Ana akış
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SafetyMonitor — Tez Benchmark")
    print(f"  GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  CUDA:   {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"  Torch:  {torch.__version__}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 70)

    results = {}

    for mode in ["crop", "scene"]:
        print(f"\n{'='*70}")
        print(f"MOD: {mode.upper()}")
        print("=" * 70)

        person_model, fire_model, helmet_model, vest_model, mask_model = load_models(mode)
        results[mode] = {}

        for video_info in TEST_VIDEOS:
            vname = video_info["name"]
            vpath = video_info["path"]
            gt    = video_info["ground_truth"]

            if not Path(vpath).exists():
                print(f"  [UYARI] Video bulunamadı: {vpath}")
                continue

            print(f"\n  Video: {vname}  ({mode.upper()})")

            # GPU belleği sıfırla
            torch.cuda.reset_peak_memory_stats()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_start = time.perf_counter()
            stats = run_video(vpath, mode, person_model, fire_model,
                              helmet_model, vest_model, mask_model)
            t_total = time.perf_counter() - t_start

            # Doğruluk değerlendirmesi (ground truth ile karşılaştırma)
            detected = {}
            detected["helmet"] = "violation" if stats["persons_with_helmet_violation"] > 0 else "ok"
            detected["vest"]   = "violation" if stats["persons_with_vest_violation"]   > 0 else "ok"
            detected["mask"]   = "violation" if stats["persons_with_mask_violation"]   > 0 else "ok"

            correct = sum(detected[k] == gt[k] for k in ["helmet", "vest", "mask"])

            print(f"    FPS (ort): {stats['avg_fps']:6.1f}  |  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
            print(f"    Kişi takibi: {stats['total_persons_tracked']}  |  Toplam süre: {t_total:.1f}s")
            print(f"    Ground truth  — helmet:{gt['helmet']:9s}  vest:{gt['vest']:9s}  mask:{gt['mask']}")
            print(f"    Tespit sonucu — helmet:{detected['helmet']:9s}  vest:{detected['vest']:9s}  mask:{detected['mask']}")
            print(f"    Doğruluk: {correct}/3  {'✓' * correct}{'✗' * (3 - correct)}")

            results[mode][vname] = {
                **stats,
                "total_time_sec": round(t_total, 2),
                "ground_truth":   gt,
                "detected":       detected,
                "correct_3":      correct,
            }

        # Modelleri bellekten at
        del person_model, fire_model, helmet_model, vest_model, mask_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── JSON çıktısı
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Metin özet raporu
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("SafetyMonitor — Tez Benchmark Özet Raporu\n")
        f.write("=" * 70 + "\n\n")

        for mode in ["crop", "scene"]:
            f.write(f"MOD: {mode.upper()}\n")
            f.write("-" * 50 + "\n")
            mode_res = results.get(mode, {})

            fps_vals = [mode_res[v]["avg_fps"] for v in mode_res]
            gpu_vals = [mode_res[v]["peak_gpu_mb"] for v in mode_res]
            acc_vals = [mode_res[v]["correct_3"]   for v in mode_res]

            f.write(f"Ortalama FPS (tüm videolar): {sum(fps_vals)/len(fps_vals):.1f}\n")
            f.write(f"Peak GPU bellek (max):        {max(gpu_vals):.0f} MB\n")
            f.write(f"Ground truth doğruluğu:      {sum(acc_vals)}/{len(acc_vals)*3} PPE sınıfı doğru\n\n")

            for vname, s in mode_res.items():
                f.write(f"  {vname}:\n")
                f.write(f"    FPS: {s['avg_fps']:.1f}  |  GPU: {s['peak_gpu_mb']:.0f} MB  |  Kişi: {s['total_persons_tracked']}\n")
                for ppe in ["helmet", "vest", "mask"]:
                    gt_val = s["ground_truth"][ppe]
                    det_val = s["detected"][ppe]
                    ok = "✓" if det_val == gt_val else "✗"
                    f.write(f"    {ppe:6s}: GT={gt_val:9s}  tespit={det_val:9s}  {ok}\n")
                f.write("\n")
            f.write("\n")

        # Karşılaştırma özeti
        f.write("KARŞILAŞTIRMA (crop vs scene)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Video':<16} {'Crop FPS':>10} {'Scene FPS':>10} {'Crop GPU(MB)':>13} {'Scene GPU(MB)':>14}\n")
        for vname in [v["name"] for v in TEST_VIDEOS]:
            cr = results.get("crop",  {}).get(vname, {})
            sc = results.get("scene", {}).get(vname, {})
            if cr and sc:
                f.write(f"{vname:<16} {cr['avg_fps']:>10.1f} {sc['avg_fps']:>10.1f} "
                        f"{cr['peak_gpu_mb']:>13.0f} {sc['peak_gpu_mb']:>14.0f}\n")

    print("\n" + "=" * 70)
    print("Benchmark tamamlandı!")
    print("  → benchmark_results.json")
    print("  → benchmark_results.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
