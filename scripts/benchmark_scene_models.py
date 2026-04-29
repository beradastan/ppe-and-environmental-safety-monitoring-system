# -*- coding: utf-8 -*-
"""
benchmark_scene_models.py
=========================
Sahne tabanlı (tam kare) PPE tespit benchmarkı.

Karşılaştırılan modeller:
  keremberke  — YOLOv8m, 11 978 görüntü, helmet+mask+glove (vest YOK)
  voxdroid    — YOLOv8,  ~3 000 görüntü, helmet+vest+mask, 200 epoch
  hansung     — YOLOv8n, ~3 000 görüntü, helmet+vest+mask
  vinayak     — YOLOv8,  ~2 800 görüntü, helmet+vest+mask (v27)

Kullanım:
    python scripts/benchmark_scene_models.py
    python scripts/benchmark_scene_models.py --video test/nohat_test.mp4
    python scripts/benchmark_scene_models.py --conf 0.30
    python scripts/benchmark_scene_models.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Model tanımları
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id":   "yihong_yolo11m",
        "file": "models/yihong/models/yolo11/pt/yolo11m.pt",
        "arch": "YOLO11m",
        "imgs": 6632,
        "no_helmet": ["NO-Hardhat"],
        "no_vest":   ["NO-Safety Vest"],
        "no_mask":   [],           # mask ajan ayrıca eklenecek
    },
    {
        "id":   "keremberke",
        "file": "models/keremberke/best.pt",
        "arch": "YOLOv8m",
        "imgs": 11978,
        # violation sınıf isimleri (küçük harf karşılaştırma)
        "no_helmet": ["no_helmet"],
        "no_vest":   [],           # bu modelde vest sınıfı yok
        "no_mask":   ["no_mask"],
    },
    {
        "id":   "hexmon",
        "file": "models/hexmon_yolo_ppe.pt",
        "arch": "YOLOv8l",
        "imgs": 0,                # bilinmiyor
        "no_helmet": ["no-hardhat"],
        "no_vest":   ["no-safety vest"],
        "no_mask":   ["no-mask"],
    },
    {
        "id":   "voxdroid",
        "file": "models/voxdroid_200epoch_best.pt",
        "arch": "YOLOv8",
        "imgs": 3000,
        "no_helmet": ["no-hardhat"],
        "no_vest":   ["no-safety vest"],
        "no_mask":   ["no-mask"],
    },
    {
        "id":   "hansung",
        "file": "models/hansung_yolov8_ppe.pt",
        "arch": "YOLOv8n",
        "imgs": 3000,
        "no_helmet": ["no-hardhat"],
        "no_vest":   ["no-safety vest"],
        "no_mask":   ["no-mask"],
    },
    {
        "id":   "vinayak",
        "file": "models/pretrained/vinayakmane/ppe.pt",
        "arch": "YOLOv8",
        "imgs": 2801,
        "no_helmet": ["no-hardhat"],
        "no_vest":   ["no-safety vest"],
        "no_mask":   ["no-mask"],
    },
]

DEFAULT_VIDEOS = [
    "test/nohat_test.mp4",
    "test/novest_test.mp4",
    "test/noppe_test.mp4",
    "test/mask_test.mp4",
    "test/general_test.mp4",
]


# ---------------------------------------------------------------------------
# Sonuç yapısı
# ---------------------------------------------------------------------------

@dataclass
class VideoResult:
    model_id:    str
    video:       str
    total_frames: int = 0
    # her violation tipi için "en az 1 tespit olan frame sayısı"
    frames_no_helmet: int = 0
    frames_no_vest:   int = 0
    frames_no_mask:   int = 0
    # toplam tespit sayıları
    dets_no_helmet: int = 0
    dets_no_vest:   int = 0
    dets_no_mask:   int = 0
    ms_list: list[float] = field(default_factory=list)
    error:   str | None  = None

    @property
    def avg_ms(self) -> float:
        return sum(self.ms_list) / len(self.ms_list) if self.ms_list else 0.0

    @property
    def fps(self) -> float:
        return 1000 / self.avg_ms if self.avg_ms > 0 else 0.0

    @property
    def violation_frame_pct(self) -> float:
        if not self.total_frames:
            return 0.0
        any_viol = self.frames_no_helmet + self.frames_no_vest + self.frames_no_mask
        # Yaklaşık: en az bir ihlal olan frame oranı
        return min(100.0, any_viol / self.total_frames * 100)


# ---------------------------------------------------------------------------
# Tek model + video çalıştırıcı
# ---------------------------------------------------------------------------

def run_model_on_video(meta: dict, video_path: str, conf: float, device: str) -> VideoResult:
    from ultralytics import YOLO
    import torch

    res = VideoResult(model_id=meta["id"], video=video_path)

    model_file = ROOT / meta["file"]
    if not model_file.exists():
        res.error = f"Model dosyası bulunamadı: {model_file}"
        return res

    video_full = ROOT / video_path
    if not video_full.exists():
        res.error = f"Video bulunamadı: {video_full}"
        return res

    try:
        yolo = YOLO(str(model_file))
        use_half = device == "cuda" and torch.cuda.is_available()
    except Exception as e:
        res.error = str(e)
        return res

    # Sınıf adı → id eşlemesi (küçük harf)
    name_to_id: dict[str, int] = {v.lower(): k for k, v in yolo.names.items()}

    def ids_for(labels: list[str]) -> list[int]:
        return [name_to_id[l.lower()] for l in labels if l.lower() in name_to_id]

    h_ids = ids_for(meta["no_helmet"])
    v_ids = ids_for(meta["no_vest"])
    m_ids = ids_for(meta["no_mask"])
    all_viol_ids = h_ids + v_ids + m_ids
    if not all_viol_ids:
        # hiç violation sınıfı eşleşmedi — tüm sınıfları çalıştır
        all_viol_ids = None  # YOLO varsayılan: tüm sınıflar

    cap = cv2.VideoCapture(str(video_full))
    if not cap.isOpened():
        res.error = "Video açılamadı"
        return res

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res.total_frames += 1

            t0 = time.perf_counter()
            preds = yolo.predict(
                frame,
                classes=all_viol_ids,
                conf=conf,
                device=device,
                half=use_half,
                verbose=False,
            )
            ms = (time.perf_counter() - t0) * 1000
            res.ms_list.append(ms)

            if not preds:
                continue
            pred = preds[0]
            if pred.boxes is None or len(pred.boxes) == 0:
                continue

            detected_ids: set[int] = set()
            for box in pred.boxes:
                cid  = int(box.cls[0])
                conf_det = float(box.conf[0])
                if conf_det >= conf:
                    detected_ids.add(cid)

            h_hit = bool(detected_ids & set(h_ids))
            v_hit = bool(detected_ids & set(v_ids))
            m_hit = bool(detected_ids & set(m_ids))

            if h_hit:
                res.frames_no_helmet += 1
                res.dets_no_helmet   += sum(1 for b in pred.boxes if int(b.cls[0]) in h_ids)
            if v_hit:
                res.frames_no_vest   += 1
                res.dets_no_vest     += sum(1 for b in pred.boxes if int(b.cls[0]) in v_ids)
            if m_hit:
                res.frames_no_mask   += 1
                res.dets_no_mask     += sum(1 for b in pred.boxes if int(b.cls[0]) in m_ids)

            frame_idx += 1
    finally:
        cap.release()

    return res


# ---------------------------------------------------------------------------
# Çıktı
# ---------------------------------------------------------------------------

def print_results(results: list[VideoResult], models: list[dict]) -> None:
    # model meta erişimi için
    meta_by_id = {m["id"]: m for m in models}

    # Video başlıkları topla
    videos = sorted({r.video for r in results})

    for video in videos:
        vname = Path(video).name
        print(f"\n{'='*72}")
        print(f"  {vname}")
        print(f"{'='*72}")
        print(f"  {'Model':<14} {'Arch':<10} {'Imgs':>6}  "
              f"{'FPS':>5}  {'Frames':>6}  "
              f"{'H↓':>6}  {'V↓':>6}  {'M↓':>6}  {'Dets(H/V/M)':>14}")
        print(f"  {'-'*68}")

        for r in [r for r in results if r.video == video]:
            if r.error:
                print(f"  {r.model_id:<14}  HATA: {r.error}")
                continue
            m = meta_by_id[r.model_id]
            has_vest = bool(m["no_vest"])
            vest_str = f"{r.frames_no_vest:>6}" if has_vest else "   N/A"
            dets_str = f"{r.dets_no_helmet}/{r.dets_no_vest}/{r.dets_no_mask}"
            print(f"  {r.model_id:<14} {m['arch']:<10} {m['imgs']:>6}  "
                  f"{r.fps:>5.1f}  {r.total_frames:>6}  "
                  f"{r.frames_no_helmet:>6}  {vest_str}  {r.frames_no_mask:>6}  "
                  f"{dets_str:>14}")

    print(f"\n{'='*72}")
    print("  Sütun açıklamaları:")
    print("    H↓ = no_helmet tespit edilen frame sayısı")
    print("    V↓ = no_vest   tespit edilen frame sayısı  (N/A = sınıf yok)")
    print("    M↓ = no_mask   tespit edilen frame sayısı")
    print("    Dets = toplam tespit kutusu sayısı (H/V/M)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  help="Tek video testi")
    ap.add_argument("--model",  help="Sadece bu modeli çalıştır (id)")
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("  [!] CUDA yok, CPU kullanılıyor.")

    videos = [args.video] if args.video else DEFAULT_VIDEOS
    videos = [v for v in videos if (ROOT / v).exists()]
    if not videos:
        print("Hiç video bulunamadı!")
        return

    models = [m for m in MODELS if not args.model or m["id"] == args.model]
    if not models:
        print(f"Model bulunamadı: {args.model}")
        return

    results: list[VideoResult] = []

    for meta in models:
        model_file = ROOT / meta["file"]
        if not model_file.exists():
            print(f"  [{meta['id']}] model dosyası yok, atlanıyor: {meta['file']}")
            continue
        for video in videos:
            print(f"  [{meta['id']}] {Path(video).name} ...", end=" ", flush=True)
            r = run_model_on_video(meta, video, args.conf, args.device)
            results.append(r)
            if r.error:
                print(f"HATA: {r.error}")
            else:
                print(f"{r.total_frames} frame, {r.fps:.1f} FPS "
                      f"(H:{r.frames_no_helmet} V:{r.frames_no_vest} M:{r.frames_no_mask})")

    print_results(results, models)


if __name__ == "__main__":
    main()
