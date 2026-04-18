# -*- coding: utf-8 -*-
"""
benchmark_models.py
===================
models/registry.yaml'daki PPE modellerini crop mimarisinde karşılaştırır.

Kullanım:
    python benchmark_models.py                         # tüm available modeller, tüm videolar
    python benchmark_models.py --video test/nohat_test.mp4
    python benchmark_models.py --model voxdroid_200ep
    python benchmark_models.py --all                   # download_required dahil (dosya varsa)
    python benchmark_models.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_id:          str
    video:             str
    total_frames:      int  = 0
    events_created:    int  = 0
    first_event_frame: int | None = None
    violation_frames:  int  = 0
    person_violations: dict[int, set[str]] = field(default_factory=dict)
    inference_ms:      list[float] = field(default_factory=list)
    error:             str | None = None

    @property
    def avg_ms(self) -> float:
        return sum(self.inference_ms) / len(self.inference_ms) if self.inference_ms else 0.0

    @property
    def p95_ms(self) -> float:
        s = sorted(self.inference_ms)
        return s[int(len(s) * 0.95)] if s else 0.0

    @property
    def fps(self) -> float:
        return 1000 / self.avg_ms if self.avg_ms > 0 else 0.0

    @property
    def first_event_sec(self) -> float | None:
        return self.first_event_frame / 30.0 if self.first_event_frame is not None else None


# ---------------------------------------------------------------------------
# Registry okuyucu
# ---------------------------------------------------------------------------

def load_registry(path: str = "models/registry.yaml") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def _label_map(model_meta: dict) -> dict[str, tuple[str, str]]:
    """registry'deki ppe_classes tanımından label→(category,status) haritası üret."""
    mapping: dict[str, tuple[str, str]] = {}
    pc = model_meta.get("ppe_classes", {})
    for key, labels in pc.items():
        if key.endswith("_present"):
            cat = key[: -len("_present")]
            status = "present"
        elif key.endswith("_missing"):
            cat = key[: -len("_missing")]
            status = "missing"
        else:
            continue
        for lbl in labels:
            mapping[lbl.strip().lower()] = (cat, status)
    return mapping


# ---------------------------------------------------------------------------
# Tek model + video çalıştırıcı
# ---------------------------------------------------------------------------

def run_model_on_video(
    model_meta:   dict,
    video_path:   str,
    person_agent,
    device:       str,
    conf_ppe:     float,
    crop_pad:     float,
    check_helmet: bool,
    check_vest:   bool,
    check_mask:   bool,
) -> ModelResult:
    from ultralytics import YOLO
    from matching.crop_ppe_matcher import CropPPEMatcher
    from event_manager import PersonEventManager

    result = ModelResult(model_id=model_meta["id"], video=video_path)
    model_file = model_meta["file"]

    if not Path(model_file).exists():
        result.error = f"Dosya bulunamadı: {model_file}"
        return result

    # PPE modelini yükle — registry'deki label map ile PPEAgent yerine doğrudan YOLO
    try:
        yolo = YOLO(model_file)
        import torch
        if device == "cuda" and torch.cuda.is_available():
            yolo.to("cuda")
    except Exception as e:
        result.error = str(e)
        return result

    label_map = _label_map(model_meta)
    ppe_class_ids = [
        cid for cid, name in yolo.names.items()
        if name.strip().lower() in label_map
    ]

    # Hafif PPEAgent wrapper — registry label map'ini kullanır
    class _RegistryPPEAgent:
        def __init__(self):
            self.model      = yolo
            self.confidence = conf_ppe
            self.device     = device
            self._ppe_class_ids = ppe_class_ids
            self._label_map = label_map

        def detect(self, frame):
            return self._infer([frame])[0]

        def detect_batch(self, crops):
            if not crops: return []
            return self._infer(crops)

        def _infer(self, imgs):
            try:
                results = self.model(
                    imgs,
                    conf=self.confidence,
                    classes=self._ppe_class_ids if self._ppe_class_ids else None,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                return [[] for _ in imgs]

            out = []
            for r in results:
                dets = []
                if r.boxes is not None:
                    for box in r.boxes:
                        name = self.model.names.get(int(box.cls[0]), "").strip().lower()
                        mapping = self._label_map.get(name)
                        if mapping is None:
                            continue
                        cat, status = mapping
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        dets.append({
                            "label"     : name,
                            "bbox"      : [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "category"  : cat,
                            "status"    : status,
                        })
                out.append(dets)
            return out

    ppe_agent    = _RegistryPPEAgent()
    crop_matcher = CropPPEMatcher(
        crop_pad=crop_pad,
        check_helmet=check_helmet,
        check_vest=check_vest,
        check_mask=check_mask,
    )
    em = PersonEventManager(
        check_helmet=check_helmet,
        check_vest=check_vest,
        check_mask=check_mask,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result.error = "Video açılamadı"
        return result

    result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warmup = 15

    for fi in range(result.total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        persons   = person_agent.detect(frame)
        confirmed = [p for p in persons if p.get("is_confirmed", True)]
        pr, _     = (
            crop_matcher.match(frame, confirmed, ppe_agent)
            if confirmed else ([], [])
        )
        er = em.process_frame(pr)

        dt = (time.perf_counter() - t0) * 1000
        if fi >= warmup:
            result.inference_ms.append(dt)

        sig = er["signature"]
        if sig["helmet_violation"] or sig["vest_violation"] or sig["mask_violation"]:
            result.violation_frames += 1

        if er["event_status"] == "new":
            result.events_created += 1
            if result.first_event_frame is None:
                result.first_event_frame = fi

        for pv in er.get("person_violations", []):
            tid = pv["track_id"]
            if tid is not None:
                result.person_violations.setdefault(tid, set()).update(pv["violations"])

    cap.release()
    return result


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------

COLS = ["model_id", "avg_ms", "fps", "p95_ms", "events", "first_event_sec", "viol_frames"]

def _score(r: ModelResult) -> float:
    """Daha erken ve daha kararlı tespit = daha yüksek puan."""
    if r.error or r.events_created == 0:
        return 0.0
    speed_score    = max(0, 100 - r.avg_ms)           # düşük ms → yüksek puan
    early_score    = max(0, 60 - (r.first_event_sec or 60)) * 2
    coverage_score = r.violation_frames / max(r.total_frames, 1) * 100
    return speed_score + early_score + coverage_score


def print_report(results_by_video: dict[str, list[ModelResult]]) -> None:
    SEP  = "=" * 90
    SEP2 = "-" * 90

    print(f"\n{SEP}")
    print("  MODEL BENCHMARK RAPORU — Crop Mimarisi (GPU)")
    print(SEP)

    for video, results in results_by_video.items():
        vname = Path(video).name
        ok    = [r for r in results if not r.error]
        err   = [r for r in results if r.error]

        print(f"\n📹  {vname}")
        print(SEP2)

        if not ok:
            print("  Tüm modeller hata verdi.")
            for r in err:
                print(f"  [{r.model_id}] {r.error}")
            continue

        # Tablo başlığı
        print(f"  {'Model':<22} {'Ort ms':>7} {'FPS':>6} {'P95 ms':>7} "
              f"{'Event':>6} {'1.Event(s)':>11} {'İhlal Fr':>9} {'Puan':>7}")
        print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*11} {'-'*9} {'-'*7}")

        scored = sorted(ok, key=_score, reverse=True)
        for i, r in enumerate(scored):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            fes   = f"{r.first_event_sec:.1f}s" if r.first_event_sec is not None else "YOK"
            score = _score(r)
            print(f"  {medal} {r.model_id:<20} {r.avg_ms:>7.1f} {r.fps:>6.1f} {r.p95_ms:>7.1f} "
                  f"{r.events_created:>6} {fes:>11} {r.violation_frames:>9} {score:>7.1f}")

        if err:
            print()
            for r in err:
                print(f"  ⚠  [{r.model_id}] HATA: {r.error}")

        # En iyi model detayı
        best = scored[0]
        print(f"\n  → En iyi: [{best.model_id}]  İhlal kişiler: ", end="")
        for tid, viols in sorted(best.person_violations.items()):
            print(f"#{tid}({','.join(sorted(viols))})", end=" ")
        print()

    # Genel sıralama
    print(f"\n{SEP}")
    print("  GENEL SIRALAMA (tüm videolar toplamı)")
    print(SEP2)

    total_scores: dict[str, float] = {}
    for results in results_by_video.values():
        for r in results:
            total_scores[r.model_id] = total_scores.get(r.model_id, 0) + _score(r)

    for rank, (mid, score) in enumerate(
        sorted(total_scores.items(), key=lambda x: x[1], reverse=True), 1
    ):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"
        print(f"  {medal}  {mid:<25}  toplam puan: {score:.1f}")

    print(SEP)
    print("\nAçıklama — Puan formülü:")
    print("  Puan = (100 - avg_ms) + (60 - first_event_sec)*2 + (ihlal_frame/total_frame)*100")
    print("  Daha erken ve daha kararlı tespit → daha yüksek puan.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, ".")

    p = argparse.ArgumentParser(
        description="PPE model benchmarkı — crop mimarisi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",    default=None,
                   help="Tek video yolu. Belirtilmezse test/ altındaki tüm mp4'ler.")
    p.add_argument("--model",    default=None,
                   help="Tek model ID (registry'deki 'id' alanı). Belirtilmezse tümü.")
    p.add_argument("--all",      action="store_true",
                   help="download_required modelleri de dahil et (dosya varsa).")
    p.add_argument("--device",   default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--conf-ppe", type=float, default=0.25, dest="conf_ppe")
    p.add_argument("--crop-pad", type=float, default=0.40, dest="crop_pad")
    p.add_argument("--check-mask", action="store_true", dest="check_mask")
    args = p.parse_args()

    registry = load_registry("models/registry.yaml")

    # Sadece PPE modellerini al
    models = [m for m in registry if m.get("role") == "ppe"]

    if not args.all:
        models = [m for m in models if m.get("status") == "available"]

    if args.model:
        models = [m for m in models if m["id"] == args.model]

    if not models:
        print("Benchmark edilecek model bulunamadı.")
        sys.exit(1)

    if args.video:
        videos = [args.video]
    else:
        videos = sorted(Path("test").glob("*.mp4"))

    # PersonTrackingAgent bir kez yükle, her model çalıştırmasında sıfırla
    from agents.person_tracking_agent import PersonTrackingAgent

    print(f"\nBenchmark başlıyor: {len(models)} model × {len(videos)} video")
    print(f"Device: {args.device}  crop_pad: {args.crop_pad}  conf_ppe: {args.conf_ppe}\n")

    results_by_video: dict[str, list[ModelResult]] = {str(v): [] for v in videos}

    for model_meta in models:
        mid = model_meta["id"]
        for video in videos:
            # Her model+video kombinasyonu için tracker sıfırla
            person_agent = PersonTrackingAgent(
                model_path="models/yakhyo_yolov8n_best.pt",
                confidence=0.40,
                device=args.device,
            )
            print(f"  [{mid}] {Path(str(video)).name} …", end=" ", flush=True)
            r = run_model_on_video(
                model_meta   = model_meta,
                video_path   = str(video),
                person_agent = person_agent,
                device       = args.device,
                conf_ppe     = args.conf_ppe,
                crop_pad     = args.crop_pad,
                check_helmet = True,
                check_vest   = True,
                check_mask   = args.check_mask,
            )
            status = f"HATA: {r.error}" if r.error else f"{r.avg_ms:.1f}ms  {r.events_created} event"
            print(status)
            results_by_video[str(video)].append(r)

    print_report(results_by_video)


if __name__ == "__main__":
    main()
