"""
Vest Agent Karsilastirma Benchmarki

Deney 1 - Egitim stratejisi:
  normal_vest_30  vs  cropvest_30   (padding=0.30)

Deney 2 - Final model karsilastirmasi:
  normal_vest_30  vs  cropvest_30  vs  vinayak   (padding=0.30)

Pipeline: person tracking -> crop -> vest predict -> metric toplama
Cikti: runs/benchmarks/vest/<exp>/<video>/<timestamp>/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

import cv2
import torch
import yaml
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root() / p).resolve()


def choose_video() -> Path:
    desktop = Path.home() / "Desktop"
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Karsilastirilacak videoyu sec",
        initialdir=str(desktop if desktop.exists() else Path.home()),
        filetypes=[
            ("Videolar", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Tum dosyalar", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit("Video secilmedi.")
    return Path(selected)


def select_device(requested: str) -> str:
    if requested.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    raise SystemExit("GPU secildi ama CUDA aktif degil. --device cpu kullan.")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class: {missing}  |  Mevcut: {list(model.names.values())}")
    return [name_to_id[n] for n in names]


def crop_with_padding(frame, box, pad_ratio: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pw = int(max(1, x2 - x1) * pad_ratio)
    ph = int(max(1, y2 - y1) * pad_ratio)
    return frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]


def best_detection(model: YOLO, result, allowed_ids: list[int], min_conf: float) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for box in result.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        label = str(model.names[cid])
        if best is None or conf > best[1]:
            best = (label, conf)
    return best if best else ("unknown", 0.0)


def vote(values: deque[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Model ve deney tanimlari
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, dict[str, Any]] = {
    "normal_vest_30": {
        "path": "models/vest_agent_30_best.pt",
        "vest_classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "Normal Vest (tam sahne, 30 epoch)",
    },
    "cropvest_30": {
        "path": "models/cropvest_agent_30_best.pt",
        "vest_classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "CropVest (krop tabanli, 30 epoch)",
    },
    "vinayak": {
        "path": "models/pretrained/vinayakmane/ppe.pt",
        "vest_classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "Vinayak PPE (vest filtreli)",
    },
    "bera_vest": {
        "path": "models/vest_agent_final_best.pt",
        "vest_classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "Bera Vest (full epoch)",
    },
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "exp1": {
        "label": "Deney 1: Egitim stratejisi (normal_vest_30 vs cropvest_30)",
        "models": ["normal_vest_30", "cropvest_30"],
        "paddings": [0.30],
        "output_subdir": "exp1_training_strategy",
    },
    "exp2": {
        "label": "Deney 2: Final model karsilastirmasi (normal_vest_30 vs cropvest_30 vs vinayak)",
        "models": ["normal_vest_30", "cropvest_30", "vinayak"],
        "paddings": [0.30],
        "output_subdir": "exp2_final_comparison",
    },
    "exp3": {
        "label": "Deney 3: Bera Vest (full epoch) vs Vinayak",
        "models": ["bera_vest", "vinayak"],
        "paddings": [0.30, 0.45, 0.60],
        "output_subdir": "exp3_bera_vs_vinayak",
    },
}

PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"
PERSON_CONF = 0.25
VEST_CONF = 0.30
TRACKER = "bytetrack.yaml"
CROP_PADDING = 0.30
IMGSZ = 640
TEMPORAL_WINDOW = 10


# ---------------------------------------------------------------------------
# Tek model benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    source: Path,
    model_key: str,
    person_model: YOLO,
    vest_model: YOLO,
    vest_ids: list[int],
    device: str,
    max_frames: int,
    crop_padding: float = CROP_PADDING,
) -> dict[str, Any]:
    states: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {source}")

    person_ids = class_ids(person_model, ["person"])

    frames = 0
    vest_conf_vals: list[float] = []
    vest_counts: Counter = Counter()
    label_changes = 0
    label_observations = 0
    last_labels: dict[int, str] = {}
    track_ids: set[int] = set()
    active_counts: list[int] = []

    started = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        person_result = person_model.track(
            frame,
            classes=person_ids,
            tracker=TRACKER,
            persist=True,
            imgsz=IMGSZ,
            conf=PERSON_CONF,
            device=device,
            verbose=False,
        )[0]

        boxes = person_result.boxes
        if boxes is None or boxes.id is None:
            active_counts.append(0)
        else:
            active_counts.append(len(boxes.id))
            for box, tid in zip(boxes.xyxy, boxes.id):
                track_id = int(tid)
                track_ids.add(track_id)

                crop = crop_with_padding(frame, box.tolist(), crop_padding)
                result = vest_model.predict(
                    crop,
                    classes=vest_ids,
                    imgsz=IMGSZ,
                    conf=VEST_CONF,
                    device=device,
                    verbose=False,
                )[0]
                label, conf = best_detection(vest_model, result, vest_ids, VEST_CONF)

                states[track_id].append(label)
                vest_counts[label] += 1
                if conf > 0:
                    vest_conf_vals.append(conf)

                if track_id in last_labels:
                    label_changes += int(last_labels[track_id] != label)
                last_labels[track_id] = label
                label_observations += 1

        if frames >= max_frames:
            break

    elapsed = time.perf_counter() - started
    cap.release()

    unknown_rate = vest_counts["unknown"] / label_observations if label_observations else 1.0
    known_rate = 1.0 - unknown_rate
    stability = 1.0 - (label_changes / max(1, label_observations - 1))
    avg_active = mean([float(v) for v in active_counts])
    continuity = min(1.0, avg_active / max(1, len(track_ids)))

    return {
        "model": model_key,
        "label": ALL_MODELS[model_key]["label"],
        "crop_padding": crop_padding,
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(frames / elapsed, 3) if elapsed else 0.0,
        "person_observations": label_observations,
        "unique_track_ids": len(track_ids),
        "avg_active_tracks": round(avg_active, 3),
        "track_continuity": round(continuity, 3),
        "avg_vest_conf": round(mean(vest_conf_vals), 3),
        "vest_unknown_rate": round(unknown_rate, 3),
        "vest_known_rate": round(known_rate, 3),
        "label_stability": round(max(0.0, stability), 3),
        "vest_counts": dict(vest_counts),
    }


# ---------------------------------------------------------------------------
# Skorlama
# ---------------------------------------------------------------------------

def add_scores(results: list[dict[str, Any]]) -> None:
    max_fps = max((r["fps"] for r in results), default=1.0) or 1.0
    for r in results:
        speed = r["fps"] / max_fps
        r["score"] = round(
            0.40 * r["vest_known_rate"]
            + 0.30 * r["label_stability"]
            + 0.20 * r["avg_vest_conf"]
            + 0.10 * speed,
            4,
        )


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------

def write_reports(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "score", "model", "label", "crop_padding", "fps", "frames", "person_observations",
        "unique_track_ids", "avg_active_tracks", "track_continuity",
        "avg_vest_conf", "vest_known_rate", "vest_unknown_rate",
        "label_stability", "elapsed_sec",
    ]

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        best = results[0]
        best_path = output_dir / "best_model.yaml"
        with best_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"best_model": best["model"], "label": best["label"], "score": best["score"]},
                f,
                sort_keys=False,
                allow_unicode=True,
            )
        print(f"  Best model: {best_path}")

    print(f"  CSV   : {csv_path}")
    print(f"  JSON  : {json_path}")


def print_comparison(results: list[dict[str, Any]]) -> None:
    row_fmt = "{:<22} {:>8} {:>7} {:>8} {:>10} {:>12} {:>9} {:>8}"
    sep = "-" * 90
    print("\n" + sep)
    print(row_fmt.format("Model", "Padding", "Score", "FPS", "Known%", "Avg_conf", "Stability", "Unknown%"))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r["model"][:22],
            f"{r['crop_padding']:.2f}",
            f"{r['score']:.4f}",
            f"{r['fps']:.1f}",
            f"{r['vest_known_rate']*100:.1f}%",
            f"{r['avg_vest_conf']:.3f}",
            f"{r['label_stability']:.3f}",
            f"{r['vest_unknown_rate']*100:.1f}%",
        ))
    print(sep)
    print("\nYelek dagilimi:")
    for r in results:
        print(f"  [{r['model']} pad={r['crop_padding']}]  {r['vest_counts']}")
    print()


# ---------------------------------------------------------------------------
# Deney runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_cfg: dict[str, Any],
    source: Path,
    person_model: YOLO,
    device: str,
    max_frames: int,
    output_base: Path,
) -> None:
    print(f"\n{'='*70}")
    print(f"  {exp_cfg['label']}")
    print(f"{'='*70}")

    loaded: dict[str, tuple[YOLO, list[int]]] = {}
    for model_key in exp_cfg["models"]:
        model_info = ALL_MODELS[model_key]
        model_path = resolve_path(model_info["path"])
        if not model_path.exists():
            print(f"[SKIP] Model bulunamadi: {model_path}")
            continue
        vest_model = YOLO(str(model_path))
        v_ids = class_ids(vest_model, model_info["vest_classes"])
        loaded[model_key] = (vest_model, v_ids)
        print(f"  Yuklendi: [{model_key}] {model_path.name}  classes={vest_model.names}")

    results: list[dict[str, Any]] = []
    paddings = exp_cfg["paddings"]
    total = len(loaded) * len(paddings)
    idx = 0

    for padding in paddings:
        for model_key, (vest_model, v_ids) in loaded.items():
            idx += 1
            print(f"\n  [{idx}/{total}] {model_key}  padding={padding}")
            result = run_benchmark(
                source=source,
                model_key=model_key,
                person_model=person_model,
                vest_model=vest_model,
                vest_ids=v_ids,
                device=device,
                max_frames=max_frames,
                crop_padding=padding,
            )
            results.append(result)
            print(f"    fps={result['fps']}  known={result['vest_known_rate']:.3f}  "
                  f"conf={result['avg_vest_conf']:.3f}  stability={result['label_stability']:.3f}")

    add_scores(results)
    results.sort(key=lambda r: r["score"], reverse=True)
    print_comparison(results)

    output_dir = output_base / exp_cfg["output_subdir"] / source.stem / time.strftime("%Y%m%d_%H%M%S")
    write_reports(output_dir, results)
    print(f"\n  Raporlar: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vest agent karsilastirma benchmarki.")
    parser.add_argument("--source", help="Video yolu. Belirtilmezse dosya secici acilir.")
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all",
        help="Hangi deneyi calistir: exp1, exp2, all (varsayilan: all)",
    )
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/benchmarks/vest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source:
        source = resolve_path(args.source)
    else:
        source = choose_video()

    if not source.exists():
        raise SystemExit(f"Video bulunamadi: {source}")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit("Sadece video dosyasi desteklenir.")

    device = select_device(args.device)

    person_path = resolve_path(PERSON_MODEL_PATH)
    if not person_path.exists():
        raise SystemExit(f"Person model bulunamadi: {person_path}")
    person_model = YOLO(str(person_path))
    print(f"Person model: {person_path.name}\n")

    output_base = resolve_path(args.output_dir)
    experiments = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}

    for _, exp_cfg in experiments.items():
        run_experiment(exp_cfg, source, person_model, device, args.max_frames, output_base)


if __name__ == "__main__":
    main()
