"""
Helmet Agent Karsilastirma Benchmarki

Karsilastirilan modeller:
  - crophelmet : models/crophelmet_agent_final_best.pt  (kendi egitilen)
  - vinayak    : models/pretrained/vinayakmane/ppe.pt   (helmet classlari filtreli)

Her iki model de ayni pipeline'da calistirilir:
  person tracking -> crop -> helmet predict -> metric toplama

Cikti: runs/benchmarks/helmet/<video_adi>/<timestamp>/
  summary.json, summary.csv, best_config.yaml
"""
from __future__ import annotations

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
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Model tanimlari
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, dict[str, Any]] = {
    "normal_helmet_30": {
        "path": "models/helmet_agent_30_best.pt",
        "helmet_classes": ["Hardhat", "NO-Hardhat"],
        "label": "Normal Helmet (tam sahne, 30 epoch)",
    },
    "crophelmet_30": {
        "path": "models/crophelmet_agent_30_best.pt",
        "helmet_classes": ["Hardhat", "NO-Hardhat"],
        "label": "CropHelmet (krop tabanli, 30 epoch)",
    },
    "crophelmet_final": {
        "path": "models/crophelmet_agent_final_best.pt",
        "helmet_classes": ["Hardhat", "NO-Hardhat"],
        "label": "CropHelmet Final (krop tabanli, final)",
    },
    "helmet_agent_final": {
        "path": "models/helmet_agent_final_best.pt",
        "helmet_classes": ["Hardhat", "NO-Hardhat"],
        "label": "Helmet Agent Final (tam sahne, final)",
    },
    "vinayak": {
        "path": "models/pretrained/vinayakmane/ppe.pt",
        "helmet_classes": ["Hardhat", "NO-Hardhat"],
        "label": "Vinayak PPE (helmet filtreli)",
    },
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "exp1": {
        "label": "Deney 1: Egitim stratejisi (normal_helmet_30 vs crophelmet_30)",
        "models": ["normal_helmet_30", "crophelmet_30"],
        "paddings": [0.30],
        "output_subdir": "exp1_training_strategy",
    },
    "exp2": {
        "label": "Deney 2: Final model karsilastirmasi (normal_helmet_30 vs crophelmet_final vs vinayak)",
        "models": ["normal_helmet_30", "crophelmet_final", "vinayak"],
        "paddings": [0.30],
        "output_subdir": "exp2_final_comparison",
    },
    "exp3": {
        "label": "Deney 3: helmet_agent_final vs crophelmet_final vs vinayak (padding karsilastirmasi)",
        "model_paddings": {
            "helmet_agent_final": [0.30, 0.45, 0.60, 0.80],
            "crophelmet_final":   [0.60, 0.80],
            "vinayak":            [0.30],
        },
        "output_subdir": "exp3_final_padding_comparison",
    },
    "exp4": {
        "label": "Deney 4: crophelmet_final pad=0.80 conf karsilastirmasi (0.30 vs 0.20 vs 0.15)",
        "model_paddings": {
            "crophelmet_final": [0.80],
        },
        "confs": [0.30, 0.20, 0.15],
        "output_subdir": "exp4_conf_tuning",
    },
}

PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"
PERSON_CONF = 0.25
HELMET_CONF = 0.30
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
    helmet_model: YOLO,
    helmet_ids: list[int],
    device: str,
    max_frames: int,
    crop_padding: float = CROP_PADDING,
    helmet_conf: float = HELMET_CONF,
) -> dict[str, Any]:
    states: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {source}")

    person_ids = class_ids(person_model, ["person"])

    frames = 0
    hardhat_conf_vals: list[float] = []
    person_conf_vals: list[float] = []
    hardhat_counts: Counter = Counter()
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
                person_conf_vals.append(float(box[4]) if box.shape[0] > 4 else float(person_result.boxes.conf[list(boxes.id).index(tid)]))

                crop = crop_with_padding(frame, box.tolist(), crop_padding)
                result = helmet_model.predict(
                    crop,
                    classes=helmet_ids,
                    imgsz=IMGSZ,
                    conf=helmet_conf,
                    device=device,
                    verbose=False,
                )[0]
                label, conf = best_detection(helmet_model, result, helmet_ids, helmet_conf)

                states[track_id].append(label)
                hardhat_counts[label] += 1
                if conf > 0:
                    hardhat_conf_vals.append(conf)

                if track_id in last_labels:
                    label_changes += int(last_labels[track_id] != label)
                last_labels[track_id] = label
                label_observations += 1

        if frames >= max_frames:
            break

    elapsed = time.perf_counter() - started
    cap.release()

    person_observations = label_observations
    unknown_rate = hardhat_counts["unknown"] / person_observations if person_observations else 1.0
    known_rate = 1.0 - unknown_rate
    stability = 1.0 - (label_changes / max(1, label_observations - 1))
    avg_active = mean([float(v) for v in active_counts])
    continuity = min(1.0, avg_active / max(1, len(track_ids)))

    return {
        "model": model_key,
        "label": ALL_MODELS[model_key]["label"],
        "crop_padding": crop_padding,
        "helmet_conf": helmet_conf,
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(frames / elapsed, 3) if elapsed else 0.0,
        "person_observations": person_observations,
        "unique_track_ids": len(track_ids),
        "avg_active_tracks": round(avg_active, 3),
        "track_continuity": round(continuity, 3),
        "avg_hardhat_conf": round(mean(hardhat_conf_vals), 3),
        "hardhat_unknown_rate": round(unknown_rate, 3),
        "hardhat_known_rate": round(known_rate, 3),
        "label_stability": round(max(0.0, stability), 3),
        "hardhat_counts": dict(hardhat_counts),
    }


# ---------------------------------------------------------------------------
# Skorlama
# ---------------------------------------------------------------------------

def add_scores(results: list[dict[str, Any]]) -> None:
    max_fps = max((r["fps"] for r in results), default=1.0) or 1.0
    for r in results:
        speed = r["fps"] / max_fps
        r["score"] = round(
            0.40 * r["hardhat_known_rate"]
            + 0.30 * r["label_stability"]
            + 0.20 * r["avg_hardhat_conf"]
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
        "avg_hardhat_conf", "hardhat_known_rate", "hardhat_unknown_rate",
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
    row_fmt = "{:<22} {:>8} {:>6} {:>7} {:>8} {:>10} {:>12} {:>9} {:>8}"
    sep = "-" * 97
    print("\n" + sep)
    print(row_fmt.format("Model", "Padding", "Conf", "Score", "FPS", "Known%", "Avg_conf", "Stability", "Unknown%"))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r["model"][:22],
            f"{r['crop_padding']:.2f}",
            f"{r.get('helmet_conf', HELMET_CONF):.2f}",
            f"{r['score']:.4f}",
            f"{r['fps']:.1f}",
            f"{r['hardhat_known_rate']*100:.1f}%",
            f"{r['avg_hardhat_conf']:.3f}",
            f"{r['label_stability']:.3f}",
            f"{r['hardhat_unknown_rate']*100:.1f}%",
        ))
    print(sep)
    print(f"\nKasik dagilimi:")
    for r in results:
        print(f"  [{r['model']} pad={r['crop_padding']}]  {r['hardhat_counts']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helmet agent karsilastirma benchmarki.")
    parser.add_argument("--source", help="Video yolu. Belirtilmezse dosya secici acilir.")
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all",
        help="Hangi deneyi calistir: exp1, exp2, exp3, all (varsayilan: all)",
    )
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/benchmarks/helmet")
    return parser.parse_args()


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

    # per-model padding desteği: model_paddings varsa onu, yoksa models+paddings kombinasyonunu kullan
    if "model_paddings" in exp_cfg:
        model_padding_pairs = [
            (model_key, pad)
            for model_key, pads in exp_cfg["model_paddings"].items()
            for pad in pads
        ]
    else:
        model_padding_pairs = [
            (model_key, pad)
            for pad in exp_cfg["paddings"]
            for model_key in exp_cfg["models"]
        ]

    confs = exp_cfg.get("confs", [HELMET_CONF])
    combinations = [(mk, pad, conf) for mk, pad in model_padding_pairs for conf in confs]

    all_model_keys = list(dict.fromkeys(mk for mk, _, _ in combinations))
    loaded: dict[str, tuple[YOLO, list[int]]] = {}
    for model_key in all_model_keys:
        model_info = ALL_MODELS[model_key]
        model_path = resolve_path(model_info["path"])
        if not model_path.exists():
            print(f"[SKIP] Model bulunamadi: {model_path}")
            continue
        helmet_model = YOLO(str(model_path))
        h_ids = class_ids(helmet_model, model_info["helmet_classes"])
        loaded[model_key] = (helmet_model, h_ids)
        print(f"  Yuklendi: [{model_key}] {model_path.name}")

    results: list[dict[str, Any]] = []
    total = sum(1 for mk, _, _ in combinations if mk in loaded)
    idx = 0

    for model_key, padding, conf in combinations:
        if model_key not in loaded:
            continue
        helmet_model, h_ids = loaded[model_key]
        idx += 1
        print(f"\n  [{idx}/{total}] {model_key}  padding={padding}  conf={conf}")
        result = run_benchmark(
            source=source,
            model_key=model_key,
            person_model=person_model,
            helmet_model=helmet_model,
            helmet_ids=h_ids,
            device=device,
            max_frames=max_frames,
            crop_padding=padding,
            helmet_conf=conf,
        )
        results.append(result)
        print(f"    fps={result['fps']}  known={result['hardhat_known_rate']:.3f}  "
              f"conf={result['avg_hardhat_conf']:.3f}  stability={result['label_stability']:.3f}")

    add_scores(results)
    results.sort(key=lambda r: r["score"], reverse=True)
    print_comparison(results)

    output_dir = output_base / exp_cfg["output_subdir"] / source.stem / time.strftime("%Y%m%d_%H%M%S")
    write_reports(output_dir, results)
    print(f"\n  Raporlar: {output_dir}")


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

    for exp_key, exp_cfg in experiments.items():
        run_experiment(exp_cfg, source, person_model, device, args.max_frames, output_base)


if __name__ == "__main__":
    main()
