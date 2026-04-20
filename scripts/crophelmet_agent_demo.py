from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tkinter import Tk, filedialog

import cv2
import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class AgentConfig:
    class_names: list[str]
    conf: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def choose_source_file() -> Path:
    desktop = Path.home() / "Desktop"
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Test kaynagini sec (video veya gorsel)",
        initialdir=str(desktop if desktop.exists() else Path.home()),
        filetypes=[
            ("Video ve gorseller", "*.mp4 *.avi *.mov *.mkv *.webm *.jpg *.jpeg *.png *.bmp *.webp"),
            ("Videolar", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Gorseller", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Tum dosyalar", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit("Dosya secilmedi.")
    return Path(selected)


def resolve_source(value: str | None) -> int | Path:
    if value is None:
        return choose_source_file()
    if value.isdigit():
        return int(value)
    return resolve_path(value)


def select_device(requested: str) -> str:
    if requested.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    raise SystemExit(
        "GPU secildi ama CUDA aktif degil. --device cpu kullan."
    )


def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {name: cid for cid, name in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class'lar: {', '.join(missing)}")
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


def helmet_color(label: str) -> tuple[int, int, int]:
    if label == "Hardhat":
        return (0, 200, 0)
    if label == "NO-Hardhat":
        return (0, 0, 230)
    return (0, 200, 255)


def readable_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def draw_label(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.65) -> None:
    x, y = xy
    y = max(26, y)
    thickness = 2
    padding = 6
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tw, th = text_size
    x2 = min(frame.shape[1] - 1, x + tw + padding * 2)
    y1 = max(0, y - th - baseline - padding * 2)
    cv2.rectangle(frame, (x, y1), (x2, y + baseline), color, -1)
    cv2.rectangle(frame, (x, y1), (x2, y + baseline), (20, 20, 20), 1)
    cv2.putText(frame, text, (x + padding, y - padding),
                cv2.FONT_HERSHEY_SIMPLEX, scale, readable_text_color(color), thickness, cv2.LINE_AA)


def draw_panel(frame, lines: list[str]) -> None:
    if not lines:
        return
    scale, thickness, padding, lh = 0.65, 2, 10, 26
    pw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for l in lines)
    pw = min(frame.shape[1] - 1, pw + padding * 2)
    ph = padding * 2 + lh * len(lines)
    cv2.rectangle(frame, (8, 8), (8 + pw, 8 + ph), (25, 25, 25), -1)
    cv2.rectangle(frame, (8, 8), (8 + pw, 8 + ph), (240, 240, 240), 1)
    for i, line in enumerate(lines):
        y = 8 + padding + 18 + i * lh
        cv2.putText(frame, line, (8 + padding, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thickness, cv2.LINE_AA)


def resize_for_display(frame, max_width: int, max_height: int):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def show_fixed_window(window_name: str, frame, config: dict[str, Any]) -> None:
    display = config.get("display", {})
    cv2.imshow(window_name, resize_for_display(frame, int(display.get("width", 1280)), int(display.get("height", 720))))


def build_runtime(person_model: YOLO, helmet_model: YOLO, config: dict[str, Any]) -> dict[str, Any]:
    agents = config["agents"]
    tracking = config["tracking"]
    person_cfg = AgentConfig(agents["person"]["classes"], float(agents["person"]["conf"]))
    hardhat_cfg = AgentConfig(agents["hardhat"]["classes"], float(agents["hardhat"]["conf"]))
    return {
        "person": person_cfg,
        "hardhat": hardhat_cfg,
        "person_ids": class_ids(person_model, person_cfg.class_names),
        "hardhat_ids": class_ids(helmet_model, hardhat_cfg.class_names),
        "tracker": str(tracking["tracker"]),
        "imgsz": int(tracking["imgsz"]),
        "crop_padding": float(tracking["crop_padding"]),
        "temporal_window": int(tracking["temporal_window"]),
    }


def process_frame(
    person_model: YOLO,
    helmet_model: YOLO,
    frame,
    runtime: dict[str, Any],
    states,
    device: str,
    overlay: dict[str, Any] | None = None,
):
    person_result = person_model.track(
        frame,
        classes=runtime["person_ids"],
        tracker=runtime["tracker"],
        persist=True,
        imgsz=runtime["imgsz"],
        conf=runtime["person"].conf,
        device=device,
        verbose=False,
    )[0]

    boxes = person_result.boxes
    if boxes is None or boxes.id is None:
        if overlay is not None:
            draw_panel(frame, [f"Model: {overlay['helmet_model']}", "Tracks: 0"])
        return frame

    track_count = len(boxes.id)
    for box, tid in zip(boxes.xyxy, boxes.id):
        track_id = int(tid)
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = crop_with_padding(frame, [x1, y1, x2, y2], runtime["crop_padding"])

        result = helmet_model.predict(
            crop,
            classes=runtime["hardhat_ids"],
            imgsz=runtime["imgsz"],
            conf=runtime["hardhat"].conf,
            device=device,
            verbose=False,
        )[0]
        label, conf = best_detection(helmet_model, result, runtime["hardhat_ids"], runtime["hardhat"].conf)

        states[track_id].append(label)
        voted = vote(states[track_id])
        color = helmet_color(voted)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.circle(frame, (x1, y1), 5, color, -1)
        draw_label(frame, f"ID {track_id} | {voted} {conf:.2f}", (x1, y1 - 10), color)

    if overlay is not None:
        draw_panel(frame, [
            f"Model: {overlay['helmet_model']}",
            f"Person: {overlay['person_model']}",
            f"Tracks: {track_count}",
        ])

    return frame


def run_live(
    person_model: YOLO,
    helmet_model: YOLO,
    source,
    person_model_key: str,
    helmet_model_key: str,
    config: dict[str, Any],
    device: str,
) -> None:
    runtime = build_runtime(person_model, helmet_model, config)
    states = defaultdict(lambda: deque(maxlen=runtime["temporal_window"]))

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise ValueError(f"Kaynak acilamadi: {source}")

    window_name = "CropHelmet Agent - q ile cik"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Live view basladi. Cikmak icin 'q' tusuna bas.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(
            person_model, helmet_model, frame, runtime, states, device,
            overlay={"helmet_model": helmet_model_key, "person_model": person_model_key},
        )
        show_fixed_window(window_name, frame, config)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(
    person_model: YOLO,
    helmet_model: YOLO,
    source: Path,
    person_model_key: str,
    helmet_model_key: str,
    config: dict[str, Any],
    device: str,
) -> None:
    runtime = build_runtime(person_model, helmet_model, config)
    states = defaultdict(lambda: deque(maxlen=runtime["temporal_window"]))

    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Gorsel acilamadi: {source}")

    frame = process_frame(
        person_model, helmet_model, frame, runtime, states, device,
        overlay={"helmet_model": helmet_model_key, "person_model": person_model_key},
    )

    window_name = "CropHelmet Agent - q ile cik"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Gorsel gosteriliyor. Cikmak icin 'q' tusuna bas.")

    while True:
        show_fixed_window(window_name, frame, config)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def run_agent_video(
    person_model: YOLO,
    helmet_model: YOLO,
    source: Path,
    helmet_model_key: str,
    config: dict[str, Any],
    device: str,
    max_frames: int | None,
) -> None:
    runtime = build_runtime(person_model, helmet_model, config)
    states = defaultdict(lambda: deque(maxlen=runtime["temporal_window"]))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = repo_root() / "runs" / "crophelmet" / f"agent_{helmet_model_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.stem}_crophelmet.mp4"

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = process_frame(
            person_model, helmet_model, frame, runtime, states, device,
            overlay={"helmet_model": helmet_model_key, "person_model": "selected"},
        )
        writer.write(frame)
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    print(f"Video kaydedildi: {output_path}")
    print("Son durumlar:")
    for tid, state in sorted(states.items()):
        print(f"  ID {tid}: {vote(state)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop-based helmet detection agent (sadece kask tespiti).")
    parser.add_argument("--source", help="Gorsel veya video yolu. Belirtilmezse dosya secici acilir.")
    parser.add_argument("--person-model", choices=["nano", "small"], default="small")
    parser.add_argument("--mode", choices=["live", "agent-video"], default="live")
    parser.add_argument("--config", default="configs/crophelmet_agent.yaml")
    parser.add_argument("--device", help="CUDA device (orn: 0) veya cpu.")
    parser.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--crop-padding", type=float)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))

    if args.tracker is not None:
        config.setdefault("tracking", {})["tracker"] = args.tracker
    if args.crop_padding is not None:
        config.setdefault("tracking", {})["crop_padding"] = args.crop_padding

    helmet_model_path = resolve_path(config["models"]["crophelmet"]["path"])
    person_model_path = resolve_path(config["person_models"][args.person_model]["path"])
    source = resolve_source(args.source)
    device = select_device(str(args.device if args.device is not None else config.get("tracking", {}).get("device", 0)))

    if not helmet_model_path.exists():
        raise SystemExit(f"Helmet model bulunamadi: {helmet_model_path}")
    if not person_model_path.exists():
        raise SystemExit(f"Person model bulunamadi: {person_model_path}")
    if isinstance(source, Path) and not source.exists():
        raise SystemExit(f"Kaynak dosya bulunamadi: {source}")

    helmet_model = YOLO(str(helmet_model_path))
    person_model = YOLO(str(person_model_path))
    print(f"Helmet model: {helmet_model_path.name}  classes={helmet_model.names}")
    print(f"Person model: {person_model_path.name}")
    print(f"Device: {device}")

    if args.mode == "live":
        if isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTENSIONS:
            run_image(person_model, helmet_model, source, args.person_model, "crophelmet", config, device)
        else:
            run_live(person_model, helmet_model, source, args.person_model, "crophelmet", config, device)
        return

    if not isinstance(source, Path) or source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit("--mode agent-video icin video kaynak gerekli.")
    run_agent_video(person_model, helmet_model, source, "crophelmet", config, device, args.max_frames)


if __name__ == "__main__":
    main()
