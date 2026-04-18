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
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
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
        title="PPE test kaynagini sec",
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
    gpu_requirements = repo_root() / "requirements-gpu.txt"
    raise SystemExit(
        "GPU secildi ama bu Python ortaminda CUDA aktif degil. "
        f"Once su komutu calistir: python -m pip install --force-reinstall -r \"{gpu_requirements}\" --timeout 1000 --retries 10"
    )


def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {name: class_id for class_id, name in model.names.items()}
    missing = [name for name in names if name not in name_to_id]
    if missing:
        raise ValueError(f"Model is missing expected classes: {', '.join(missing)}")
    return [name_to_id[name] for name in names]


def crop_with_padding(frame, box, pad_ratio: float):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(box_width * pad_ratio)
    pad_y = int(box_height * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def best_detection(model: YOLO, result, allowed_ids: list[int], min_conf: float) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id not in allowed_ids or conf < min_conf:
            continue
        label = str(model.names[cls_id])
        if best is None or conf > best[1]:
            best = (label, conf)
    return best if best else ("unknown", 0.0)


def vote(values: deque[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def readable_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = (0.114 * color[0]) + (0.587 * color[1]) + (0.299 * color[2])
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def draw_label(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.65) -> None:
    x, y = xy
    y = max(26, y)
    thickness = 2
    padding = 6
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_width, text_height = text_size
    x2 = min(frame.shape[1] - 1, x + text_width + padding * 2)
    y1 = max(0, y - text_height - baseline - padding * 2)
    cv2.rectangle(frame, (x, y1), (x2, y + baseline), color, -1)
    cv2.rectangle(frame, (x, y1), (x2, y + baseline), (20, 20, 20), 1)
    cv2.putText(
        frame,
        text,
        (x + padding, y - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        readable_text_color(color),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, lines: list[str]) -> None:
    if not lines:
        return
    scale = 0.65
    thickness = 2
    padding = 10
    line_height = 26
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines)
    panel_width = min(frame.shape[1] - 1, width + padding * 2)
    panel_height = padding * 2 + line_height * len(lines)
    cv2.rectangle(frame, (8, 8), (8 + panel_width, 8 + panel_height), (25, 25, 25), -1)
    cv2.rectangle(frame, (8, 8), (8 + panel_width, 8 + panel_height), (240, 240, 240), 1)
    for index, line in enumerate(lines):
        y = 8 + padding + 18 + index * line_height
        cv2.putText(frame, line, (8 + padding, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compliance_status(hardhat: str, vest: str) -> tuple[str, tuple[int, int, int]]:
    hardhat_ok = hardhat in {"hat", "Hardhat"}
    vest_ok = vest in {"vest", "Safety Vest"}
    hardhat_missing = hardhat in {"nohat", "NO-Hardhat"}
    vest_missing = vest in {"novest", "NO-Safety Vest"}

    if hardhat_ok and vest_ok:
        return "OK", (0, 190, 0)
    if hardhat_missing and vest_missing:
        return "NO HARDHAT + NO VEST", (0, 0, 230)
    if hardhat_missing and vest_ok:
        return "NO HARDHAT", (0, 140, 255)
    if hardhat_ok and vest_missing:
        return "NO VEST", (0, 140, 255)
    if hardhat_missing:
        return "NO HARDHAT / VEST ?", (0, 215, 255)
    if vest_missing:
        return "HARDHAT ? / NO VEST", (0, 215, 255)
    return "UNKNOWN", (0, 215, 255)


def resize_for_display(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def show_fixed_window(window_name: str, frame, config: dict[str, Any]) -> None:
    display = config.get("display", {})
    max_width = int(display.get("width", 1280))
    max_height = int(display.get("height", 720))
    cv2.imshow(window_name, resize_for_display(frame, max_width, max_height))


def run_direct_predict(model: YOLO, source: Path, imgsz: int, conf: float, device: str, model_key: str) -> None:
    model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=True,
        project=str(repo_root() / "runs" / "wesjos"),
        name=f"direct_{model_key}",
        exist_ok=True,
    )


def model_agent_classes(config: dict[str, Any], ppe_model_key: str, agent_name: str) -> list[str]:
    model_config = config.get("models", {}).get(ppe_model_key, {})
    model_agents = model_config.get("agents", {})
    if agent_name in model_agents and "classes" in model_agents[agent_name]:
        return list(model_agents[agent_name]["classes"])
    return list(config["agents"][agent_name]["classes"])


def build_agent_runtime(
    person_model: YOLO,
    ppe_model: YOLO,
    config: dict[str, Any],
    ppe_model_key: str = "vinayak_ppe",
) -> dict[str, Any]:
    agents = config["agents"]
    tracking = config["tracking"]

    person = AgentConfig(agents["person"]["classes"], float(agents["person"]["conf"]))
    hardhat = AgentConfig(model_agent_classes(config, ppe_model_key, "hardhat"), float(agents["hardhat"]["conf"]))
    vest = AgentConfig(model_agent_classes(config, ppe_model_key, "vest"), float(agents["vest"]["conf"]))

    return {
        "person": person,
        "hardhat": hardhat,
        "vest": vest,
        "person_ids": class_ids(person_model, person.class_names),
        "hardhat_ids": class_ids(ppe_model, hardhat.class_names),
        "vest_ids": class_ids(ppe_model, vest.class_names),
        "tracker": str(tracking["tracker"]),
        "imgsz": int(tracking["imgsz"]),
        "crop_padding": float(tracking["crop_padding"]),
        "temporal_window": int(tracking["temporal_window"]),
    }


def override_tracking(config: dict[str, Any], tracker: str | None, crop_padding: float | None) -> None:
    tracking = config.setdefault("tracking", {})
    if tracker is not None:
        tracking["tracker"] = tracker
    if crop_padding is not None:
        tracking["crop_padding"] = crop_padding


def process_agent_frame(
    person_model: YOLO,
    ppe_model: YOLO,
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
            draw_panel(frame, [f"PPE: {overlay['ppe_model']}", f"Person: {overlay['person_model']}", "Tracks: 0"])
        return frame

    track_count = len(boxes.id)
    for box, track_id_tensor in zip(boxes.xyxy, boxes.id):
        track_id = int(track_id_tensor)
        x1, y1, x2, y2 = map(int, box.tolist())
        person_crop = crop_with_padding(frame, [x1, y1, x2, y2], runtime["crop_padding"])

        hardhat_result = ppe_model.predict(
            person_crop,
            classes=runtime["hardhat_ids"],
            imgsz=runtime["imgsz"],
            conf=runtime["hardhat"].conf,
            device=device,
            verbose=False,
        )[0]
        hardhat_label, hardhat_conf = best_detection(
            ppe_model, hardhat_result, runtime["hardhat_ids"], runtime["hardhat"].conf
        )

        vest_result = ppe_model.predict(
            person_crop,
            classes=runtime["vest_ids"],
            imgsz=runtime["imgsz"],
            conf=runtime["vest"].conf,
            device=device,
            verbose=False,
        )[0]
        vest_label, vest_conf = best_detection(ppe_model, vest_result, runtime["vest_ids"], runtime["vest"].conf)

        states[track_id]["hardhat"].append(hardhat_label)
        states[track_id]["vest"].append(vest_label)

        hardhat_vote = vote(states[track_id]["hardhat"])
        vest_vote = vote(states[track_id]["vest"])
        status, color = compliance_status(hardhat_vote, vest_vote)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.circle(frame, (x1, y1), 5, color, -1)
        draw_label(
            frame,
            f"ID {track_id} | {status} | H:{hardhat_vote} {hardhat_conf:.2f} | V:{vest_vote} {vest_conf:.2f}",
            (x1, y1 - 10),
            color,
        )

    if overlay is not None:
        draw_panel(
            frame,
            [
                f"PPE: {overlay['ppe_model']}",
                f"Person: {overlay['person_model']}",
                f"Tracker: {runtime['tracker']} | Padding: {runtime['crop_padding']}",
                f"Tracks: {track_count}",
            ],
        )

    return frame


def run_agent_video(
    person_model: YOLO,
    ppe_model: YOLO,
    source: Path,
    model_key: str,
    config: dict[str, Any],
    device: str,
    max_frames: int | None,
) -> None:
    runtime = build_agent_runtime(person_model, ppe_model, config, ppe_model_key=model_key)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = repo_root() / "runs" / "wesjos" / f"agent_{model_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.stem}_agent_demo.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    states = defaultdict(
        lambda: {
            "hardhat": deque(maxlen=runtime["temporal_window"]),
            "vest": deque(maxlen=runtime["temporal_window"]),
        }
    )
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame = process_agent_frame(
            person_model,
            ppe_model,
            frame,
            runtime,
            states,
            device,
            overlay={"ppe_model": model_key, "person_model": "selected"},
        )

        writer.write(frame)

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    print(f"Saved agent demo video: {output_path}")
    print("Final track states:")
    for track_id, state in sorted(states.items()):
        print(f"  ID {track_id}: hardhat={vote(state['hardhat'])}, vest={vote(state['vest'])}")


def run_live_camera(
    person_model: YOLO,
    ppe_model: YOLO,
    source: int | Path,
    model_key: str,
    person_model_key: str,
    config: dict[str, Any],
    device: str,
) -> None:
    runtime = build_agent_runtime(person_model, ppe_model, config, ppe_model_key=model_key)
    states = defaultdict(
        lambda: {
            "hardhat": deque(maxlen=runtime["temporal_window"]),
            "vest": deque(maxlen=runtime["temporal_window"]),
        }
    )

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open live source: {source}")

    window_name = "Wesjos PPE Agent Live - press q to quit"
    print("Live view started. Press q in the video window to quit.")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = process_agent_frame(
            person_model,
            ppe_model,
            frame,
            runtime,
            states,
            device,
            overlay={"ppe_model": model_key, "person_model": person_model_key},
        )
        show_fixed_window(window_name, frame, config)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_live_image(
    person_model: YOLO,
    ppe_model: YOLO,
    source: Path,
    model_key: str,
    person_model_key: str,
    config: dict[str, Any],
    device: str,
) -> None:
    runtime = build_agent_runtime(person_model, ppe_model, config, ppe_model_key=model_key)
    states = defaultdict(
        lambda: {
            "hardhat": deque(maxlen=runtime["temporal_window"]),
            "vest": deque(maxlen=runtime["temporal_window"]),
        }
    )

    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Could not open image source: {source}")

    frame = process_agent_frame(
        person_model,
        ppe_model,
        frame,
        runtime,
        states,
        device,
        overlay={"ppe_model": model_key, "person_model": person_model_key},
    )
    window_name = "Wesjos PPE Agent Image - press q to quit"
    print("Image view opened. Press q in the image window to quit.")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        show_fixed_window(window_name, frame, config)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test wesjos hard-hat/safety-vest pretrained models.")
    parser.add_argument("--source", help="Image or video path. If omitted, a file picker opens on Desktop.")
    parser.add_argument("--model", choices=["vinayak_ppe"], default="vinayak_ppe")
    parser.add_argument(
        "--person-model",
        choices=["nano", "small"],
        default="small",
    )
    parser.add_argument("--mode", choices=["live", "direct", "agent-video"], default="live")
    parser.add_argument("--config", default="configs/pretrained_wesjos.yaml")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="CUDA device id such as 0, or cpu. Defaults to config tracking.device.")
    parser.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"], help="Tracker config for person IDs.")
    parser.add_argument("--crop-padding", type=float, help="Extra padding around each person crop, e.g. 0.30.")
    parser.add_argument("--max-frames", type=int, help="Limit frames for quick video tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    override_tracking(config, tracker=args.tracker, crop_padding=args.crop_padding)
    model_path = resolve_path(config["models"][args.model]["path"])
    person_model_path = resolve_path(config["person_models"][args.person_model]["path"])
    source = resolve_source(args.source)
    device = select_device(str(args.device if args.device is not None else config.get("tracking", {}).get("device", 0)))

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not person_model_path.exists():
        raise SystemExit(f"Person model file not found: {person_model_path}")
    if isinstance(source, Path) and not source.exists():
        raise SystemExit(f"Source file not found: {source}")

    ppe_model = YOLO(str(model_path))
    person_model = YOLO(str(person_model_path))
    print(f"Loaded PPE model: {model_path}")
    print(f"Loaded person model: {person_model_path}")
    print(f"PPE classes: {ppe_model.names}")
    print(f"Person classes: {person_model.names}")
    print(f"Device: {device}")

    if args.mode == "live":
        if isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTENSIONS:
            run_live_image(
                person_model,
                ppe_model,
                source,
                model_key=args.model,
                person_model_key=args.person_model,
                config=config,
                device=device,
            )
            return
        run_live_camera(
            person_model,
            ppe_model,
            source,
            model_key=args.model,
            person_model_key=args.person_model,
            config=config,
            device=device,
        )
        return

    if args.mode == "direct":
        if not isinstance(source, Path):
            raise SystemExit("--mode direct requires an image or video path.")
        run_direct_predict(ppe_model, source, imgsz=args.imgsz, conf=args.conf, device=device, model_key=args.model)
        return

    if not isinstance(source, Path) or source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit("--mode agent-video requires a video source.")
    run_agent_video(
        person_model,
        ppe_model,
        source,
        model_key=args.model,
        config=config,
        device=device,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
