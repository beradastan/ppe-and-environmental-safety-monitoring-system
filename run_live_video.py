# -*- coding: utf-8 -*-
"""
run_live_video.py
=================
Canlı video akışı için event-bazlı alarm yönetimi.

Kullanım:
    python run_live_video.py
    python run_live_video.py --camera 1
    python run_live_video.py --video test/ppe_test1.mp4
    python run_live_video.py --offline
"""

import sys
import os
import cv2
import json
import time
from pathlib import Path

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

# UTF-8 çıktı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

# ── Renkler ────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def draw_detections(frame, hr, vr, fr):
    """
    Ajan tespit sonuçlarını frame üzerine çiz.
    Bbox'lar 640x640 preprocessed space'den orijinal frame boyutuna scale edilir.
    Tespit pipeline'ına dokunulmaz; sadece görselleştirme.
    """
    h, w = frame.shape[:2]
    sx, sy = w / 640.0, h / 640.0  # scale faktörleri

    # (items_list, renk, kalınlık)
    layers = [
        (hr.get("detections", []), (0, 200,   0), 2),   # Hardhat      → yeşil
        (hr.get("warnings",    []), (0,   0, 220), 2),   # NO-Hardhat   → kırmızı
        (vr.get("detections", []), (200, 200,  0), 2),   # Safety Vest  → mavi-yeşil
        (vr.get("warnings",    []), (0, 140, 255), 2),   # NO-Vest      → turuncu
        (fr.get("detections", []), (0,  80, 255), 2),   # Fire         → kırmızı-turuncu
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for items, color, thickness in layers:
        for det in items:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), font, 0.55, color, 2)

    return frame


def setup_display_frame(frame, event_info):
    """
    Frame üzerine bilgi yaz (ekran gösterimi için).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    event_id = event_info.get("event_id", "N/A")
    event_status = event_info.get("event_status", "idle").upper()

    if event_status in ["NEW", "UPDATE"]:
        color = (0, 0, 255)  # Kırmızı
    elif event_status == "RESOLVED":
        color = (255, 0, 0)  # Mavi
    else:
        color = (0, 255, 0)  # Yeşil

    cv2.putText(
        frame,
        f"EVENT: {event_id} [{event_status}]",
        (10, 30),
        font,
        font_scale,
        color,
        thickness,
    )

    sig = event_info.get("alarm_signature")
    if sig is not None:
        info_text = (
            f"Helmet: {getattr(sig, 'helmet_violation_count', 0)}, "
            f"Vest: {getattr(sig, 'vest_violation_count', 0)}, "
            f"Fire: {getattr(sig, 'fire_detected', False)}"
        )
        cv2.putText(
            frame,
            info_text,
            (10, 70),
            font,
            font_scale,
            color,
            thickness,
        )

    repeat = event_info.get("repeat_count", 0)
    cv2.putText(
        frame,
        f"Repeat: {repeat}",
        (10, 110),
        font,
        font_scale,
        color,
        thickness,
    )

    return frame


def save_event_report(event_info, report_data, results_dir: Path):
    """
    Event raporu + JSON kaydet.
    """
    if not event_info.get("should_save"):
        return None

    results_dir.mkdir(exist_ok=True)
    event_id = event_info.get("event_id", "no_event")
    event_dir = results_dir / event_id
    event_dir.mkdir(exist_ok=True)

    event_status = event_info.get("event_status", "unknown")
    if event_status == "new":
        suffix = "new"
    elif event_status == "update":
        existing = list(event_dir.glob(f"{event_id}_update_*.txt"))
        update_num = len(existing) + 1
        suffix = f"update_{update_num:02d}"
    else:
        suffix = "resolved"

    txt_file = event_dir / f"{event_id}_{suffix}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVENT: {event_id} [{event_status.upper()}]\n")
        f.write(f"Zaman: {report_data.get('timestamp', 'N/A')}\n")
        f.write(f"Tekrar: {event_info.get('repeat_count', 0)}\n")
        f.write(f"Süre: {event_info.get('duration_sec', 0):.1f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("--- CNN TESPİT VERİSİ ---\n")
        f.write(report_data.get("structured", "N/A"))
        f.write("\n--- LLM GÜVENLİK RAPORU ---\n")
        f.write(report_data.get("report", "N/A"))
        f.write("\n")

    json_file = event_dir / f"{event_id}_{suffix}.json"

    # AlarmSignature → frontend'in beklediği signature formatına dönüştür
    sig = event_info.get("alarm_signature")
    if sig is not None:
        h_count = getattr(sig, "helmet_violation_count", 0)
        v_count = getattr(sig, "vest_violation_count", 0)
        signature = {
            "helmet_missing_ids": list(range(1, h_count + 1)),
            "vest_missing_ids":   list(range(1, v_count + 1)),
            "fire_detected":      getattr(sig, "fire_detected", False),
            "fire_confidence":    getattr(sig, "fire_confidence", 0.0),
        }
    else:
        signature = {}

    json_data = {
        "event_id":     event_id,
        "event_status": event_status,
        "timestamp":    report_data.get("timestamp"),
        "repeat_count": event_info.get("repeat_count"),
        "duration_sec": event_info.get("duration_sec"),
        "change_reason": event_info.get("change_reason", ""),
        "signature":    signature,
        "llm_report":   report_data.get("report"),
        "structured":   report_data.get("structured"),
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"  {CYAN}[KAYIT] {event_id}/{suffix}{RESET}")
    return {"txt": txt_file, "json": json_file, "suffix": suffix}


def open_video_source(video_file, camera_idx):
    """
    Video veya kamera kaynağını aç.
    """
    print(f"\n{BOLD}[3/4] Video kaynağı açılıyor...{RESET}")

    if video_file:
        video_path = Path(video_file)

        if not video_path.exists():
            candidate = Path("test") / video_file
            if candidate.exists():
                video_path = candidate

        if not video_path.exists():
            print(f"  {RED}[HATA] Video dosyası bulunamadı: {video_path.resolve()}{RESET}")
            sys.exit(1)

        cap = cv2.VideoCapture(str(video_path.resolve()))
        if not cap.isOpened():
            print(f"  {RED}[HATA] Video açılamadı: {video_path.resolve()}{RESET}")
            sys.exit(1)

        print(f"  {GREEN}✓ Video dosyası: {video_path.resolve()}{RESET}")
        return cap

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"  {RED}[HATA] Kamera açılamadı: {camera_idx}{RESET}")
        sys.exit(1)

    print(f"  {GREEN}✓ Kamera {camera_idx}{RESET}")
    return cap


def main():
    # ── Argümanları işle ───────────────────────────────────────────
    offline = "--offline" in sys.argv
    camera_idx = 0
    video_file = None

    if "--camera" in sys.argv:
        idx = sys.argv.index("--camera") + 1
        if idx < len(sys.argv):
            camera_idx = int(sys.argv[idx])

    if "--video" in sys.argv:
        idx = sys.argv.index("--video") + 1
        if idx < len(sys.argv):
            video_file = sys.argv[idx]

    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  FACTORY SAFETY — CANLI VIDEO + EVENT-BAZLI YÖNETİM{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"  Kaynak  : {'Kamera ' + str(camera_idx) if not video_file else 'Video: ' + video_file}")
    print(f"  LLM     : {'[OFFLINE / Mock]' if offline else '[Ollama / mistral]'}")
    print(f"  Cihaz   : {_DEVICE.upper()}")
    print(f"  Timeout : 10 saniye")
    print(f"  İşleme  : saniyede 1 kez")

    # ── Ajanları yükle ─────────────────────────────────────────────
    print(f"\n{BOLD}[1/4] CNN Ajanları yükleniyor...{RESET}")
    try:
        from agents.specific_agents import HelmetAgent, VestAgent, FireAgent

        helmet_agent = HelmetAgent(device=_DEVICE)
        vest_agent = VestAgent(device=_DEVICE)
        fire_agent = FireAgent(device=_DEVICE)

        print(f"  {GREEN}✓ HelmetAgent{RESET}")
        print(f"  {GREEN}✓ VestAgent{RESET}")
        print(f"  {GREEN}✓ FireAgent{RESET}")
    except Exception as e:
        print(f"  {RED}[HATA] Ajan yüklenemedi: {e}{RESET}")
        sys.exit(1)

    # ── LLM + Event Manager ────────────────────────────────────────
    print(f"\n{BOLD}[2/4] LLM + Event Manager başlatılıyor...{RESET}")
    try:
        from llm.llm_coordinator import OllamaLLMCoordinator
        from event_manager import EventManager

        llm = OllamaLLMCoordinator(model_name="mistral", offline_mode=offline)
        event_manager = EventManager(timeout_sec=10.0)

        print(f"  {GREEN}✓ LLMCoordinator{RESET}")
        print(f"  {GREEN}✓ EventManager (timeout=10s){RESET}")
    except Exception as e:
        print(f"  {RED}[HATA] LLM/EventManager başlatılamadı: {e}{RESET}")
        sys.exit(1)

    # ── Video kaynağını aç ─────────────────────────────────────────
    cap = open_video_source(video_file, camera_idx)

    # ── Ana döngü ayarları ─────────────────────────────────────────
    print(f"\n{BOLD}[4/4] Video işleniyor (ESC = çıkış)...{RESET}\n")

    # Sabit pencere boyutu (1280x720)
    win_w, win_h = 1280, 720
    cv2.namedWindow("Factory Safety - Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Factory Safety - Live", win_w, win_h)

    results_dir = Path("results")
    frame_count = 0
    event_count = 0

    process_interval_sec = 1.0
    last_process_time = 0.0

    last_event_info = {
        "event_id": "N/A",
        "event_status": "idle",
        "repeat_count": 0,
        "alarm_signature": None,
        "should_save": False,
    }
    _empty = {"detections": [], "warnings": []}
    last_hr, last_vr, last_fr = _empty, _empty, _empty

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n{CYAN}Video sonu ulaşıldı.{RESET}")
                break

            frame_count += 1
            report_data = None

            now = time.monotonic()
            should_process_now = (now - last_process_time) >= process_interval_sec

            if should_process_now:
                last_process_time = now

                try:
                    hr = helmet_agent.detect(frame)
                    vr = vest_agent.detect(frame)
                    fr = fire_agent.detect(frame)
                except Exception as e:
                    print(f"{RED}[HATA] Deteksiyon: {e}{RESET}")
                    continue

                last_hr, last_vr, last_fr = hr, vr, fr
                event_info = event_manager.process_frame(hr, vr, fr)
                last_event_info = event_info

                # LLM sadece event kayda değer durumda çağrılsın
                if event_info.get("should_save"):
                    report_data = llm.generate_alarm_report(
                        hr,
                        vr,
                        fr,
                        image_name=f"frame_{frame_count}",
                        use_minimal_format=True,
                    )

                if event_info.get("should_save") and report_data:
                    event_count += 1
                    saved = save_event_report(event_info, report_data, results_dir)

                    if saved:
                        event_id = event_info.get("event_id")
                        suffix = saved["suffix"]
                        event_dir = results_dir / event_id
                        img_path = event_dir / f"{event_id}_{suffix}.jpg"
                        cv2.imwrite(str(img_path), frame)

            display_frame = frame.copy()
            draw_detections(display_frame, last_hr, last_vr, last_fr)
            setup_display_frame(display_frame, last_event_info)
            cv2.imshow("Factory Safety - Live", display_frame)

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | Events: {event_count}")

            if cv2.waitKey(1) & 0xFF == 27:
                print(f"\n{CYAN}Kullanıcı tarafından durduruldu.{RESET}")
                break

    except KeyboardInterrupt:
        print(f"\n{CYAN}Durduruldu.{RESET}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        resolved = event_manager.force_resolve_event()
        if resolved:
            print(f"\n{CYAN}Son event kapatıldı.{RESET}")

        print(f"\n{BOLD}{'=' * 65}{RESET}")
        print("  ÖZET")
        print(f"{BOLD}{'=' * 65}{RESET}")
        print(f"  Toplam frame     : {frame_count}")
        print(f"  Event sayısı     : {event_count}")
        print(f"  Sonuçlar         : results/")
        print(f"{BOLD}{'=' * 65}{RESET}\n")


if __name__ == "__main__":
    main()