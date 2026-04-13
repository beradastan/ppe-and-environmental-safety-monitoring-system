# -*- coding: utf-8 -*-
"""
run_with_llm.py
===============
test/ klasöründeki tüm görüntüleri CNN ajanlarıyla işler,
EVENT-BAZLI alarm yönetimi ile sadece anlamlı değişiklikleri kaydeder.

🎥 FRAME-BAZLI ÇALIŞMA:
- Her frame CNN'den geçer
- Event manager aynı alarmı tekrar sayamaz
- Sadece yeni/update/resolved event'lerde kayıt yapılır

Kullanım:
    python run_with_llm.py              # test/ klasörü (default)
    python run_with_llm.py test/        # özel klasör
    python run_with_llm.py --offline    # Ollama olmadan mock rapor
    python run_with_llm.py --video 0    # Canlı video (0 = default kamera)
"""

import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# UTF-8 çıktı (Windows cp1254 sorununu önler)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

# ── Renkler ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  {RED}[HATA] Gorunru okunamadi: {path.name} -> {e}{RESET}")
        return None


def save_report(report_data: dict, results_dir: Path):
    """
    Raporu results/ klasörüne kaydet.
    Event-bazlı: event_id/evt_0001_start.txt, evt_0001_start.json, vb.
    """
    results_dir.mkdir(exist_ok=True)

    event_id = report_data.get("event_id", "no_event")
    event_status = report_data.get("event_status", "unknown")
    event_dir = results_dir / event_id

    if event_status in ["new", "update", "resolved"]:
        event_dir.mkdir(exist_ok=True)

        # Status suffix
        if event_status == "new":
            suffix = "start"
        elif event_status == "update":
            # Update için sıra numarası
            existing = list(event_dir.glob(f"{event_id}_update_*.txt"))
            update_num = len(existing) + 1
            suffix = f"update_{update_num:02d}"
        else:  # resolved
            suffix = "resolved"

        # TXT kayıt
        txt_file = event_dir / f"{event_id}_{suffix}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"EVENT: {event_id} [{event_status.upper()}]\n")
            f.write(f"Görüntü: {report_data.get('image', 'N/A')}\n")
            f.write(f"Zaman: {report_data.get('timestamp', 'N/A')}\n")
            f.write(f"Tekrar: {report_data.get('repeat_count', 0)}\n")
            f.write(f"Süre: {report_data.get('duration_sec', 0):.1f}s\n")
            if report_data.get('change_reason'):
                f.write(f"Değişim: {report_data['change_reason']}\n")
            f.write("=" * 60 + "\n\n")
            f.write("--- CNN TESPİT VERİSİ ---\n")
            f.write(report_data.get("structured", "N/A"))
            f.write("\n--- LLM GÜVENLİK RAPORU ---\n")
            f.write(report_data.get("report", "N/A"))
            f.write("\n")

        # JSON kayıt
        json_file = event_dir / f"{event_id}_{suffix}.json"
        json_data = {
            "event_id": event_id,
            "event_status": event_status,
            "image": report_data.get("image"),
            "timestamp": report_data.get("timestamp"),
            "repeat_count": report_data.get("repeat_count"),
            "duration_sec": report_data.get("duration_sec"),
            "change_reason": report_data.get("change_reason"),
            "alarm": report_data.get("alarm"),
            "llm_called": report_data.get("llm_called"),
            "structured": report_data.get("structured"),
            "report": report_data.get("report"),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"  {CYAN}[KAYIT] {event_id}/{txt_file.name}{RESET}")
        return txt_file

    return None


def main():
    # ── Argüman işle ───────────────────────────────────────────────
    offline = "--offline" in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    test_dir = Path(args[0]) if args else Path("test")

    if not test_dir.exists():
        print(f"{RED}[HATA] Klasör bulunamadi: {test_dir}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  FACTORY SAFETY — CNN + LLM ALARM PIPELINE{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Klasör  : {test_dir}")
    print(f"  LLM modu: {'[OFFLINE / Mock]' if offline else '[Ollama / mistral]'}")

    # ── Ajanları yükle ─────────────────────────────────────────────
    print(f"\n{BOLD}[1/3] CNN Ajanlar yukleniyor...{RESET}")
    try:
        from agents.specific_agents import HelmetAgent, VestAgent, FireAgent
        helmet_agent = HelmetAgent(device="cpu")
        vest_agent   = VestAgent(device="cpu")
        fire_agent   = FireAgent(device="cpu")
        print(f"  {GREEN}OK HelmetAgent -> {helmet_agent.model_name}{RESET}")
        print(f"  {GREEN}OK VestAgent   -> {vest_agent.model_name}{RESET}")
        print(f"  {GREEN}OK FireAgent   -> {fire_agent.model_name}{RESET}")
    except Exception as e:
        print(f"  {RED}[HATA] Agent yuklenemedi: {e}{RESET}")
        sys.exit(1)

    # ── LLM koordinatörü yükle ─────────────────────────────────────
    print(f"\n{BOLD}[2/3] LLM Koordinatoru baslatiliyor...{RESET}")
    try:
        from llm.llm_coordinator import OllamaLLMCoordinator
        from event_manager import EventManager
        
        llm = OllamaLLMCoordinator(
            model_name="mistral",
            offline_mode=offline
        )
        event_manager = EventManager(timeout_sec=10.0, event_id_prefix="evt")
        
        print(f"  {GREEN}OK LLMCoordinator (model=mistral, offline={offline}){RESET}")
        print(f"  {GREEN}OK EventManager (timeout=10s){RESET}")
    except Exception as e:
        print(f"  {RED}[HATA] Sistem yuklenemedi: {e}{RESET}")
        sys.exit(1)

    # ── Görüntü listesi ────────────────────────────────────────────
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    images = sorted({p for ext in exts for p in test_dir.glob(ext)}, key=lambda p: p.name)

    if not images:
        print(f"\n{RED}[HATA] Goruntu bulunamadi: {test_dir}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}[3/3] {len(images)} goruntu isleniyor...{RESET}")
    print("─" * 65)

    results_dir = Path("results")
    saved_reports = []
    alarm_count = 0

    for idx, img_path in enumerate(images, 1):
        print(f"\n{BOLD}[{idx}/{len(images)}] {img_path.name}{RESET}")

        img = load_image(img_path)
        if img is None:
            continue

        # CNN tespitleri
        try:
            hr = helmet_agent.detect(img)
            vr = vest_agent.detect(img)
            fr = fire_agent.detect(img)
        except Exception as e:
            print(f"  {RED}[HATA] Tespit hatasi: {e}{RESET}")
            continue

        # CNN özet satırı
        h_str = f"{hr.get('detection_count',0)}OK/{hr.get('warning_count',0)}!!"
        v_str = f"{vr.get('detection_count',0)}OK/{vr.get('warning_count',0)}!!"
        f_str = f"{fr.get('detection_count',0)} yangın"
        
        # Event manager ile alarm yönet
        event_info = event_manager.process_frame(hr, vr, fr)
        has_event = event_info["event_status"] in ["new", "update", "resolved"]
        
        if has_event:
            alarm_count += 1
            alarm_label = f"{RED}[ALARM]{RESET}"
            event_status_str = f" [{event_info['event_status'].upper()}]"
        else:
            alarm_label = f"{GREEN}[GUVENLI]{RESET}"
            event_status_str = ""
        
        print(f"  CNN  | Helmet:{h_str}  Vest:{v_str}  Fire:{f_str}  {alarm_label}{event_status_str}")

        # LLM raporu (sadece alarm başladığında)
        report_data = None
        if event_info.get("event_status") == "new":
            report_data = llm.generate_alarm_report(
                hr, vr, fr, image_name=img_path.name,
                use_minimal_format=True,
                event_info=event_info
            )

        # Rapor yazdır (varsa)
        if report_data and report_data.get("llm_called"):
            print(f"\n  {BOLD}[LLM RAPORU]{RESET}")
            for line in report_data["report"].splitlines():
                print(f"  {YELLOW}{line}{RESET}")
        
        # Sadece event varsa ve rapor varsa kayıt
        if event_info.get("should_save") and report_data:
            out = save_report(report_data, results_dir)
            if out:
                saved_reports.append(out)

    # ── Genel özet ─────────────────────────────────────────────────
    print(f"\n\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  OZET{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Toplam goruntu   : {len(images)}")
    print(f"  Alarm tetiklenen : {alarm_count}")
    print(f"  LLM raporu uretilen: {len(saved_reports)}")
    if saved_reports:
        print(f"\n  Kaydedilen raporlar:")
        for r in saved_reports:
            print(f"    -> results/{r.name}")
    print(f"{BOLD}{'='*65}{RESET}\n")


if __name__ == "__main__":
    main()
