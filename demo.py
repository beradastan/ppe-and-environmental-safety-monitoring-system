# -*- coding: utf-8 -*-
"""
demo.py  —  Factory Safety demo başlatıcı
==========================================
Backend, frontend ve tarayıcıyı sırayla başlatır.
Pipeline komutlarını ekrana yazar.

Kullanım:
    python demo.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent

# Windows'ta yeni konsol penceresi aç
_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"

BANNER = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════╗
║          Factory Safety Monitor  —  Demo         ║
╚══════════════════════════════════════════════════╝{RESET}
"""

PIPELINE_HELP = f"""
{BOLD}Pipeline başlatmak için yeni bir terminal açın:{RESET}

  Test videosu:
    python run_live_video.py --video test/intel_safety_full.mp4 --display
    python run_live_video.py --video test/github_hardhat.mp4 --display
    python run_live_video.py --video test/pexels_construction1.mp4 --display
    python run_live_video.py --video test/veo3_construction.mp4 --display

  Kamera ile (kamera kimliği ve bölge opsiyonel):
    python run_live_video.py --camera 0 --display
    python run_live_video.py --camera 0 --camera-id cam_01 --zone "Uretim Hatti A" --display

{BOLD}Durdurmak için bu pencerede Ctrl+C yapın.{RESET}
"""


def find_npm_cmd() -> str | None:
    """npm.cmd yolunu bul: PATH → PyCharm node → sistem node."""
    # 1) Zaten PATH'te varsa kullan
    found = shutil.which("npm.cmd") or shutil.which("npm")
    if found:
        return found

    # 2) PyCharm'ın indirdiği node sürümlerini tara
    pycharm_base = Path.home() / "AppData/Roaming/JetBrains"
    candidates = sorted(
        pycharm_base.glob("PyCharm*/node/versions/*/npm.cmd"),
        reverse=True,  # en yeni sürüm önce
    )
    if candidates:
        return str(candidates[0])

    # 3) Standart sistem node konumları
    for p in [
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
    ]:
        if Path(p).exists():
            return p

    return None


def step(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg}", flush=True)


def kill_port(port: int) -> None:
    """Windows: verilen TCP portunu dinleyen process'i öldür."""
    if sys.platform != "win32":
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                pid = line.split()[-1]
                subprocess.run(
                    ["taskkill", "/F", "/PID", pid],
                    capture_output=True,
                )
    except Exception:
        pass


def kill_tree(p: subprocess.Popen) -> None:
    """Process ve tüm alt process'lerini öldür."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(p.pid)],
            capture_output=True,
        )
    else:
        p.terminate()


def wait_ready(url: str, timeout: int = 20) -> bool:
    """URL'nin yanıt verdiğini bekle."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    # UTF-8 + ANSI renk desteği (Windows 10+)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        os.system("color")

    print(BANNER)

    # npm yolunu bul
    npm_cmd = find_npm_cmd()
    if not npm_cmd:
        print(f"  {YELLOW}HATA: npm bulunamadı. Node.js kurulu olduğundan emin olun.{RESET}")
        sys.exit(1)

    # npm dizinini PATH'e ekle (yeni konsol penceresi için)
    npm_dir = str(Path(npm_cmd).parent)
    env = os.environ.copy()
    if npm_dir not in env.get("PATH", ""):
        env["PATH"] = npm_dir + os.pathsep + env.get("PATH", "")

    # Önceki çalıştırmadan kalan process'leri temizle
    print("  Eski process'ler temizleniyor...", flush=True)
    kill_port(5050)
    kill_port(5173)
    time.sleep(1)

    procs: list[subprocess.Popen] = []

    try:
        # 1 — Backend
        step(1, 3, "Backend başlatılıyor  (http://localhost:5050) ...")
        backend = subprocess.Popen(
            [sys.executable, "backend/app.py"],
            cwd=ROOT,
            env=env,
            creationflags=_NEW_CONSOLE,
        )
        procs.append(backend)

        if wait_ready("http://localhost:5050/api/stats", timeout=12):
            print(f"       {GREEN}✓ Backend hazır{RESET}")
        else:
            print(f"       {YELLOW}⚠ Backend henüz yanıt vermiyor, devam ediliyor...{RESET}")

        # 2 — Frontend
        step(2, 3, "Frontend başlatılıyor  (http://localhost:5173) ...")
        frontend = subprocess.Popen(
            f'"{npm_cmd}" run dev',
            cwd=ROOT / "frontend",
            env=env,
            shell=True,
            creationflags=_NEW_CONSOLE,
        )
        procs.append(frontend)

        if wait_ready("http://localhost:5173", timeout=20):
            print(f"       {GREEN}✓ Frontend hazır{RESET}")
        else:
            print(f"       {YELLOW}⚠ Frontend yanıt vermiyor — npm/node kurulu mu?{RESET}")
            print(f"         npm yolu: {npm_cmd}")

        # 3 — Tarayıcı
        step(3, 3, "Tarayıcı açılıyor ...")
        time.sleep(1)
        webbrowser.open("http://localhost:5173")
        print(f"       {GREEN}✓ Açıldı{RESET}")

        print(PIPELINE_HELP)

        # Backend kapanana kadar bekle (Ctrl+C ile çıkılır)
        backend.wait()

    except KeyboardInterrupt:
        print(f"\n  Demo durduruluyor...")
    finally:
        for p in procs:
            try:
                kill_tree(p)
            except Exception:
                pass
        print("  Kapatıldı.\n")


if __name__ == "__main__":
    main()
