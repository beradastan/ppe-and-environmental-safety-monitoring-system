from __future__ import annotations

import io
import shutil
import socket
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT     = Path(__file__).parent.resolve()
VENV_PY  = ROOT / ".venv" / "Scripts" / "python.exe"
FRONTEND = ROOT / "frontend"
BACKEND_PORT  = 5050
FRONTEND_PORT = 5173

def _c(code: str, text: str) -> str:
    codes = {"cyan": "36", "green": "32", "yellow": "33", "red": "31", "gray": "90", "bold": "1"}
    return f"\033[{codes[code]}m{text}\033[0m"

def header() -> None:
    print()
    print()

def step(msg: str)  -> None: print(f"  {_c('cyan',  '►')} {msg}")
def ok(msg: str)    -> None: print(f"  {_c('green', '✓')} {msg}")
def warn(msg: str)  -> None: print(f"  {_c('yellow','⚠')} {msg}")
def fail(msg: str)  -> None: print(f"  {_c('red',   '✗')} {msg}")

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def _powershell() -> str:
    candidates = [
        shutil.which("powershell"),
        shutil.which("pwsh"),
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("PowerShell bulunamadi")

def open_terminal(title: str, cmd: str) -> None:
    subprocess.Popen(
        [_powershell(), "-NoExit", "-Command",
         f"$Host.UI.RawUI.WindowTitle = '{title}'; {cmd}"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

def check_env() -> bool:
    step("Ortam kontrol ediliyor...")
    ok_flag = True

    if not VENV_PY.exists():
        fail("Python sanal ortami bulunamadi: .venv")
        print(_c("gray", "       Cozum: python -m venv .venv  &&  .venv\\Scripts\\pip install -r requirements.txt"))
        ok_flag = False
    else:
        ok("Python sanal ortami mevcut")

    if not (FRONTEND / "node_modules").exists():
        fail("Frontend bagimliliklari eksik: frontend\\node_modules")
        print(_c("gray", "       Cozum: cd frontend  &&  npm install"))
        ok_flag = False
    else:
        ok("Frontend bagimliliklari mevcut")

    return ok_flag

def start_backend() -> None:
    if port_in_use(BACKEND_PORT):
        warn(f"Port {BACKEND_PORT} zaten kullanimda — Backend atlaniyor")
        return
    step(f"Backend baslatiliyor  (http://localhost:{BACKEND_PORT})...")
    cmd = f"Set-Location '{ROOT}'; & '{VENV_PY}' -m backend.app"
    open_terminal(f"Backend (:{BACKEND_PORT})", cmd)
    ok("Backend penceresi acildi")

def start_frontend() -> None:
    if port_in_use(FRONTEND_PORT):
        warn(f"Port {FRONTEND_PORT} zaten kullanimda — Frontend atlaniyor")
        return
    npm = shutil.which("npm") or "npm"
    step(f"Frontend baslatiliyor  (http://localhost:{FRONTEND_PORT})...")
    cmd = f"Set-Location '{FRONTEND}'; {npm} run dev"
    open_terminal(f"Frontend (:{FRONTEND_PORT})", cmd)
    ok("Frontend penceresi acildi")

def summary() -> None:
    sep = _c("gray", "  " + "─" * 45)
    print()
    print(sep)
    print(f"  Backend   →  http://localhost:{BACKEND_PORT}")
    print(f"  Frontend  →  http://localhost:{FRONTEND_PORT}")
    print()
    print(_c("gray", "  Pipeline icin CameraSetup sayfasini kullanin veya:"))
    print(_c("gray", "  python pipeline/run_live_video.py --mode crop --camera 0 \\"))
    print(_c("gray", "    --camera-id cam_01 --zone \"Uretim Hatti A\" --display"))
    print(sep)
    print()

def main() -> None:
    header()
    if not check_env():
        sys.exit(1)
    print()
    start_backend()
    start_frontend()
    summary()

if __name__ == "__main__":
    main()
