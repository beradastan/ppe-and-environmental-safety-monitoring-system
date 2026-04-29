# -*- coding: utf-8 -*-
"""
test_llm_options.py
===================
LLM rapor seçeneklerini 5 farklı senaryo ile test eder.
Her seçenek aynı senaryolara karşı çalıştırılır.

Kullanım:
    python test_llm_options.py
"""
from __future__ import annotations
import yaml, requests, time, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────

with open("config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}
LLM_CFG = _cfg.get("llm", {})
BASE_URL = LLM_CFG.get("base_url", "http://localhost:11434")
MODEL    = LLM_CFG.get("model", "mistral")
TEMP     = LLM_CFG.get("temperature", 0.1)
TIMEOUT  = LLM_CFG.get("timeout", 120)

# ── Test senaryoları ─────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label": "Tek kişi, tek ihlal (baret eksik)",
        "persons": [{"track_id": 2, "violations": ["no_helmet"]}],
        "scene": {},
        "total_persons": 4,
        "repeat_count": 1,
        "event_type": "ppe_violation",
    },
    {
        "label": "İki kişi, farklı ihlaller",
        "persons": [
            {"track_id": 1, "violations": ["no_helmet", "no_vest"]},
            {"track_id": 3, "violations": ["no_vest"]},
        ],
        "scene": {},
        "total_persons": 5,
        "repeat_count": 1,
        "event_type": "ppe_violation",
    },
    {
        "label": "Bir kişi, hiçbir KKD yok",
        "persons": [{"track_id": 5, "violations": ["no_helmet", "no_vest", "no_mask"]}],
        "scene": {},
        "total_persons": 3,
        "repeat_count": 1,
        "event_type": "ppe_violation",
    },
    {
        "label": "Yangın tespiti (PPE yok)",
        "persons": [],
        "scene": {"fire_detected": True},
        "total_persons": 0,
        "repeat_count": 1,
        "event_type": "fire_detected",
    },
    {
        "label": "3. tekrar — çoklu tehlike (yangın + PPE)",
        "persons": [
            {"track_id": 2, "violations": ["no_helmet"]},
            {"track_id": 4, "violations": ["no_vest"]},
        ],
        "scene": {"fire_detected": True},
        "total_persons": 6,
        "repeat_count": 3,
        "event_type": "multi_hazard",
    },
]

# ── Ortak yardımcı ───────────────────────────────────────────────────────────

_VIO_LABEL = {"no_helmet": "baret", "no_vest": "yelek", "no_mask": "maske"}


def _scenario_to_input(sc: dict) -> str:
    """Senaryo → LLM'e gönderilecek 'Input:' bloğu."""
    persons      = sc["persons"]
    scene        = sc["scene"]
    repeat_count = sc["repeat_count"]
    total_persons = sc["total_persons"]
    event_type   = sc["event_type"]

    violator_count = len(persons)
    repeat_suffix  = f" — {repeat_count}. tekrar" if repeat_count > 1 else ""

    if event_type == "fire_detected":
        type_tr = "Yangın/duman"
    elif event_type == "ppe_violation":
        type_tr = f"KKD ihlali ({violator_count} kişi)" if violator_count > 1 else "KKD ihlali"
    else:
        type_tr = f"Çoklu tehlike ({violator_count} kişi)" if violator_count > 1 else "Çoklu tehlike"

    lines = [f"Olay tipi: {type_tr}{repeat_suffix}"]

    if scene.get("fire_detected"):
        lines.append("Sahne: yangın tespit edildi")
    if scene.get("smoke_detected"):
        lines.append("Sahne: duman tespit edildi")

    if total_persons > 0 and violator_count > 0:
        lines.append(f"Sahada: {total_persons} kişi ({violator_count} ihlal)")

    for p in persons:
        viols  = p.get("violations", [])
        active = [v for v in viols if v in _VIO_LABEL]
        if len(active) >= 2:
            lines.append(f"Kişi #{p['track_id']}: hiçbir KKD yok")
        else:
            parts = [_VIO_LABEL[v] + "=YOK" for v in active]
            lines.append(f"Kişi #{p['track_id']}: {', '.join(parts)}")

    return "\n".join(lines)


def _call(prompt: str, system: str | None = None) -> str:
    body: dict = {
        "model": MODEL, "prompt": prompt,
        "temperature": TEMP, "stream": False,
    }
    if system:
        body["system"] = system
    try:
        r = requests.post(f"{BASE_URL}/api/generate", json=body, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"[HTTP {r.status_code}]"
    except Exception as e:
        return f"[HATA: {e}]"


# ── Seçenek 1 — Yapılandırılmış (Durum / Risk / Aksiyon) ────────────────────

_SYS_1 = (
    "Sen bir fabrika iş güvenliği AI analistisin.\n"
    "Verilen olay verilerine dayanarak TAM OLARAK şu 3 satırlı formatı döndür:\n"
    "Durum: <tespiti 1 cümlede özetle>\n"
    "Risk: <DÜŞÜK / ORTA / YÜKSEK> — <kısa gerekçe>\n"
    "Aksiyon: <derhal yapılması gereken tek eylem>\n\n"
    "Kurallar:\n"
    "- Sadece verilen verileri kullan, bilgi uydurma.\n"
    "- Her satır tam olarak o etiketle başlamalı.\n"
    "- Türkçe yaz.\n\n"
    "Örnek:\n"
    "Olay tipi: KKD ihlali\nSahada: 3 kişi (1 ihlal)\nKişi #2: baret=YOK\n"
    "---\n"
    "Durum: Sahada 3 kişiden 1'i (Kişi #2) baret takmıyor.\n"
    "Risk: ORTA — tek kişi, tek ihlal türü.\n"
    "Aksiyon: Kişi #2'yi derhal baret takması için uyarın.\n\n"
    "Örnek:\n"
    "Olay tipi: Çoklu tehlike (2 kişi)\nSahne: yangın tespit edildi\nSahada: 6 kişi (2 ihlal)\nKişi #2: baret=YOK\nKişi #4: yelek=YOK\n"
    "---\n"
    "Durum: Sahada yangın tespit edildi; kişi #2 baret, kişi #4 yelek takmıyor.\n"
    "Risk: YÜKSEK — aktif yangın riski ve eksik KKD bir arada.\n"
    "Aksiyon: Alanı tahliye edin ve yangın ekibini çağırın.\n\n"
    "Şimdi sadece 3 satırı yaz, başka hiçbir şey yazma."
)


def option1(sc: dict) -> str:
    inp = _scenario_to_input(sc)
    return _call(inp, system=_SYS_1)


# ── Seçenek 2 — 2-3 cümle serbest analiz ────────────────────────────────────

_SYS_2 = (
    "Sen bir fabrika iş güvenliği uzmanısın.\n"
    "Verilen olay verisini analiz et ve profesyonel, akıcı Türkçe ile 2-3 cümle yaz.\n"
    "İlk cümle: Ne tespit edildi.\n"
    "İkinci cümle: Bu durumun riski veya önemi.\n"
    "Üçüncü cümle (gerekirse): Önerilen acil eylem.\n\n"
    "Kurallar:\n"
    "- Sadece verilen verileri kullan.\n"
    "- Resmi ve net bir dil kullan.\n"
    "- Kişi numaralarını (#ID) doğru kullan.\n\n"
    "Örnek:\n"
    "Olay tipi: KKD ihlali\nSahada: 3 kişi (1 ihlal)\nKişi #2: baret=YOK\n"
    "---\n"
    "Sahada 3 kişiden 1'i, Kişi #2, baret takmadan çalışmaktadır. "
    "Baş yaralanması riski ciddi iş kazalarına yol açabilir. "
    "Kişi #2 derhal uyarılmalı ve baret takılması sağlanmalıdır.\n\n"
    "Şimdi sadece analizi yaz, başka hiçbir şey ekleme."
)


def option2(sc: dict) -> str:
    inp = _scenario_to_input(sc)
    return _call(inp, system=_SYS_2)


# ── Seçenek 3 — Önem seviyesine göre farklı ton ──────────────────────────────

def _severity(sc: dict) -> str:
    if sc["scene"].get("fire_detected") or sc["scene"].get("smoke_detected"):
        return "HIGH"
    total_viols = sum(len(p["violations"]) for p in sc["persons"])
    if len(sc["persons"]) >= 2 or total_viols >= 3:
        return "MEDIUM"
    return "LOW"


_SYS_3_LOW = (
    "Sen bir fabrika iş güvenliği asistanısın.\n"
    "Düşük öncelikli bir KKD ihlali tespit edildi.\n"
    "Kısa, sakin ve bilgilendirici 1-2 cümle yaz. Panik yaratma.\n"
    "Kişi numarasını ve eksik KKD'yi belirt, sonunda basit bir hatırlatma yap.\n"
    "Sadece Türkçe, sadece verilen verileri kullan."
)

_SYS_3_MEDIUM = (
    "Sen bir fabrika iş güvenliği uzmanısın.\n"
    "Orta öncelikli bir KKD ihlali tespit edildi — birden fazla kişi veya ciddi eksik.\n"
    "2-3 cümle yaz: önce durumu açıkla, sonra riski vurgula, sonra net bir eylem öner.\n"
    "Resmi ve kararlı bir ton kullan. Sadece Türkçe, sadece verilen verileri kullan."
)

_SYS_3_HIGH = (
    "Sen bir fabrika acil durum AI sistemisin.\n"
    "KRİTİK durum tespit edildi — yangın veya çoklu ağır ihlal.\n"
    "2 cümle yaz: ilk cümle tehlikeyi net ifade et (büyük harf kullanabilirsin), "
    "ikinci cümle derhal yapılması gereken acil eylemi söyle.\n"
    "Kısa, güçlü ve aciliyet hissettiren bir dil kullan. Sadece Türkçe."
)


def option3(sc: dict) -> str:
    sev = _severity(sc)
    sys_map = {"LOW": _SYS_3_LOW, "MEDIUM": _SYS_3_MEDIUM, "HIGH": _SYS_3_HIGH}
    inp = _scenario_to_input(sc)
    result = _call(inp, system=sys_map[sev])
    return f"[Seviye: {sev}]\n{result}"


# ── Ana test akışı ───────────────────────────────────────────────────────────

OPTIONS = [
    ("Seçenek 2 — 2-3 Cümle Serbest Analiz",  option2),
    ("Seçenek 3 — Önem Seviyesine Göre Ton",   option3),
]

SEP = "─" * 60


def main():
    print(f"\nModel: {MODEL}")
    print(f"URL:   {BASE_URL}\n")

    # Bağlantı kontrolü
    try:
        requests.get(f"{BASE_URL}/api/tags", timeout=5)
    except Exception:
        print("HATA: Ollama'ya bağlanılamadı. 'ollama serve' çalışıyor mu?")
        sys.exit(1)

    for sc in SCENARIOS:
        print(f"\n{'═' * 60}")
        print(f"SENARYO: {sc['label']}")
        print(f"Girdi:\n  {_scenario_to_input(sc).replace(chr(10), chr(10) + '  ')}")
        print(SEP)

        for opt_label, opt_fn in OPTIONS:
            print(f"\n  ▶ {opt_label}")
            t0 = time.time()
            result = opt_fn(sc)
            elapsed = time.time() - t0
            for line in result.strip().splitlines():
                print(f"    {line}")
            print(f"    ({elapsed:.1f}s)")

    print(f"\n{'═' * 60}")
    print("Test tamamlandı.")


if __name__ == "__main__":
    main()
