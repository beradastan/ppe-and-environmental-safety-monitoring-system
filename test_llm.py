# -*- coding: utf-8 -*-
"""LLM kalite testi — farklı system prompt stratejilerini karşılaştırır."""
import sys, time, requests
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "hf.co/QuantFactory/Turkish-Llama-8b-DPO-v0.1-GGUF:Q4_K_M"
URL   = "http://localhost:11434/api/generate"

SYSTEM_TR = (
    "Sen bir fabrika iş güvenliği yapay zekasısın. "
    "Görevin: sana verilen olay verisini okuyarak 1-2 cümlelik sade Türkçe açıklama yazmak.\n"
    "Kurallar:\n"
    "- Yalnızca verilen verideki bilgileri kullan, hiçbir şey uydurma.\n"
    "- Belirsiz durumları ihlal olarak yazma.\n"
    "- Süre bilgisi yazma.\n"
    "- Öneri veya tavsiye verme.\n"
    "- Madde işareti, başlık veya İngilizce kelime kullanma.\n"
    "- Sadece tespit edilen ihlali ve kişileri belirt.\n"
    "- En fazla 2 cümle yaz."
)

SYSTEM_EN = (
    "You are a factory safety AI. Write ONE Turkish sentence describing only the detected violation. "
    "Do not mention seconds or duration. Do not add recommendations. Do not mention unknown fields. "
    "Use only the data given.\n\n"
    "Examples:\n"
    "Input: Olay tipi: KKD ihlali\nKişi #1: baret=YOK, yelek=var\nOutput: Kişi #1 baret takmıyor.\n\n"
    "Input: Olay tipi: KKD ihlali\nKişi #2: baret=YOK, yelek=YOK\nKişi #3: baret=var, yelek=YOK\nOutput: Kişi #2 baret ve yelek takmıyor; kişi #3 yelek takmıyor.\n\n"
    "Input: Olay tipi: Yangın/duman\nSahne: yangın tespit edildi (güven: 91%)\nOutput: Sahada yangın tespit edildi.\n\n"
    "Input: Olay tipi: Çoklu tehlike\nSahne: duman tespit edildi (güven: 80%)\nKişi #5: baret=YOK, yelek=var\nOutput: Sahada duman tespit edildi; kişi #5 baret takmıyor.\n\n"
    "Now write only the Output line for the given input. Nothing else."
)

test_cases = [
    ("S1 — Tek kişi baret ihlali",
     "Olay tipi: KKD ihlali\nKişi durumları:\n  Kişi #2: baret=YOK, yelek=var, ihlal süresi=5s\n\nAçıklama:"),
    ("S2 — Çoklu PPE ihlali",
     "Olay tipi: KKD ihlali\nKişi durumları:\n  Kişi #1: baret=YOK, yelek=YOK, ihlal süresi=8s\n  Kişi #3: baret=var, yelek=YOK, ihlal süresi=3s\n\nAçıklama:"),
    ("S3 — Yangın tespiti",
     "Olay tipi: Yangın/duman\nSahne: yangın tespit edildi (güven: 89%)\nKişi durumları:\n  (kişi tespiti yok)\n\nAçıklama:"),
    ("S4 — Çoklu tehlike",
     "Olay tipi: Çoklu tehlike (yangın + KKD ihlali)\nSahne: duman tespit edildi (güven: 76%)\nKişi durumları:\n  Kişi #4: baret=YOK, yelek=var, ihlal süresi=6s\n\nAçıklama:"),
    ("S5 — Unknown hallucination testi",
     "Olay tipi: KKD ihlali\nKişi durumları:\n  Kişi #7: yelek=YOK, ihlal süresi=4s\n\nAçıklama:"),
]

for sys_label, system in [("TR system prompt", SYSTEM_TR), ("EN system prompt", SYSTEM_EN)]:
    print(f"\n{'#'*60}")
    print(f"  {sys_label}")
    print(f"{'#'*60}")
    for label, prompt in test_cases:
        start = time.time()
        resp = requests.post(URL, json={
            "model": MODEL, "system": system, "prompt": prompt,
            "temperature": 0.1, "stream": False,
        }, timeout=60)
        elapsed = time.time() - start
        raw = resp.json().get("response", "").strip()
        print(f"\n{'='*55}")
        print(f"{label}  [{elapsed:.1f}s]")
        print("-"*55)
        print(raw or "(boş)")
