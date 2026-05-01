import json
import re
import requests


class SafetyReportAgent:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        timeout: int = 120,
        facility_name: str = "Fabrika",
    ):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.timeout = timeout
        self.facility_name = facility_name

    def generate_report(self, summary_json: dict) -> str:
        prompt = self._build_prompt(summary_json)
        raw = self._call_ollama(prompt)
        return self._strip_thinking(raw)

    def _call_ollama(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.3,
                "stream": False,
            }
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"[LLM hata: HTTP {response.status_code}]"
        except requests.exceptions.ConnectionError:
            return "[LLM bağlantı hatası: Ollama çalışıyor mu? `ollama serve` kontrol et]"
        except Exception as e:
            return f"[LLM hatası: {e}]"

    def _strip_thinking(self, text: str) -> str:
        # Qwen3 thinking bloklarını temizle (<think>...</think>)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _build_prompt(self, summary_json: dict) -> str:
        report_type = summary_json.get("report_type", "daily")
        type_label = {"daily": "Günlük", "weekly": "Haftalık", "monthly": "Aylık"}.get(
            report_type, report_type.capitalize()
        )

        risk = summary_json.get("risk_summary", {})
        comparison = summary_json.get("comparison", {})
        violation_counts = summary_json.get("violation_counts", {})
        location_breakdown = summary_json.get("location_breakdown", [])

        # En kritik lokasyonu ve ihlali önceden belirle — modelin dikkatini çek
        top_location = location_breakdown[0] if location_breakdown else None
        top_violation = max(violation_counts, key=violation_counts.get) if violation_counts else None
        has_comparison = comparison.get("trend") not in (None, "no_data")

        context_lines = []
        if top_violation:
            context_lines.append(f"- En yüksek ihlal türü: {top_violation} ({violation_counts[top_violation]} olay)")
        if top_location:
            context_lines.append(f"- En kritik lokasyon: {top_location['zone']} / {top_location['camera_id']} ({top_location['event_count']} olay)")
        if has_comparison:
            context_lines.append(
                f"- Önceki döneme göre trend: {comparison['trend']} "
                f"({comparison.get('change_percent', '?')}%, önceki dönem: {comparison.get('previous_period_events', '?')} olay)"
            )
        context_lines.append(
            f"- Risk: {risk.get('risk_level')} (normalize skor: {risk.get('normalized_score')}/100)"
        )

        context_block = "\n".join(context_lines)

        sections = (
            "**Genel Değerlendirme** — dönem özeti, risk skoru ve normalize değeri, toplam olay sayısı\n"
            "**Kritik Bulgular** — en yüksek ihlal türü ve sayısı; en kritik bölge/kamera ve olay yoğunluğu"
        )
        if has_comparison:
            sections += "\n**Trend Analizi** — önceki döneme göre değişim yüzdesi, artış/azalış nedenleri"
        sections += "\n**Eylem Önerileri** — yalnızca veriden çıkan bulgulara dayalı, kamera/bölge adı belirterek, en fazla 3 madde"

        return (
            "/think\n"
            f"Sen {self.facility_name} iş güvenliği izleme sisteminin analiz asistanısın.\n"
            "İhlal türü karşılıkları: helmet_violation=baret ihlali, vest_violation=güvenlik yeleği ihlali, mask_violation=maske ihlali, fire_detected=yangın tespiti.\n"
            f"Aşağıdaki {type_label} güvenlik istatistiklerine dayanarak resmi Türkçe rapor yaz.\n"
            "\n"
            "Ön bilgi (veriden çıkarıldı):\n"
            f"{context_block}\n"
            "\n"
            "Rapor bölümleri:\n"
            f"{sections}\n"
            "\n"
            "Kurallar:\n"
            "- Her iddianda veriden bir sayı veya lokasyon adı referans ver.\n"
            "- 'İhlaller artmış' gibi genel ifade kullanma; 'baret ihlali 17'den 27'ye çıkmış (+59%)' gibi spesifik ol.\n"
            "- Öneri bölümünde kamera/bölge adını mutlaka belirt.\n"
            "- Uydurma, sadece aşağıdaki JSON'daki sayıları kullan.\n"
            "- Resmi Türkçe, paragraf başlıkları bold.\n"
            "\n"
            "Veri:\n"
            f"{json.dumps(summary_json, ensure_ascii=False, indent=2)}\n"
            "\n"
            "Rapor:"
        )
