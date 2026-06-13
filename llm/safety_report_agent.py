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
            return f"[LLM error: HTTP {response.status_code}]"
        except requests.exceptions.ConnectionError:
            return "[LLM connection error: Is Ollama running? Check `ollama serve`]"
        except Exception as e:
            return f"[LLM error: {e}]"

    def _strip_thinking(self, text: str) -> str:
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

        top_location = location_breakdown[0] if location_breakdown else None
        top_violation = max(violation_counts, key=violation_counts.get) if violation_counts else None
        has_comparison = comparison.get("trend") not in (None, "no_data")

        duration = summary_json.get("duration_summary", {})
        repeat   = summary_json.get("repeat_summary", {})
        multi    = summary_json.get("multi_violation_event_count", 0)
        most_active = summary_json.get("most_active_day") or summary_json.get("most_active_week")

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
        if duration.get("average_duration_sec"):
            context_lines.append(
                f"- Ortalama olay süresi: {round(duration['average_duration_sec'])} sn "
                f"(maks: {round(duration['max_duration_sec'])} sn)"
            )
        if repeat.get("total_repeat_count"):
            context_lines.append(
                f"- Tekrar eden ihlal sayısı: {repeat['total_repeat_count']} "
                f"(olay başına ort: {repeat['average_repeat_count']})"
            )
        if multi:
            context_lines.append(f"- Çoklu ihlal içeren olay sayısı: {multi}")
        if most_active:
            label = most_active.get("date") or most_active.get("week", "")
            context_lines.append(f"- En yoğun gün/hafta: {label} ({most_active.get('event_count', '?')} olay)")

        context_block = "\n".join(context_lines)

        sections = (
            "**Dönem Özeti** — risk skoru ve seviyesi, normalize değeri, toplam olay sayısı; "
            "varsa önceki döneme kıyasla %değişim ve bu değişimin sahada ne anlama geldiği; "
            "dönemin genel güvenlik tablosunu kapsamlı biçimde değerlendir; en az 4 cümle\n"
            "**İhlal Analizi** — her ihlal türü sayı ve yüzdesiyle ayrı ayrı ele alınmalı; "
            "dönemin birincil sorunu net biçimde vurgulanmalı; ihlaller arasındaki oran farkları yorumlanmalı; "
            "yangın/duman varsa ciddiyeti ve diğer ihlallerden farkı ayrıca belirtilmeli; en az 4 cümle\n"
            "**Lokasyon Analizi** — en kritik bölge/kamera adı ve olay sayısı; "
            "yoğunlaşmanın yapısal mı rastlantısal mı olduğu gerekçesiyle açıklanmalı; "
            "diğer bölgelerle karşılaştırma yapılmalı; en az 4 cümle\n"
            "**Olay Karakteristikleri** — ortalama ve maksimum olay süresi yorumlanmalı "
            "(uzun süreli olaylar denetim gecikmesine işaret eder); "
            "tekrar eden ihlaller ve çoklu ihlal içeren olaylar değerlendirilmeli; "
            "varsa en yoğun gün/saat dilimi analiz edilmeli; en az 4 cümle\n"
        )
        if has_comparison:
            sections += (
                "**Trend Değerlendirmesi** — önceki döneme geçişin ayrıntılı analizi; "
                "artışın/azalışın hangi ihlal türünden ve hangi bölgeden kaynaklandığı; "
                "mevcut hız devam ederse önümüzdeki dönem için sayısal projeksiyon; en az 4 cümle\n"
            )
        sections += (
            "**Genel Değerlendirme** — dönemin tüm boyutlarını bir arada yorumlayan kapanış paragrafı; "
            "risk seviyesinin sürdürülebilirliği, öne çıkan örüntüler ve sistemin genel durumu ele alınmalı; "
            "öneri vermeden analitik bir sonuç yaz; en az 4 cümle"
        )

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
            "- Rapor uzun ve ayrıntılı olmalı; her bölüm kendi içinde tam ve kapsamlı biçimde yazılmalı.\n"
            "- Her iddianda veriden bir sayı veya lokasyon adı referans ver.\n"
            "- 'İhlaller artmış' gibi genel ifade kullanma; 'baret ihlali 17'den 27'ye çıkmış (+59%)' gibi spesifik ol.\n"
            "- Öneri verme; sadece analiz ve değerlendirme yap.\n"
            "- Uydurma, sadece aşağıdaki JSON'daki sayıları kullan.\n"
            "- Resmi Türkçe, paragraf başlıkları bold.\n"
            "\n"
            "Veri:\n"
            f"{json.dumps(summary_json, ensure_ascii=False, indent=2)}\n"
            "\n"
            "Rapor:"
        )
