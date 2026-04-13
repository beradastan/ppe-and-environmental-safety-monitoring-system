# CNN → LLM Veri Akışı Dokümantasyonu

Bu klasörde 3 kapsamlı rehber bulunmaktadır. Seçim yapın:

---

## 📚 Dosyalar

### 1. **CNN_TO_LLM_DATA_FORMAT.md** ← BAŞLA BURADAN
📖 Teorik ve derinlemesine açıklama
- CNN ajanlarının çıktı formatları (dictionary yapısı)
- Her ajanın (HelmetAgent, VestAgent, FireAgent) detaylı çıktı örnekleri
- Alarm tetikleme mantığı
- format_for_llm() fonksiyonunun çalışması
- LLM prompt yapısı
- Complete data flow

**Okuma süresi:** ~10 dakika  
**Hedef kitle:** Sistemi anlamak isteyenler, yeni başlayanlar

---

### 2. **CNN_TO_LLM_DATA_FORMAT_VISUAL.md** ← GÖRSELLİ AÇIKLAMA
📊 ASCII diagramlar ve flowchart'larla görselleştirilmiş
- Her ajan için detaylı box diagram'ı
- Input → Output dönüşüm görselleri
- Alarm kontrol flowchart'ı
- Complete data flow örneği

**Okuma süresi:** ~15 dakika  
**Hedef kitle:** Visual learner'lar, akış takip etmek isteyenler

---

### 3. **CNN_TO_LLM_QUICK_REFERENCE.md** ← HIZLI REFERANS
⚡ Tablo ve code snippet'ler
- CNN çıktı özeti tablosu
- Dictionary yapılarının hızlı kodu
- Alarm koşulları
- Field açıklamaları tablosu
- Pratik kullanım örnekleri
- Debug ipuçları
- Ayarlanabilir parametreler

**Okuma süresi:** ~5 dakika (bölüm bölüm)  
**Hedef kitle:** Kod yazacaklar, referans arayacaklar

---

## 🎯 Ne Arıyorsanız?

### "Sistemi tam olarak anlayalım"
👉 **CNN_TO_LLM_DATA_FORMAT.md** + **CNN_TO_LLM_DATA_FORMAT_VISUAL.md**

### "Hızlı bir kod örneği istiyorum"
👉 **CNN_TO_LLM_QUICK_REFERENCE.md** → "Pratik Kullanım" bölümü

### "CNN çıktılarını değiştirecek miyim"
👉 **CNN_TO_LLM_QUICK_REFERENCE.md** → "🔍 Field Açıklamaları" + "⚙️ Ayarlanabilir Parametreler"

### "Veri nereden nereye akıyor"
👉 **CNN_TO_LLM_DATA_FORMAT_VISUAL.md** → "4. ALARM KONTROLÜ MANTIK AKIŞI" + "5. COMPLETE DATA FLOW ÖRNEK"

### "Sorun giderelim"
👉 **CNN_TO_LLM_QUICK_REFERENCE.md** → "🐛 Debug İpuçları"

---

## 🔑 Key Points

### CNN Ajanları
```
HelmetAgent    → detection_count (baret giyenler) + warning_count (baretsiz) ⚠️
VestAgent      → detection_count (yelek giyen)    + warning_count (yeleksiz) ⚠️
FireAgent      → detection_count (yangın)                                    ⚠️
```

### Alarm Koşulu
```python
alarm = helmet_warning > 0 OR vest_warning > 0 OR fire_detection > 0
```

### LLM Akışı
```
CNN çıktısı → format_for_llm() → Yapılandırılmış metin
                                        ↓
                           + Sistem instruksiyonu
                                        ↓
                           Ollama/Mistral → Türkçe rapor
```

### Sonuç
```python
{
    "alarm": True/False,
    "llm_called": True/False,
    "report": "LLM raporu...",
    "structured": "CNN verisi...",
    "image": "dosya_adi.jpg",
    "timestamp": "2026-04-12T..."
}
```

---

## 🚀 Başlangıç

```bash
# Sistem hakkında bilgi
cat CNN_TO_LLM_DATA_FORMAT.md

# Görselleri gör
cat CNN_TO_LLM_DATA_FORMAT_VISUAL.md

# Hızlıca referans bak
cat CNN_TO_LLM_QUICK_REFERENCE.md

# Sistem test et
python run_with_llm.py --offline
```

---

## 📂 İlgili Kodlar

| Dosya | Amaç |
|-------|------|
| `agents/specific_agents.py` | CNN ajanları, `postprocess()` → çıktı formatı |
| `llm/llm_coordinator.py` | `format_for_llm()`, `generate_alarm_report()`, prompt |
| `run_with_llm.py` | Complete pipeline örneği |

---

## ❓ Sorular?

1. **"Neden 3 ajan?"**
   - Baret (HelmetAgent), Yelek (VestAgent), Yangın (FireAgent) — 3 ayrı tehlike

2. **"Neden detections vs warnings ayrımı?"**
   - Pozitif tespitler (güvenli) vs Negatif tespitler (uyarı/alarm)

3. **"LLM neden sadece alarm durumunda çağırılıyor?"**
   - Performans ve maliyet (Ollama API çağrısı pahalı)

4. **"Prompt'u değiştirebilir miyim?"**
   - Evet! `llm/llm_coordinator.py` → `generate_alarm_report()` → `prompt` stringi

5. **"CNN çıktısını başka formata çevirebilir miyim?"**
   - Evet! `agents/specific_agents.py` → `postprocess()` metodları, sonra `format_for_llm()` güncelle

---

## 📝 Versiyon

- **Oluşturuldu:** 2026-04-12
- **İçeriği güncelleme:** HIZLI REFERANS (3 ana dosya)
- **Sistem durumu:** llm_coordinator v1 + CNN ajanları aktif

---

**Önerilen okuma sırası:**
1. Bu dosya (index.md) — 2 dakika
2. CNN_TO_LLM_DATA_FORMAT.md — 10 dakika
3. CNN_TO_LLM_DATA_FORMAT_VISUAL.md — 15 dakika (opsiyonel)
4. CNN_TO_LLM_QUICK_REFERENCE.md — Gerektiğinde referans

