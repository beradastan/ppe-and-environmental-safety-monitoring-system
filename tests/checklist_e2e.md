# Uçtan Uca Test Kontrol Listesi

Sistem çalışırken (backend + frontend + pipeline) manuel olarak doğrulanacak özellikler.

## Sistem Başlatma

```bash
# Terminal 1 — Backend
python -m backend.app

# Terminal 2 — Frontend
cd frontend && npm run dev

# Terminal 3 — Pipeline (crop modu)
python run_live_video.py --mode crop --camera 0 --camera-id cam_01 --zone "Test Alanı" --display

# veya video ile
python run_live_video.py --mode crop --source test/nohat_test.mp4 --camera-id cam_01 --zone "Test Alanı"
```

## Otomatik Testler

```bash
# API testi (backend çalışırken)
python tests/test_api.py

# Crop modu tespit testi (GPU, ~5 dk)
python tests/test_pipeline_crop.py --max-frames 300

# Scene modu tespit testi (GPU, ~5 dk)
python tests/test_pipeline_scene.py --max-frames 300
```

---

## Manuel Kontrol Listesi

### 1. Kontrol Paneli (Dashboard)
- [ ] Aktif Alarm sayacı görünüyor
- [ ] Bugünkü İhlal sayacı görünüyor
- [ ] İhlal Dağılımı grafiği yükleniyor
- [ ] Son Alarmlar listesi görünüyor
- [ ] "Tümünü gör →" linki Alarmlar sayfasına götürüyor
- [ ] Dark/Light tema geçişi çalışıyor

### 2. Alarmlar (AlertHistory)
- [ ] Sadece **closed** (kapalı) olaylar geliyor
- [ ] Tarih filtresi çalışıyor
- [ ] İhlal türü filtresi çalışıyor
- [ ] Temizle butonu filtreleri sıfırlıyor
- [ ] Bir olaya tıklanınca sağda timeline açılıyor
- [ ] Not eklenebiliyor

### 3. Raporlar
- [ ] Günlük / Haftalık / Aylık sekmeleri çalışıyor
- [ ] Grafik yükleniyor
- [ ] Risk skoru ve seviyesi görünüyor
- [ ] "YZ Güvenlik Raporu" butonu çalışıyor
- [ ] LLM raporu ~15-30 sn içinde geliyor (socket ile)
- [ ] CSV indirme çalışıyor
- [ ] PDF indirme çalışıyor
- [ ] Kaydedilmiş raporlar listesi görünüyor

### 4. Kamera Kurulumu
- [ ] Kamera / Video Dosyası seçimi çalışıyor
- [ ] Tespit Modu: Crop-Based / Scene-Based seçimi çalışıyor
- [ ] "Sistemi Başlat" butonu pipeline'ı başlatıyor
- [ ] Sistem Durumu paneli güncelleniyor (Sistem: Çalışıyor)
- [ ] "Sistemi Durdur" pipeline'ı durduruyor

### 5. Ayarlar
- [ ] Toggle'lar (Baret/Yelek/Maske/Yangın tespiti) çalışıyor
- [ ] Alarm Onay Süresi slider'ı çalışıyor
- [ ] Yangın filtresi slider'ları çalışıyor
- [ ] "Kaydet" butonu config.yaml'a yazıyor
- [ ] "✓ Kaydedildi" mesajı çıkıyor

### 6. Alarm Akışı (Pipeline çalışırken)
- [ ] İhlal tespit edilince `new_alert` socket eventi geliyor
- [ ] Dashboard'da aktif alarm sayısı artıyor
- [ ] Alarmlar sayfası yenileniyor (closed olduğunda)
- [ ] LLM per-alarm raporu yazılıyor (event detayında)
- [ ] Event kapanınca `event_closed` eventi geliyor

### 7. Kamera İzleme
- [ ] Kamera bağlantısı kesilince "Kamera bağlantısı kesildi!" banner'ı çıkıyor
- [ ] Donma tespiti çalışıyor (lens kapatılınca)

### 8. Genel UI
- [ ] Navbar sekmeler: Kontrol Paneli / Alarmlar / Raporlar / Kamera / Ayarlar
- [ ] Navbar: "Güvenlik Monitörü" yazıyor
- [ ] Hiçbir yerde İngilizce kullanıcıya görünen kelime yok (Crop-Based/Scene-Based/PPE hariç)

---

## Test Sonuçları

| Test | Durum | Not |
|------|-------|-----|
| API testi | | |
| Crop modu tespit | | |
| Scene modu tespit | | |
| Dashboard | | |
| Alarmlar | | |
| Raporlar | | |
| Kamera Kurulumu | | |
| Ayarlar | | |
| Alarm Akışı | | |
| Kamera İzleme | | |
