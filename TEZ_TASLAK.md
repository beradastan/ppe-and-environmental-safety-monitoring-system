# FABRİKA İŞ GÜVENLİĞİ İZLEME SİSTEMİ
## (Mezuniyet Projesi)

---

## ÖZET

Bu çalışmada, fabrika ortamlarında kişisel koruyucu ekipman (KKD) uyumunun ve yangın/duman tehlikelerinin gerçek zamanlı olarak izlenmesini sağlayan bütünleşik bir iş güvenliği sistemi olan SafetyMonitor geliştirilmiştir. Sistem, YOLOv8 mimarisi tabanlı sekiz ayrı uzman modelden oluşan çok ajanlı bir yapı üzerine inşa edilmiştir; kişi tespiti, baret/yelek/maske sınıflandırması ve yangın/duman algılama görevleri birbirinden bağımsız modellerle ele alınmaktadır. ByteTrack çok nesne takip algoritması ile geliştirilen TrackReattacher bileşeni, kısa süreli oklüzyonlarda oluşan takip kimliği değişikliklerini çok sinyalli skor mekanizmasıyla birleştirerek kararlı kişi kimliği üretmektedir.

Literatürdeki mevcut çalışmalardan farklı olarak sistem, PPE tespiti için iki ayrı mimari mod sunmaktadır: kişi anatomik bölgesi kırpımları üzerinde çalışan crop-based mod ve tam kare üzerinde geometrik atama yapan scene-based mod. Gerçekleştirilen model karşılaştırmalarında crop-based modun ihlal sınıfı tespitinde önemli kazanımlar sağladığı görülmüştür; baret ihlali (NO-Hardhat) recall değeri scene modda 0,610 iken crop modda 0,884'e yükselmiştir. İşlem hızı bakımından scene-based mod kişi sayısından bağımsız tutarlı bir performans sunmakta (ortalama 46,4 FPS), crop-based mod ise yüksek kişi yoğunluğunda FPS düşüşü yaşamaktadır (26,6–54,5 FPS arası).

Tespit pipeline'ı, durum makinesi tabanlı olay yöneticisi, PostgreSQL veritabanı, Flask/Socket.IO tabanlı REST API ve React+Vite web arayüzüyle bütünleşik bir sistem oluşturmaktadır. Yerel Ollama çerçevesiyle çalıştırılan qwen3:8b dil modeli, her ihlal olayı için bireysel Türkçe özetler ve seçilen dönem için kapsamlı güvenlik raporları otomatik olarak üretmektedir. Sistem, dört farklı test videosu üzerinde gerçekleştirilen değerlendirmede ground truth ihlal kararlarının %75–83 oranında doğru sınıflandırıldığını göstermiştir.

**Anahtar Kelimeler:** İş güvenliği izleme, kişisel koruyucu ekipman tespiti, YOLOv8, ByteTrack, çok nesne takibi, nesne tespiti, büyük dil modeli, gerçek zamanlı video analizi

---

## ABSTRACT

In this study, SafetyMonitor, an integrated occupational safety monitoring system for real-time detection of personal protective equipment (PPE) compliance and fire/smoke hazards in factory environments, has been developed. The system is built upon a multi-agent architecture consisting of eight specialized YOLOv8 models; the tasks of person detection, helmet/vest/mask classification, and fire/smoke detection are handled by independent models. The TrackReattacher component, developed alongside the ByteTrack multi-object tracking algorithm, consolidates temporary tracking identity changes caused by brief occlusions through a multi-signal weighted scoring mechanism, thereby producing stable person identities.

In contrast to existing approaches in the literature, the system provides two distinct architectural modes for PPE detection: a crop-based mode that operates on anatomical region crops of each tracked person, and a scene-based mode that performs geometric assignment over the full frame. Comparative evaluations reveal that the crop-based mode achieves significant improvements in violation class detection; the NO-Hardhat recall increases from 0.610 in scene mode to 0.884 in crop mode. In terms of processing speed, the scene-based mode delivers consistent throughput regardless of person count (average 46.4 FPS), while the crop-based mode experiences FPS degradation under high person density (ranging from 26.6 to 54.5 FPS).

The detection pipeline is integrated with an event state machine, a PostgreSQL database, a Flask/Socket.IO REST API, and a React+Vite web interface to form a cohesive system. The qwen3:8b language model, operated locally via the Ollama framework, automatically generates individual Turkish summaries for each violation event and comprehensive safety reports for selected time periods. Evaluation across four test videos demonstrates that ground-truth violation decisions are correctly classified at a rate of 75–83%.

**Keywords:** Occupational safety monitoring, personal protective equipment detection, YOLOv8, ByteTrack, multi-object tracking, object detection, large language model, real-time video analysis

---

## SİMGELER VE KISALTMALAR

| Kısaltma | Açıklama |
|----------|----------|
| API | Application Programming Interface — Uygulama Programlama Arayüzü |
| ByteTrack | Zhang ve ark. (2022) çok nesne takip algoritması |
| CNN | Convolutional Neural Network — Evrişimli Sinir Ağı |
| CORS | Cross-Origin Resource Sharing |
| CPU | Central Processing Unit — Merkezi İşlem Birimi |
| CSV | Comma-Separated Values — Virgülle Ayrılmış Değerler |
| CUDA | Compute Unified Device Architecture (NVIDIA GPU paralel hesaplama platformu) |
| DB | Database — Veritabanı |
| DeepSORT | Deep Simple Online and Realtime Tracking (çok nesne takip algoritması) |
| FPS | Frames Per Second — Saniyedeki Kare Sayısı |
| GPU | Graphics Processing Unit — Grafik İşlem Birimi |
| HTTP | Hypertext Transfer Protocol |
| ILO | International Labour Organization — Uluslararası Çalışma Örgütü |
| IoU | Intersection over Union — Kesişim/Birleşim Oranı |
| JSONB | JSON Binary (PostgreSQL ikili JSON veri tipi) |
| KKD | Kişisel Koruyucu Ekipman |
| LLM | Large Language Model — Büyük Dil Modeli |
| mAP | Mean Average Precision — Ortalama Hassasiyet |
| OSHA | Occupational Safety and Health Administration |
| PDF | Portable Document Format |
| PPE | Personal Protective Equipment — Kişisel Koruyucu Ekipman |
| R-CNN | Region-based Convolutional Neural Network |
| REST | Representational State Transfer |
| SGK | Sosyal Güvenlik Kurumu |
| SORT | Simple Online and Realtime Tracking (çok nesne takip algoritması) |
| SPA | Single Page Application — Tek Sayfalık Uygulama |
| SQL | Structured Query Language — Yapılandırılmış Sorgulama Dili |
| SSD | Single Shot MultiBox Detector (nesne tespit mimarisi) |
| USB | Universal Serial Bus |
| VRAM | Video Random Access Memory — Video Rastgele Erişim Belleği |
| YOLO | You Only Look Once (gerçek zamanlı nesne tespit mimarisi) |
| YOLOv8 | You Only Look Once sürüm 8 (Ultralytics, 2023) |

---

# 1. GİRİŞ

Sanayileşmenin hızla ilerlediği günümüzde iş güvenliği, üretim sektörlerinin en kritik sorunlarından biri olmaya devam etmektedir. Uluslararası Çalışma Örgütü (ILO) verilerine göre dünya genelinde her yıl yaklaşık 2,3 milyon çalışan iş kazası veya meslek hastalığı nedeniyle hayatını kaybetmekte, 374 milyonun üzerinde çalışan ise iş kazası geçirmektedir [1]. Türkiye'de Sosyal Güvenlik Kurumu (SGK) istatistikleri incelendiğinde, imalat sektörünün iş kazası sayısı bakımından en yüksek paya sahip sektörler arasında yer aldığı görülmektedir [2]. Bu kazaların önemli bir bölümünün temelinde kişisel koruyucu ekipman (KKD) kullanımına uyumsuzluk yatmaktadır. Baret, güvenlik yeleği ve yüz maskesi gibi temel KKD'lerin düzenli ve doğru biçimde kullanılması; baş yaralanmaları, ezilme ve solunum yolu hasarları gibi ciddi iş kazası türlerini büyük ölçüde önleyebilmektedir [3].

Geleneksel iş güvenliği denetimi büyük ölçüde periyodik fiziksel kontrollere ve güvenlik görevlilerinin anlık gözlemlerine dayanmaktadır. Bu yaklaşım; iş gücü maliyeti yüksekliği, insan kaynaklı dikkat dağınıklığı ve geniş çalışma alanlarının anlık izlenememesi gibi temel sınırlılıklar barındırmaktadır. Öte yandan mevcut kamera tabanlı sistemlerin çoğu yalnızca kayıt işlevi görmekte; ihlalleri gerçek zamanlı olarak tespit edip bildirme kapasitesinden yoksun kalmaktadır. Yapay zeka tabanlı görüntü işleme teknolojilerindeki son gelişmeler, bu alandaki boşluğu doldurmak için güçlü bir fırsat sunmaktadır.

**_Nesne Tespiti ve Derin Öğrenme_**

Derin öğrenme tabanlı nesne tespiti, özellikle evrişimli sinir ağlarının (CNN) gelişimiyle birlikte hızla olgunlaşmıştır. YOLO (You Only Look Once) mimarisinin 2016 yılında Redmon ve arkadaşları tarafından tanıtılmasıyla birlikte gerçek zamanlı nesne tespitinde yeni bir dönem başlamıştır [4]. Sonraki yıllarda YOLOv3, YOLOv5 ve YOLOv8 gibi sürümlerle sürekli geliştirilen bu mimari; hız ve doğruluk dengesi açısından sektörde geniş kabul görmüştür. Ultralytics tarafından geliştirilen YOLOv8 [5], esnek model boyutları (nano'dan x-large'a), yerleşik veri artırma desteği ve çeşitli görev türleri (tespit, segmentasyon, sınıflandırma) ile endüstriyel uygulamalar için güçlü bir temel sunmaktadır. KKD tespiti alanında YOLOv8 tabanlı çalışmalar; baret, yelek ve maske tespitinde yüksek doğruluk oranları elde edilebileceğini ortaya koymuştur [6, 7].

**_Çok Nesne Takibi_**

Statik nesne tespitinin ötesinde, fabrika ortamlarında kişilerin zaman içindeki hareketlerinin tutarlı biçimde izlenmesi büyük önem taşımaktadır. Çok nesne takibi (Multi-Object Tracking, MOT) alanında SORT [8] ve DeepSORT [9] gibi algoritmalar yaygın kullanım bulmuştur. Ancak bu yöntemler, düşük güven skorlu tespitlerde ve kısa süreli oklüzyonlarda kimlik kaybı yaşayabilmektedir. Zhang ve arkadaşlarının 2022 yılında önerdiği ByteTrack algoritması [10]; tespit güven eşiğinin altında kalan adayları da izleme sürecine dahil ederek kayıp oranını önemli ölçüde azaltmış ve birçok MOT kıyaslamasında üst sıralarda yer almıştır. Bununla birlikte, herhangi bir takip algoritması kısa süreli oklüzyonlar sonrasında kişi kimliklerini yeniden atayabilmektedir. Bu sorun, iş güvenliği izleme bağlamında kritik bir anlam taşımaktadır; zira kimlik değişimi, bir kişinin ihlal geçmişinin bölünmesine ve yanlış alarm üretilmesine yol açmaktadır.

**_Büyük Dil Modelleri ve Raporlama_**

Son yıllarda büyük dil modellerinin (Large Language Model, LLM) mühendislik sistemlerine entegrasyonu hız kazanmıştır. GPT-4 [11] ve Llama [12] gibi modellerin iş güvenliği alanındaki uygulamaları; olay raporlarının otomatik özetlenmesi, risk analizi ve karar destek sistemleri üzerine yoğunlaşmaktadır. Ancak bu çalışmaların büyük çoğunluğu bulut tabanlı API'lere dayandığından gizlilik, gecikme ve maliyet sorunları gündeme gelmektedir. Ollama gibi yerel çalıştırma çerçevelerinin gelişmesiyle birlikte, kompakt dil modellerinin doğrudan uç donanımlarda çalıştırılması mümkün hâle gelmiştir [13].

**_Bu Çalışmanın Kapsamı ve Katkıları_**

Bu çalışmada, gerçek zamanlı KKD uyum denetimi ve yangın tespiti için bütünleşik bir fabrika güvenliği izleme sistemi olan SafetyMonitor tasarlanmış ve geliştirilmiştir. Sistem; YOLOv8 tabanlı nesne tespiti, ByteTrack çok nesne takibi, özgün bir kimlik yeniden bağlama bileşeni (TrackReattacher), olay durumu makinesi, PostgreSQL tabanlı veri yönetimi, yerel LLM ile otomatik rapor üretimi ve React tabanlı gerçek zamanlı web arayüzünden oluşmaktadır.

Literatürdeki mevcut çalışmalardan farklı olarak bu sistem iki ayrı PPE tespit mimarisi sunmaktadır: kırpık görüntü üzerinde çalışan *crop-based mod* ve tam kare üzerinde PPE-kişi eşleştirmesi yapan *scene-based mod*. Kullanıcı, ortam özelliklerine ve donanım kapasitesine göre bu modlar arasında geçiş yapabilmektedir. Bunun yanı sıra sistem, her ihlal olayını bağımsız bir durum makinesiyle yönetmekte; tekrar sayısı ve süre gibi niceliksel metrikleri hesaplamakta ve bu veriler üzerinden yerel LLM aracılığıyla günlük, haftalık ve aylık Türkçe güvenlik raporları otomatik olarak üretilmektedir.

Çalışmanın özgün katkıları şu şekilde özetlenebilir:

- Aynı pipeline içinde seçilebilir iki PPE tespit modu (crop-based ve scene-based) tasarımı ve karşılaştırmalı değerlendirmesi
- Çok sinyalli ağırlıklı benzerlik fonksiyonu kullanan TrackReattacher bileşeninin geliştirilmesi
- Temporal voting ve olay durum makinesi ile gürültü bastırma ve tutarlı alarm yönetimi
- Yerel LLM entegrasyonu ile gizlilik odaklı, gerçek zamanlı Türkçe güvenlik raporu üretimi
- Tüm bileşenleri tek çatı altında birleştiren açık kaynaklı, genişletilebilir sistem mimarisi

Tezin geri kalanı şu şekilde düzenlenmiştir: İkinci bölüm, sistemin tasarım ve geliştirme sürecine ilişkin yöntemi kapsamlı biçimde ele almaktadır. Üçüncü bölümde deneysel bulgular sunulmakta ve iki tespit modunun karşılaştırmalı değerlendirmesi yapılmaktadır. Dördüncü ve son bölümde ise elde edilen sonuçlar özetlenmekte ve gelecekteki çalışmalara yönelik öneriler sunulmaktadır.

---

---

# 2. YÖNTEM

## 2.1 Araştırma Deseni ve Sistem Mimarisi

### 2.1.1 Genel Sistem Tasarımı ve Bileşenler

SafetyMonitor, birden fazla yazılım katmanının koordineli çalışmasına dayanan çok bileşenli bir sistem olarak tasarlanmıştır. Sistem mimari açıdan dört ana katmandan oluşmaktadır: tespit katmanı, uygulama katmanı, veri katmanı ve sunum katmanı.

**Tespit katmanı**, `run_live_video.py` adlı birleşik pipeline betiği tarafından yürütülmektedir. Bu bileşen; kamera veya video kaynağından ham görüntü çerçevelerini okumakta, YOLOv8 modelleri aracılığıyla kişi, KKD ve yangın tespiti gerçekleştirmekte, ByteTrack ile çok nesne takibi yapmakta ve olay durumu makinesini çalıştırarak ihlalleri kayıt altına almaktadır. Pipeline, bağımsız bir süreç (subprocess) olarak başlatılmakta; backend API'si ile HTTP üzerinden iletişim kurmaktadır.

**Uygulama katmanı**, Python tabanlı Flask web çerçevesi ve Flask-SocketIO kütüphanesi üzerine inşa edilmiş bir REST API sunucusundan oluşmaktadır. Backend; olay yönetimi, raporlama, konfigürasyon ve pipeline kontrolü için HTTP endpoint'leri sağlamakta; aynı zamanda Socket.IO protokolü aracılığıyla gerçek zamanlı istemci bildirimleri göndermektedir. Sunucu 5050 numaralı port üzerinde çalışmaktadır.

**Veri katmanı**, PostgreSQL ilişkisel veritabanı sistemi ile dosya sistemi tabanlı yedek mekanizmasından oluşmaktadır. Veritabanı erişilir durumda olduğunda tüm olay verileri PostgreSQL'e yazılmaktadır. Veritabanı bağlantısı kesildiğinde sistem, `results/` dizininde JSON dosyaları oluşturarak çalışmayı sürdürmekte; bağlantı yeniden kurulduğunda ise Watchdog tabanlı dosya izleyici bu verileri otomatik olarak veritabanına aktarmaktadır.

**Sunum katmanı**, React ve Vite kullanılarak geliştirilmiş tek sayfalık bir web uygulamasından oluşmaktadır. Uygulama; anlık istatistikler, alarm geçmişi, dönemsel raporlar, kamera kurulum arayüzü ve sistem ayarları olmak üzere beş ana sayfa sunmaktadır. Geliştirme ortamında 5173 numaralı port üzerinde çalışan frontend, üretim ortamında Flask tarafından statik dosya olarak sunulmaktadır.

Bu dört katmana ek olarak sistem, yerel olarak çalışan Ollama LLM sunucusunu raporlama amacıyla kullanmaktadır. LLM bileşeni, dönemsel güvenlik raporlarının otomatik üretilmesinden sorumludur ve sisteme gevşek bağlı (loosely coupled) bir servis olarak entegre edilmiştir.

### 2.1.2 Kullanılan Teknolojiler ve Araçlar

Sistemin geliştirilmesinde seçilen teknolojiler; performans, ekosistem olgunluğu ve açık kaynak erişilebilirliği kriterleri gözetilerek belirlenmiştir.

**YOLOv8 (Ultralytics):** Nesne tespiti için Ultralytics'in YOLOv8 mimarisi [5] kullanılmıştır. YOLOv8, anchor-free tasarımı ve CSPNet tabanlı omurgasıyla önceki YOLO sürümlerine kıyasla daha yüksek mAP değerleri elde etmektedir. Sistemde toplam altı ayrı YOLOv8 modeli kullanılmaktadır: kişi tespiti, yangın/duman tespiti ve crop ile scene modları için ayrı ayrı baret, yelek, maske tespit modelleri. Model boyutu seçiminde nano (YOLOv8n) ve medium (YOLOv8m) mimarileri değerlendirilmiş; tespit görevi ve veri seti büyüklüğüne göre farklı modeller tercih edilmiştir.

**ByteTrack:** Çok nesne takibi için ByteTrack algoritması [10] entegre edilmiştir. ByteTrack, yüksek güven skorlu tespitlerin yanı sıra düşük güven skorlu adayları da Kalman filtresi tahminleriyle eşleştirerek kimlik kaybını minimize etmektedir. Algoritmanın davranışı, `bytetrack.yaml` yapılandırma dosyası aracılığıyla izleme tampon uzunluğu (`track_buffer`), eşleştirme eşiği (`match_thresh`) ve minimum kutu alanı (`min_box_area`) gibi parametreler ile özelleştirilmektedir.

**Flask ve Flask-SocketIO:** Backend API, Python'ın Flask mikro web çerçevesi [14] üzerine inşa edilmiştir. Gerçek zamanlı olay bildirimleri için Flask-SocketIO [15] kütüphanesi kullanılmış; bu sayede sunucu, istemcilere WebSocket bağlantısı üzerinden anlık mesaj gönderebilmektedir.

**PostgreSQL:** Kalıcı veri depolama için PostgreSQL ilişkisel veritabanı yönetim sistemi tercih edilmiştir. Python tarafında psycopg2 kütüphanesi bağlantı adaptörü olarak kullanılmaktadır. Olayların imza ve kişi verileri gibi yarı yapılandırılmış alanlar için PostgreSQL'in JSONB veri türünden yararlanılmıştır.

**React ve Vite:** Kullanıcı arayüzü, bileşen tabanlı React kütüphanesi [16] kullanılarak geliştirilmiştir. Geliştirme ve derleme araçları olarak Vite [17] tercih edilmiş; grafik görselleştirmeleri için Recharts [18] kütüphanesi kullanılmıştır.

**Ollama:** Yerel LLM çalıştırma altyapısı olarak Ollama [13] kullanılmıştır. qwen2.5:7b modeli [19]; Türkçe dil desteği, makul çıkarım süresi ve 6 GB VRAM sınırı içinde çalışabilmesi nedeniyle tercih edilmiştir.

**ReportLab:** PDF raporu üretimi için Python'ın ReportLab kütüphanesi [20] kullanılmıştır. CSV dışa aktarımı ise Python'ın standart `csv` modülü ile gerçekleştirilmekte; Excel uyumluluğu için UTF-8 BOM kodlaması uygulanmaktadır.

**Donanım:** Sistemin geliştirilmesi ve test edilmesi NVIDIA RTX 3060 6 GB GPU'ya sahip bir iş istasyonunda gerçekleştirilmiştir. CUDA destekli çıkarım, CPU'ya kıyasla belirgin biçimde daha yüksek kare hızı sağlamaktadır. Sistem; CUDA mevcut olmadığında otomatik olarak CPU moduna geçecek şekilde yapılandırılmıştır.

### 2.1.3 Konfigürasyon Yönetimi

Sistemin tüm çalışma zamanı parametreleri, kök dizinde yer alan `config.yaml` dosyasında merkezi olarak yönetilmektedir. Bu yaklaşım; kod değişikliğine gerek kalmaksızın ortama özgü ayarların yapılabilmesini ve farklı donanım konfigürasyonlarına hızlıca uyum sağlanabilmesini mümkün kılmaktadır.

Yapılandırma dosyası yedi ana bölümden oluşmaktadır:

`database` bölümü; PostgreSQL bağlantı parametrelerini (host, port, veritabanı adı, kullanıcı bilgileri) ve veritabanının etkin olup olmadığını belirten `enabled` bayrağını içermektedir.

`backend` bölümü; Flask sunucusunun dinlediği host ve port bilgisini ile CORS izin verilen kökenlerin listesini barındırmaktadır.

`detection` bölümü; kişi tespiti için minimum güven skoru (`person_confidence: 0.4`), KKD tespiti için güven skoru (`ppe_confidence: 0.35`), yangın tespiti için güven skoru (`fire_confidence: 0.5`) ve bir takibin kararlı kabul edilmesi için gereken minimum ardışık tespit sayısını (`min_hits: 3`) tanımlamaktadır.

`ppe_pipeline` bölümü; her KKD türü için ayrı güven eşikleri, temporal voting pencere boyutu (`temporal_window: 30`), kamera izleme parametreleri (donma ve karanlık tespiti için eşik değerleri ve çerçeve sayıları), yangın filtre parametreleri (`fire_min_area_ratio`, `fire_growth_factor`, `fire_growth_window`) ve olay durum geçiş sürelerini içermektedir.

`models` bölümü; cihaz seçimi (`device: cuda`), tüm modlarda ortak kullanılan kişi ve yangın modeli yolları ile `crop` ve `scene` alt bölümleri altında moda özgü PPE modeli yollarını tanımlamaktadır.

`event_manager` bölümü; bir ihlal için yeni olay açılması (`new_confirm_sec: 3.0`) ve açık bir olayın kapatılması (`resolved_confirm_sec: 5.0`) için gereken süre eşiklerini, yangın onaylama ve temizleme çerçeve sayılarını içermektedir.

`llm` bölümü; Ollama sunucu adresi, kullanılacak model adı, sıcaklık parametresi (`temperature: 0.3`) ve maksimum bekleme süresini (`timeout: 120`) barındırmaktadır.

### 2.1.4 Bileşenler Arası İletişim ve Veri Akışı

Sistem bileşenleri arasındaki iletişim iki temel protokol üzerine kuruludur: senkron REST API çağrıları ve asenkron Socket.IO mesajlaşması.

**Tespit Pipeline'ından Backend'e İletişim:** Tespit pipeline'ı, bir olay oluştuğunda veya güncellendiğinde doğrudan HTTP istekleri göndermektedir. Yeni bir ihlal olayı tespit edildiğinde `POST /api/events` isteği gönderilmekte; mevcut bir olay güncellendiğinde `PATCH /api/events/<id>/close` veya `PATCH /api/events/<id>/llm` endpoint'leri çağrılmaktadır. Kamera durum değişikliklerinde ise `POST /api/pipeline/camera-status` kullanılmaktadır. Bu iletişim modeli; pipeline ve backend'in bağımsız süreçler olarak çalışmasına ve birbirinden izole bir şekilde yeniden başlatılabilmesine olanak tanımaktadır.

**Backend'den Frontend'e İletişim:** Gerçek zamanlı bildirimler Socket.IO protokolü aracılığıyla iletilmektedir. Yeni bir alarm oluştuğunda `new_alert` eventi; bir olay kapandığında `event_closed` eventi; kamera durumu değiştiğinde `camera_status` eventi; LLM raporu hazır olduğunda `report_llm_ready` veya `report_llm_error` eventi tüm bağlı istemcilere yayınlanmaktadır. Bu yaklaşım, HTTP polling'e gerek kalmaksızın anlık güncelleme sağlamaktadır.

**Frontend'den Backend'e İletişim:** Kullanıcı arayüzü, tüm veri okuma ve yazma işlemleri için REST API endpoint'lerini kullanmaktadır. `api.js` modülü; olay listeleme, timeline görüntüleme, rapor oluşturma, pipeline başlatma/durdurma ve konfigürasyon güncelleme gibi işlemler için merkezi HTTP istemci fonksiyonlarını barındırmaktadır.

Sistemin uçtan uca veri akışı şu sırayı izlemektedir: kamera veya video kaynağından alınan ham çerçeve → YOLOv8 ile tespit → ByteTrack ile takip → TrackReattacher ile kimlik kararlılaştırma → temporal voting ile karar → olay durum makinesi → veritabanına yazma → Socket.IO ile frontend bildirimi → kullanıcı arayüzünde görüntüleme.

---

## 2.2 Veri Setleri ve Model Eğitimi

### 2.2.1 Veri Seti Seçim Kriterleri

Bu çalışmada her ajan için görev odaklı ayrı veri setleri hazırlanmıştır. Ana kaynak olarak Construction Site Safety sınıf yapısına sahip YOLO formatındaki veri seti kullanılmıştır. Bu veri seti; `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery` ve `vehicle` olmak üzere on sınıf içermektedir. Veri seti seçiminde YOLO formatı uyumluluğu, sınıf dengesi ve yapı sahnesine özgü görsel çeşitlilik temel kriterler olarak benimsenmiştir.

Çok ajanlı mimaride tek bir genel model yerine görev bazlı uzman ajanlar kullanılmıştır. Bu tercih; her PPE öğesinin farklı görsel ölçeğe, farklı konumsal bağlama ve farklı hata davranışına sahip olmasından kaynaklanmaktadır. Baret küçük ve baş bölgesine bağlı, yelek gövde bölgesini kaplayan orta/büyük bir nesne, maske ise küçük ve yüz bölgesine özgü bir ekipmandır. Yangın ve duman ise kişi üzerinde taşınmayan sahne olaylarıdır; dolayısıyla PPE ajanlarından bağımsız olarak ele alınmıştır.

### 2.2.2 Kullanılan Veri Setleri

Her ajan için ilgili sınıflar kaynak veri setinden filtrelenerek ayrı veri setleri oluşturulmuştur. Çizelge 2.1'de tüm veri setlerinin eğitim, doğrulama ve test görüntü sayıları ile hedef sınıflar özetlenmektedir.

**Çizelge 2.1.** Ajan bazlı veri seti özeti

| Veri Seti | Kullanım | Eğitim | Doğrulama | Test | Sınıflar |
|---|---|---:|---:|---:|---|
| `helmet_dataset` | Scene baret | 2605 | 114 | 82 | Hardhat, NO-Hardhat |
| `vest_dataset` | Scene yelek | 2605 | 114 | 82 | Safety Vest, NO-Safety Vest |
| `mask_dataset` | Scene maske | 2035 | 61 | 44 | Mask, NO-Mask |
| `person_dataset` | Kişi tespiti | 2506 | 84 | 59 | Person |
| `crophelmet_dataset` | Crop baret | 4057 | 112 | 113 | Hardhat, NO-Hardhat |
| `cropvest_dataset` | Crop yelek | 4879 | 111 | 118 | Safety Vest, NO-Safety Vest |
| `cropmask_dataset` | Crop maske | 4401 | 91 | 96 | Mask, NO-Mask |
| `firesmoke_dataset` | Yangın/duman | 12813 | 6068 | 2237 | fire, other, smoke |

Scene tabanlı ajanlarda tam görüntüler korunmuş; yalnızca hedef sınıf etiketleri süzülmüş ve kaynak sınıf kimlik numaraları hedef ajan kimlik numarasına yeniden eşlenmiştir. Örneğin maske veri setinde kaynak `Mask` (sınıf 1) → hedef sınıf 0, kaynak `NO-Mask` (sınıf 3) → hedef sınıf 1 olarak dönüştürülmüştür.

Crop tabanlı veri setleri ise farklı bir üretim süreciyle hazırlanmıştır. Bu süreç bir sonraki alt bölümde ayrıntılı olarak açıklanmaktadır.

### 2.2.3 Veri Ön İşleme ve Crop Veri Seti Üretimi

**Scene veri seti hazırlama:** Kaynak YOLO etiket dosyaları okunmuş, hedef ajan için gerekli sınıf kimlik numaraları tutulmuş, hedef sınıf kalmayan görüntüler veri setinden çıkarılmış ve yeni `data.yaml` dosyası hedef sınıf isimleriyle oluşturulmuştur.

**Crop veri seti üretimi:** Crop tabanlı ajanlar için doğrudan tam görüntüler kullanılmamıştır. Kaynak veri seti üzerinde bir kişi tespit modeli çalıştırılmış, her kişi bounding box'u için %30 dolgu (padding) eklenerek genişletilmiş ve görüntü sınırları içine kırpılmış bir bölge elde edilmiştir. Genişletilmiş bölge hesabında kullanılan formüller şu şekildedir:

```
padding_x = person_width  × 0.30
padding_y = person_height × 0.30
crop_x1 = max(0,            person_x1 − padding_x)
crop_y1 = max(0,            person_y1 − padding_y)
crop_x2 = min(image_width,  person_x2 + padding_x)
crop_y2 = min(image_height, person_y2 + padding_y)
```

Her PPE etiketi için etiketin kırpık alan içinde kalan kesişim oranı (label IoA) hesaplanmıştır:

```
label_ioa = area(label_bbox ∩ crop_bbox) / area(label_bbox)
```

Etiketin yeterince kırpık alan içinde kalıp kalmadığı bu oran üzerinden kontrol edilmiştir. Baret ve yelek için eşik 0.60, maske için 0.35 olarak belirlenmiştir. Maskenin eşiğinin daha düşük tutulmasının nedeni, yüz bölgesinin kişi bounding box'unun kenarlarına yakın kalabilmesi ve maske etiketlerinin küçük olmasıdır. Eşiği geçen etiketlerin koordinatları kırpık görüntü koordinat sistemine dönüştürülmüş ve yeni YOLO etiket dosyaları oluşturulmuştur.

Bu yaklaşım sayesinde scene tabanlı ve crop tabanlı yaklaşımlar aynı temel veri kaynağından türetilmiş, ancak farklı görsel temsil biçimleriyle eğitilmiştir.

**Artırma (Augmentation):** Tüm eğitimlerde Ultralytics'in yerleşik artırma pipeline'ı kullanılmıştır. Uygulanan başlıca artırma teknikleri; mozaik bileşimi (mosaic), HSV renk uzayı değişimleri (ton, doygunluk, parlaklık), yatay çevirme (%50 olasılıkla), ölçek değişimi, konum kaydırma ve rastgele bölge silmedir. Ek olarak Albumentations kütüphanesi aracılığıyla düşük olasılıklı (0.01) bulanıklaştırma, gri tonlama ve CLAHE lokal kontrast iyileştirmesi uygulanmıştır. Son eğitim epochlarında mozaik kapatılmıştır (`close_mosaic`).

### 2.2.4 Model Eğitimi

Tüm ajanlar Google Colab ortamında NVIDIA Tesla T4 GPU kullanılarak eğitilmiştir. Eğitim altyapısı Ultralytics 8.4.39, Python 3.12.13 ve PyTorch 2.10.0 üzerine kurulmuştur. Her eğitimde önceden eğitilmiş YOLOv8 ağırlıkları başlangıç noktası (transfer learning) olarak kullanılmış; model baş katmanı hedef sınıf sayısına göre yeniden yapılandırılmıştır.

**Transfer learning stratejisi:** Önceden eğitilmiş modelin özellik çıkarım katmanları dondurulmamış; tüm ağ ağırlıkları görev verisine göre ince ayar yapılmıştır. Bu yaklaşım, genel nesne tespiti bilgisinin korunmasını sağlarken yeni görev için daha az epoch ile yakınsama imkânı sunmaktadır.

**YOLOv8n/s/m seçimi:** PPE ajanlarının büyük çoğunluğunda `yolov8s.pt` (small) mimarisi tercih edilmiştir. Bunun nedeni, küçük ölçekli veri setlerinde daha büyük mimarilerin aşırı öğrenmeye yol açma riskidir. Yangın/duman ajanı ve scene maske ajanı için `yolov8m.pt` (medium) kullanılmıştır; zira yangın/duman veri seti büyük ve sahne bağlamı geniş olduğundan, maske ise sahne modunda küçük nesne olduğundan daha büyük kapasite ve yüksek giriş çözünürlüğünden (imgsz=960) fayda sağlamaktadır.

Çizelge 2.2'de tüm ajanların temel eğitim parametreleri özetlenmektedir.

**Çizelge 2.2.** Ajan bazlı eğitim parametreleri

| Ajan | Base model | Epoch hedefi | Gerçek epoch | imgsz | Batch | Optimizer | Patience |
|---|---|---:|---:|---:|---:|---|---:|
| Kişi tespiti | yolov8s | 100 | 68 | 640 | 16 | AdamW | 20 |
| Yangın/duman | yolov8m | 100 | — | 640 | 32 | auto | 30 |
| Crop baret | yolov8s | 140 | 135 | 640 | 16 | auto | 25 |
| Crop yelek | yolov8s | 100 | 100 | 640 | 16 | AdamW | 30 |
| Crop maske | yolov8s | 200 | 130 | 640 | 16 | auto | 25 |
| Scene baret | yolov8s | 200 | 109 | 640 | 16 | auto | 25 |
| Scene yelek | yolov8s | 200 | 124 | 640 | 16 | auto | 25 |
| Scene maske | yolov8m | 200 | — | 960 | 16 | AdamW | 30 |

Bazı eğitimlerde `optimizer='auto'` ayarı kullanılmıştır. Bu durumda Ultralytics, veri seti büyüklüğü ve batch boyutuna göre optimizer seçimini otomatik yapmaktadır. Örneğin crop yelek ajanında `optimizer=auto` ile AdamW (lr=0.001667, momentum=0.9) seçilmiştir.

**Early stopping:** Tüm eğitimlerde doğrulama metriği belirli sayıda epoch boyunca iyileşmediğinde eğitim otomatik olarak durdurulmuştur. Bu mekanizma hem eğitim süresini kısaltmış hem de aşırı öğrenmeyi azaltmıştır. Her eğitim sonunda `best.pt` ve `last.pt` checkpoint dosyaları ile `results.csv` ve `args.yaml` konfigürasyon dosyaları Google Drive'a yedeklenmiştir.

### 2.2.5 Model Değerlendirme Metrikleri

Model performansları dört temel metrik üzerinden değerlendirilmiştir: Precision (kesinlik), Recall (duyarlılık), mAP@0.50 ve mAP@0.50:0.95.

**Precision**, modelin pozitif tahmin ettiği örneklerin gerçekten pozitif olma oranını ölçer. **Recall**, gerçekte pozitif olan örneklerin modelin doğru tespit edebildiği oranını ifade eder. **mAP@0.50**, IoU eşiği 0.50 olduğunda tüm sınıflar için hesaplanan ortalama hassasiyet alanını (Average Precision) gösterir. **mAP@0.50:0.95** ise IoU eşiğini 0.50 ile 0.95 arasında 0.05 adımlarla değiştirip ortalamasını alır; bu daha katı bir metrik olup model kalibrasyonunu daha iyi yansıtır.

Tüm model değerlendirmeleri eğitim sürecinden bağımsız tutulan test bölmesi üzerinde gerçekleştirilmiştir. Sınıf dengesizliği göz önünde bulundurularak her sınıf için ayrı Precision ve Recall değerleri de raporlanmıştır.

### 2.2.6 Model Karşılaştırmaları ve Nihai Seçim Kararları

**Yangın/duman ajanı:** İlk denemede yangın ve duman için ayrı modeller eğitilmiştir. Ancak duman modeli özellikle zayıf kalmıştır (mAP@0.50: 0.485). Üç sınıfı birleştiren `fire_smoke_other_agent_final_best.pt` modeli belirgin biçimde daha iyi sonuç vermiş (mAP@0.50: 0.837) ve final sistemde ortak ajan olarak kullanılmıştır.

**Yelek ajanı (crop modu):** Crop yelek ajanı için `compare_vest.py` betiği ile iki farklı model karşılaştırılmıştır. `cropvest_agent_final_best.pt` modeli doğrulama bölmesinde tüm sınıflarda daha dengeli metrikler ürettiği için final seçilmiştir.

**Baret karşılaştırması (crop vs scene):** Scene baret modelinde `NO-Hardhat` sınıfı Recall değeri 0.610 ile düşük kalırken, crop baret modelinde bu değer 0.884'e yükselmiştir. Bu fark, crop tabanlı yaklaşımın küçük ve baş bölgesine bağlı nesnelerde sağladığı odaklanma avantajını doğrudan yansıtmaktadır.

**Scene maske ajanı:** Önceki `yolov8s` tabanlı model güçlü sonuç verse de maske küçük bir nesne olduğundan final scene maske ajanı için `yolov8m` backbone ve `imgsz=960` kombinasyonu tercih edilmiştir. Bu seçim daha yüksek model kapasitesi ve giriş çözünürlüğü ile küçük nesne tespitini desteklemeyi amaçlamaktadır.

Final ajan seçimi yalnızca en yüksek tekil metriklere göre değil; scene/crop karşılaştırma amacını karşılama, kendi eğitilmiş modüler sistem gereksinimi ve gerçek video pipeline'ındaki davranış birlikte değerlendirilerek yapılmıştır.

---

## 2.3 Detection Pipeline

### 2.3.1 Pipeline Genel Akışı ve Kaynak Yönetimi

Tespit pipeline'ı, `run_live_video.py` adlı tek bir Python betiği içinde bütünleşik biçimde gerçekleştirilmiştir. Pipeline başlatıldığında üç tür görüntü kaynağı kabul edilmektedir: USB veya yerleşik web kamerası (kamera indeksiyle belirtilir), yerel video dosyası ve RTSP/HTTP ağ akışı. Kaynak türü `--source` argümanıyla belirlenmekte; `--mode crop` veya `--mode scene` argümanıyla hangi PPE tespit mimarisinin kullanılacağı seçilmektedir.

Her çerçeve (frame) işlenirken belirli bir sıra izlenmektedir: önce kişi tespiti ve takibi gerçekleştirilmekte, ardından seçili moda göre PPE tespiti yapılmakta, temporal voting ile kararlı PPE durumu belirlenmekte, olay durum makinesi çalıştırılmakta ve yangın tespiti gerçekleştirilmektedir. Kamera izleme ise her çerçevede arka planda yürütülen bağımsız bir süreç olarak çalışmaktadır.

Modeller yalnızca bir kez yüklenmekte ve tüm pipeline boyunca bellekte tutulmaktadır. CUDA desteği mevcutsa tüm çıkarım işlemleri GPU üzerinde gerçekleştirilmekte; aksi takdirde CPU'ya otomatik geçiş yapılmaktadır. PPE çıkarımı performans optimizasyonu amacıyla her 4 çerçevede bir (`PPE_INFER_EVERY = 4`), yangın çıkarımı ise her 5 çerçevede bir (`FIRE_INFER_EVERY = 5`) yapılmaktadır; takip ve olay makinesi ise her çerçevede çalışmaya devam etmektedir.

### 2.3.2 Kişi Tespiti ve ByteTrack Takibi

Her çerçeve, kişi tespiti modeline tam boyutuyla verilmektedir. Model, çerçeve içindeki insan figürlerine karşılık gelen sınırlayıcı kutular (bounding box) üretmektedir. Bu kutular, Ultralytics'in entegre takip arayüzü aracılığıyla doğrudan ByteTrack algoritmasına aktarılmakta ve her kişiye benzersiz bir takip kimliği (track ID) atanmaktadır.

ByteTrack algoritması, standart izleyicilerden farklı olarak yalnızca yüksek güven skorlu tespitlerle değil, düşük güven skorlu adaylarla da eşleştirme yapmaktadır. Bu özellik, kısa süreli oklüzyonlarda ya da modelin bir kişiyi geçici olarak düşük güvenle tespit ettiği durumlarda kimlik sürekliliğini korumaktadır. ByteTrack'in davranışı `bytetrack.yaml` dosyası aracılığıyla yapılandırılmaktadır: `track_buffer` parametresi izleme tampon uzunluğunu (kayıp track'ların bellekte tutulacağı çerçeve sayısını), `match_thresh` eşleştirme eşiğini ve `min_box_area` geçerli bir tespit için minimum kutu alanını belirlemektedir.

### 2.3.3 Stabil Kişi Kimliği — TrackReattacher

ByteTrack dahil tüm çok nesne izleyicileri, belirli durumlarda kişi kimliklerini yeniden atayabilmektedir. Bu durum, bir kişinin başka bir kişinin arkasına geçtiği (oklüzyon) veya kısa süre görüş alanı dışına çıktığı senaryolarda ortaya çıkmaktadır. Kimlik değişimi, iş güvenliği izleme bağlamında ciddi bir sorun oluşturmaktadır: aynı kişinin farklı kimliklerle izlenmesi, ihlal geçmişinin bölünmesine ve hatalı alarm üretilmesine yol açmaktadır.

Bu sorunu çözmek amacıyla özgün bir kimlik kararlılaştırma bileşeni olan TrackReattacher (`tracking_identity.py`) geliştirilmiştir. Bu bileşen, ham ByteTrack kimliklerini (`raw_tid`) kararlı kişi kimliklerine (`stable_pid`) eşlemekte; kayıp bir track yeniden göründüğünde aynı kişi olup olmadığını birden fazla sinyal kullanarak değerlendirmektedir.

**Çok sinyalli skor fonksiyonu:** Bir yeni tespitin bellekteki kayıp bir track ile eşleşip eşleşmediği ağırlıklı bir skor fonksiyonuyla belirlenmektedir. Toplam skor aşağıdaki beş bileşenin ağırlıklı toplamından oluşmaktadır:

```
S = 0.40 × S_merkez + 0.25 × S_alan + 0.15 × S_oran + 0.15 × S_zaman + 0.05 × S_ppe
```

- **S_merkez (0.40):** Yeni tespitin merkezi ile hafızadaki son konumdan hız tahminiyle öngörülen konumun mesafe benzerliği. Dinamik eşik olarak kişi yüksekliğinin 0.35 katı veya en az 40 piksel kullanılmaktadır.
- **S_alan (0.25):** Yeni bbox alanı ile hafızadaki bbox alanının oransal benzerliği. Bu sinyal, aynı uzaklıkta görünen kişilerin benzer alan değerlerine sahip olduğunu varsaymaktadır.
- **S_oran (0.15):** Bounding box en-boy oranı benzerliği. Aynı kişinin duruşu genellikle kısa sürede değişmemektedir.
- **S_zaman (0.15):** Kaybolma süresi benzerliği. Uzun süre kayıp kalan track'lerin eşleşme olasılığı düşüktür.
- **S_ppe (0.05):** PPE imzası benzerliği. Kayıp track'in son bilinen baret/yelek/maske durumu, yeni tespitin PPE durumuyla karşılaştırılmaktadır. Durum bilinmiyorsa nötr değer (0.5) kullanılmaktadır.

Toplam skorun 0.70 eşiğini geçmesi durumunda eşleştirme yapılmakta ve yeni ham kimlik, mevcut kararlı kimliğe bağlanmaktadır. Eşiği geçemeyen tespitler yeni bir kararlı kimlik olarak kaydedilmektedir. Bellekteki kayıp track'ler 2 saniyelik (`REATTACH_WINDOW_SEC = 2.0`) veya 60 çerçevelik (`MAX_GAP_FRAMES = 60`) süre sonunda temizlenmektedir.

Greedy atama algoritmasıyla en yüksek skorlu eşleştirme seçilmekte ve bir kararlı kimliğin aynı anda yalnızca tek bir ham kimliğe atanması sağlanmaktadır. `stable_pid`, tüm PPE durum takibinin, olay oluşturmanın ve veritabanı yazımının temel referans noktasıdır.

### 2.3.4 Crop-Based PPE Detection Modu

Crop tabanlı modda PPE tespiti, tam çerçeve yerine izlenen kişiye ait bölgesel görüntü üzerinde gerçekleştirilmektedir. Bu yaklaşımın temel mantığı, modelin tüm sahne gürültüsünden arındırılmış, yalnızca ilgili anatomik bölgeyi içeren odaklı bir görüntü üzerinde karar vermesini sağlamaktır.

**Bölge kırpma:** Her kişi için baret, yelek ve maske tespitleri farklı kırpık bölgeler üzerinde yapılmaktadır. Kırpık bölgeler, kişi bounding box'unun boyutlarına oransal olarak hesaplanmaktadır:

- **Baret (helmet):** Kişi kutusunun üst bölgesi kullanılmaktadır. Yatayda kişi genişliğinin %10'u kadar dışarı taşılırken dikey olarak yukarıda %15, aşağıda kişi yüksekliğinin %40'ına kadar inilmektedir.
- **Yelek (vest):** Kişi kutusunun gövde/torso bölgesi kullanılmaktadır. Yatayda %15 genişletme yapılırken dikey olarak kişi yüksekliğinin %10'undan %90'ına kadar olan bölge alınmaktadır.
- **Maske (mask):** Kişi kutusunun üst/yüz bölgesi kullanılmaktadır. Yatayda %15 genişletme yapılırken dikey olarak yukarıda %10, aşağıda kişi yüksekliğinin %45'ine kadar inilmektedir.

Kırpık bölge görüntü sınırları içine kırpılmakta ve minimum 40×40 piksel boyutu koşulunu sağlamayan kırpıklar işleme alınmamaktadır.

**Geometrik doğrulama (`_validate_ppe_scored`):** Kırpık üzerindeki PPE tespiti ham model çıktısıyla bırakılmamaktadır. Tespit koordinatları kırpık koordinat sisteminden tam çerçeve koordinatlarına geri dönüştürülmekte ve anatomik bölgeyle örtüşme oranı hesaplanmaktadır. Bu oran belirli bir eşiğin altında kalan tespitler geçersiz sayılmaktadır. Ayrıca komşu kişi kutularıyla çakışma kontrol edilmekte; bir PPE tespitinin birden fazla kişiyle çakışması durumunda hedef kişiye ait olup olmadığı değerlendirilmektedir.

**One-to-one greedy atama (`_global_assign_ppe`):** Birden fazla kişi olduğu durumlarda aynı PPE tespitinin iki ayrı kişiye atanmasını önlemek amacıyla açgözlü (greedy) bir atama stratejisi uygulanmaktadır. Her PPE kategorisi için en yüksek doğrulama skoruna sahip eşleştirme önce kesinleştirilmekte, böylece bire-bir atama sağlanmaktadır.

### 2.3.5 Scene-Based PPE Detection Modu

Sahne tabanlı modda PPE ajanları tam çerçeve üzerinde çalışmaktadır. Bu yaklaşımda kişi kırpma işlemi yapılmamakta; PPE modeli tüm sahneye uygulanmakta ve üretilen tespitler geometrik bir atama yöntemiyle kişilere bağlanmaktadır.

**Kişi-PPE eşleştirmesi (`_best_scene`):** Her PPE tespiti için içerme oranı (inside_frac) hesaplanmaktadır. Bu oran, PPE bounding box'unun kişi bounding box'u içinde kalan kesişim alanının PPE kutusu alanına oranı olarak tanımlanmaktadır:

```
inside_frac = alan(PPE_bbox ∩ kişi_bbox) / alan(PPE_bbox)
```

`inside_frac` değeri 0.40 eşiğini geçen tespitler ilgili kişiye atanabilir kabul edilmektedir. Bu eşiğin 1.0 değil 0.40 olarak belirlenmesinin nedeni, bounding box'ların hareket bulanıklığı, perspektif veya model hassasiyeti nedeniyle tam örtüşme sağlamamasıdır. Aynı kişi için aynı kategoride birden fazla uygun tespit bulunması durumunda en yüksek güven skoruna sahip tespit seçilmektedir.

Atama işlemi baret, yelek ve maske kategorileri için bağımsız olarak uygulanmakta; herhangi bir kategori için uygun tespit bulunamadığında o kişinin o kategorideki durumu belirsiz (`unknown`) olarak işaretlenmektedir.

### 2.3.6 İki Modun Tasarım Farkları ve Trade-off'ları

Crop tabanlı ve sahne tabanlı yaklaşımlar aynı temel altyapı (kişi tespiti, ByteTrack, TrackReattacher, temporal voting, olay makinesi) üzerine inşa edilmiş olmakla birlikte PPE tespit stratejisi açısından temel farklılıklar barındırmaktadır.

Crop tabanlı yaklaşımın temel avantajı, PPE modelinin arka plan karmaşasından arındırılmış, anatomik bölgeyle sınırlı görüntü üzerinde çalışmasıdır. Baret ve maske gibi küçük ve konumsal olarak belirli nesnelerin tespitinde kırpık görüntünün sağladığı odaklanma etkisi ölçülebilir katkı sunmaktadır. Öte yandan bu yaklaşım kişi tespiti kalitesine doğrudan bağımlıdır; kişi bounding box'u yanlışsa kırpık bölge de hatalı olacaktır. Ayrıca her kişi için ayrı çıkarım yapılması, kişi sayısıyla doğru orantılı bir hesaplama maliyeti getirmektedir.

Sahne tabanlı yaklaşımın avantajı ise tam çerçeve bağlamını korumasıdır. PPE modeli sahnenin tamamını görmekte, kişi bounding box kalitesinden bağımsız olarak tespitler yapılmaktadır. Ancak birden fazla kişinin bulunduğu sahnelerde PPE kutularının doğru kişiye atanması zorlaşmaktadır. Özellikle küçük nesneler için, tüm sahnede arka plan, kalabalık ve uzaklık etkilerinin modeli yanıltma olasılığı artmaktadır.

### 2.3.7 Yangın ve Duman Tespiti

Yangın ve duman tespiti her iki çalışma modunda da aynı şekilde tam çerçeve üzerinde gerçekleştirilmektedir. Yangın ve duman kişi üzerinde taşınan PPE öğeleri olmadığından, kırpık bölge yaklaşımı bu tespit görevi için uygun değildir; tam sahne bağlamının korunması gerekmektedir.

**Çıkarım sıklığı:** Yangın modeli her 5 çerçevede bir (`FIRE_INFER_EVERY = 5`) çalıştırılmaktadır. Bu tercih, hesaplama maliyetini azaltırken gerçek zamanlı uyarı için yeterli tepki hızını koruyan bir denge kurmaktadır.

**Alan oranı filtresi:** Modelin tespit ettiği yangın bounding box'larının çerçeve alanına oranı hesaplanmaktadır. Bu oran `fire_min_area_ratio` eşiğinin (varsayılan: 0.027) altında kalan tespitler görmezden gelinmektedir. Bu filtre, uzaktaki küçük ateş veya parlamaların yanlış alarm üretmesini önlemektedir.

**Büyüme filtresi:** Gerçek bir yangın olayı zamanla büyüyen bir görsel alana sahip olma eğilimindedir. Bu varsayımdan yola çıkılarak bir büyüme filtresi uygulanmaktadır. Son `fire_growth_window` çerçeve boyunca algılanan yangın alanı geçmişi tutulmakta; bu geçmişin eski yarısının ortalaması ile yeni yarısının ortalaması karşılaştırılmaktadır. Yeni yarının ortalaması, eski yarının ortalamasının `fire_growth_factor` (varsayılan: 1.5) katını geçiyorsa alan büyümekte kabul edilmektedir:

```
eski_ort = ortalama(geçmiş[-pencere:-yarı])
yeni_ort = ortalama(geçmiş[-yarı:])
büyüyor  = yeni_ort / eski_ort ≥ fire_growth_factor
```

Bir tespit hem alan oranı hem büyüme koşullarından en az birini karşıladığında ham yangın sinyali olarak kabul edilmektedir. Duman (`smoke`) tespitleri ise bu filtrelere tabi tutulmaksızın doğrudan iletilmektedir; çünkü duman kısa sürede büyük alanlara yayılabilmekte ve alan oranı eşiğini her zaman karşılamayabilmektedir.

### 2.3.8 Temporal Voting (Zamansal Oylama)

Nesne tespit modelleri, aynı kişiye ait ardışık çerçevelerde tutarsız çıktılar üretebilmektedir. Örneğin baret takan bir kişi için model bir çerçevede `Hardhat`, bir sonrakinde `NO-Hardhat` kararı verebilir. Bu tutarsızlığın doğrudan olay üretimine yansıması, gereksiz alarmların ve kimlik karmaşasının önüne geçilmesini güçleştirmektedir.

Bu sorunu çözmek amacıyla her izlenen kişi (`stable_pid`) için ayrı bir geçmiş penceresi tutulmaktadır. Her çerçevede üretilen PPE kararı bu pencereye eklenmekte; olay makinesi güncel kararı almak için pencerenin tamamında çoğunluk oylaması (majority voting) yapmaktadır. Pencere boyutu `temporal_window` parametresiyle belirlenmektedir (üretim değeri: 20 çerçeve — bölüm 3.4.6'da gerçekleştirilen optimizasyon deneyi ile belirlenmiştir).

Oylama mantığına göre penceredeki kararların çoğunluğu belirli bir PPE sınıfını gösteriyorsa o sınıf geçerli karar olarak kabul edilmektedir. Bilinmeyen (`unknown`) kararların yoğun olduğu durumlarda bilinen kararlar arasında çoğunluk aranmaktadır. Bu mekanizma, geçici algılama hatalarının, ışık değişimlerinin veya kısa süreli görüş engellerinin sistemi yanlış alarma yönlendirmesini engellemektedir.

### 2.3.9 Kamera İzleme Sistemi

Sürekli izleme gerektiren fabrika ortamlarında kamera arızaları veya görüş bozulmaları tespit edilemeyen ihlallere yol açabilmektedir. Bu riski azaltmak amacıyla pipeline her çerçevede kamera durumunu üç ayrı kriter açısından değerlendirmektedir.

**Karanlık (Dark) tespiti:** Her çerçeve gri tona dönüştürülmekte ve ortalama piksel parlaklığı 255 değerine oranlanarak normalize edilmektedir. Bu değer `cam_dark_thresh` eşiğinin (varsayılan: 0.03) altında kalırsa kamera karartı sayacı artırılmaktadır. Sayaç `cam_dark_frames` (varsayılan: 60 çerçeve) değerini geçtiğinde kamera durumu `dark` olarak bildirilmektedir. Parlaklık eşiğin üstüne çıktığında sayaç sıfırlanmaktadır.

**Donma (Freeze) tespiti:** Ardışık iki çerçeve arasındaki fark görüntüsü (`cv2.absdiff`) hesaplanmakta ve ortalama piksel farkı 255 değerine oranlanmaktadır. Bu değer `cam_freeze_diff` eşiğinin (varsayılan: 0.002) altında kalırsa donma sayacı artırılmaktadır. Sayaç `cam_freeze_frames` (varsayılan: 60 çerçeve) değerini geçtiğinde durum `frozen` olarak bildirilmektedir. Donma kontrolü yalnızca kamera karanlık olmadığında yapılmaktadır.

**Çevrimdışı (Offline) tespiti:** Video veya kamera akışı kesildiğinde OpenCV'nin `cap.read()` işlevi başarısız olmakta ve sistem anında `offline` durumunu bildirmektedir.

Üç durumun herhangi birinde durum değişikliği algılandığında backend'e `POST /api/pipeline/camera-status` isteği gönderilmekte; backend bu bildirimi Socket.IO `camera_status` eventi olarak tüm bağlı istemcilere iletmektedir. Kullanıcı arayüzünde global bir uyarı banner'ı görüntülenmektedir.

---

## 2.4 Olay Yönetimi (Event State Machine)

### 2.4.1 Motivasyon ve Durum Geçiş Diyagramı

Nesne tespit modelleri saniyede onlarca çerçeve işleyebilmektedir. Bu çerçevelerin her birinde ihlal tespiti gerçekleştiğinde her çerçeveyi bağımsız bir alarm olarak kaydetmek hem veritabanını gereksiz verilerle doldurmakta hem de operatörün dikkatini dağıtmaktadır. Öte yandan modellerin geçici hata üretmesi kaçınılmazdır; tek bir çerçevede yanlış alarm, kısa süreli ışık değişimi veya anlık görüş engeli sistemi yanıltabilmektedir.

Bu sorunu çözmek amacıyla her tespit çerçevesi yerine ihlal olaylarını zaman içinde birleştiren bir durum makinesi tasarlanmıştır. Durum makinesi dört durumdan oluşmaktadır:

- **idle (bekleme):** Aktif ihlal yoktur. Sistem tespit beklenmektedir.
- **new (yeni):** İhlal belirli bir süre boyunca sürekliliğini korumuş ve olay ilk kez açılmıştır. Bu durum yalnızca bir kez tetiklenir ve veritabanına kayıt bu noktada yapılır.
- **active (aktif):** İhlal devam etmektedir. Tekrar sayacı artmakta, olay güncellenmektedir.
- **closed (kapandı):** İhlal belirli bir süre boyunca gözlemlenmemiş ve olay kapatılmıştır.

Geçiş sırası `idle → new → active → closed → idle` biçiminde gerçekleşmektedir.

### 2.4.2 Durum Geçiş Koşulları

**idle → new geçişi:** İhlal sinyali ilk alındığında bir onay sayacı başlatılmaktadır. Onay süresi boyunca (`new_confirm_sec`, varsayılan: 3.0 saniye) ihlal sinyali süreklilik gösterirse olay açılmaktadır. Bu mekanizma, anlık hatalı tespitlerin alarm üretmesini engellemektedir.

Onay süresi boyunca kısa süreli sinyal kayıpları sistemi sıfırlamamaktadır. Son ihlal tespitinden bu yana geçen süre `confirm_gap_tolerance` (varsayılan: 1.0 saniye) eşiğini aşmadığı sürece onay sayacı korunmaktadır. Bu tolerans, temporal voting penceresinin boş kaldığı geçici durumlarda gereksiz sıfırlamaları önlemektedir.

**new → active geçişi:** Olay açıldıktan sonra ihlal devam ettikçe durum `active` olarak korunmaktadır. Her çerçevede `repeat_count` sayacı artırılmakta ve olayın son görülme zamanı (`last_seen`) güncellenmektedir.

**active → closed geçişi:** İhlal sinyali kesildiğinde bir temizlenme sayacı başlatılmaktadır. Bu sayaç `resolved_confirm_sec` (varsayılan: 5.0 saniye) boyunca ihlal gözlemlenmezse olay kapatılmaktadır. Beş saniyelik bu bekleme süresi, kısa süreli kişi oklüzyonlarının veya temporal voting boşluklarının olayı erken kapatmasını engellemektedir. Temizlenme onayı sırasında `repeat_count` artışı durdurulmaktadır.

### 2.4.3 Yangın Debounce

Yangın tespitinin geçici parlamalar veya yanlış sınıflandırmalar nedeniyle tetiklenmesini önlemek amacıyla ayrı bir debounce mekanizması uygulanmaktadır. Ham yangın sinyali doğrudan kabul edilmemekte; `fire_confirm_frames` (varsayılan: 2 çerçeve) ardışık tespitte onaylanmaktadır. Benzer şekilde yangın sinyali kesildiğinde `fire_clear_frames` (varsayılan: 2 çerçeve) ardışık temiz tespit ardından yangın durumu sıfırlanmaktadır.

Bu yapı sayesinde tek bir hatalı çerçeve yangın alarmı tetiklemezken, gerçek ve süregelen bir yangın birkaç çerçeve içinde onaylanmaktadır.

### 2.4.4 Per-Kişi İhlal Takibi

Durum makinesi yalnızca sahne düzeyinde değil, kişi bazında da ihlal durumunu takip etmektedir. Her izlenen kişi (`stable_pid`) için bir `PersonViolationRecord` kaydı tutulmaktadır. Bu kayıt; kişinin kimliğini, aktif ihlal tiplerini (`no_helmet`, `no_vest`, `no_mask`) ve ihlalin başlangıç zamanını içermektedir.

**Geçmiş penceresi:** Her kişi için son `history_frames` (varsayılan: 8) çerçevede ihlal olup olmadığını gösteren bir geçmiş kuyruğu tutulmaktadır. Bu kuyruk, kişi kameradan çıktığında kullanılmaktadır.

**Kameradan çıkış grace period:** Bir kişi kamera görüş alanından çıktığında durum hemen silinmemektedir. Önce son 8 çerçevede ihlal oranına bakılmaktadır. Çerçevelerin en az yarısında ihlal varsa (`violation_ratio ≥ 0.5`), kişi ihlaller kaydedilerek `exit_grace_sec` (varsayılan: 2.0 saniye) boyunca ihlal durumu korunmaktadır. Bu süre sonunda kayıt temizlenmektedir. Çerçevelerin çoğunluğu temiz çıkmışsa kayıt anında silinmektedir. Bu mekanizma, çerçeve görüş alanı kenarında gelip giden kişilerin olay durumunu gereksiz yere dalgalandırmasını engellemektedir.

### 2.4.5 Event Signature, repeat_count ve duration_sec

**Event Signature (Olay İmzası):** Her olay, hangi ihlal tiplerini içerdiğini tanımlayan bir imza yapısıyla temsil edilmektedir. İmza dört Boolean değerden oluşmaktadır: `helmet_violation`, `vest_violation`, `mask_violation` ve `fire_detected`. İmza, o anki `_person_states` tablosundaki tüm kişilerin ihlal kayıtlarından türetilmektedir; herhangi bir kişide belirli bir ihlal tipi varsa o imza alanı `True` olmaktadır. Bu tasarım, kaç kişinin ihlal yaptığından bağımsız olarak sahne düzeyinde hangi ihlal tipinin aktif olduğunu özetlemektedir.

**repeat_count:** Olay aktif olduğu süre boyunca her çerçevede artırılan bir sayaçtır. Bu sayaç, ihlalin ne kadar süre devam ettiğinin dolaylı bir ölçüsüdür ve olay kapandığında veritabanına yazılmaktadır. Temizlenme onayı (`confirming_resolved`) aşamasında sayaç artırılmamaktadır; yalnızca aktif ihlal varlığında artış gerçekleşmektedir.

**duration_sec:** Olayın açıldığı andan son ihlal tespitine kadar geçen süreyi saniye cinsinden ifade etmektedir. `ActiveEvent` sınıfında `last_seen - start_time` formülüyle hesaplanmaktadır. Olay kapandığında hesaplanan son değer backend aracılığıyla veritabanına yazılmaktadır.

**Kayıt mantığı:** Durum makinesinin yalnızca `new` durumuna geçişte `should_save = True` döndürmesi, her çerçevenin değil yalnızca yeni açılan olayların veritabanına kaydedilmesini sağlamaktadır. `active` ve `closed` durumları ise mevcut kaydın güncellenmesiyle sonuçlanmaktadır.

---

## 2.5 Backend Sistemi

### 2.5.1 Flask Uygulama Mimarisi

Backend, Python'un Flask mikro web çerçevesi üzerine inşa edilmiştir. Flask, Blueprint gibi modüler yapılar kullanmaksızın tek bir `app.py` dosyasında tüm endpoint tanımlarını barındırmaktadır. Bu yaklaşım, küçük ve orta ölçekli projelerde okunabilirliği artırmakta ve bağımlılıkları minimize etmektedir.

Gerçek zamanlı iletişim için Flask-SocketIO kütüphanesi entegre edilmiştir. SocketIO örneği `async_mode="threading"` parametresiyle başlatılmaktadır; bu sayede eventlet veya gevent gibi ek eşzamansız çerçevelere ihtiyaç duyulmamakta, Python'un standart thread modeli kullanılmaktadır. Sunucu başlatıldığında üç işlem sırasıyla gerçekleştirilmektedir: dosya sistemi izleyicisi (`ResultsWatcher`) başlatılmakta, veritabanı bağlantısı kurulmakta ve rapor zamanlayıcısı etkinleştirilmektedir.

CORS (Cross-Origin Resource Sharing) politikası `flask-cors` kütüphanesiyle yönetilmekte; yalnızca `config.yaml` dosyasında tanımlı kaynakların (localhost:5173, localhost:5174) isteklerine izin verilmektedir.

### 2.5.2 REST API Tasarımı

Backend, kullanım amaçlarına göre gruplanmış REST endpoint'leri sunmaktadır. Çizelge 2.3'te tüm endpoint'ler özetlenmektedir.

**Çizelge 2.3.** REST API endpoint'leri

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | /api/events | Filtrelenmiş olay listesi (?date, ?violation_type, ?status) |
| GET | /api/events/\<id\> | Tek olay timeline ve notları |
| POST | /api/events/\<id\>/note | Operatör notu ekle |
| PATCH | /api/events/\<id\>/close | Olayı kapat (repeat_count, duration_sec ile) |
| PATCH | /api/events/\<id\>/false-positive | Yanlış tespit olarak işaretle |
| GET | /api/images/\<id\>/\<dosya\> | Olay fotoğrafı (JPEG) |
| GET | /api/stats | Dashboard istatistikleri |
| GET | /api/reports | Grafik verisi (günlük/haftalık/aylık) |
| GET | /api/reports/summary | Risk skoru, trend, lokasyon analizi |
| POST | /api/reports/summary/llm | LLM raporu üret (asenkron) |
| GET | /api/reports/saved | Kayıtlı LLM raporları listesi |
| GET | /api/reports/saved/\<id\> | Tek kayıtlı rapor |
| GET | /api/reports/export/csv | CSV dışa aktar |
| GET | /api/reports/export/pdf | PDF dışa aktar |
| GET | /api/pipeline/status | Pipeline durumu |
| POST | /api/pipeline/start | Pipeline başlat |
| POST | /api/pipeline/stop | Pipeline durdur |
| GET | /api/pipeline/browse | Windows dosya seçici |
| POST | /api/pipeline/camera-status | Kamera durum bildirimi |
| GET | /api/config | PPE konfig oku |
| PUT | /api/config | PPE konfig güncelle |

Rapor endpoint'lerinde dönem hesaplama birleşik bir yardımcı fonksiyon (`_summary_date_range`) aracılığıyla yapılmaktadır. Günlük raporlar tek bir tarihi, haftalık raporlar o haftanın Pazartesi'sinden Pazar'ına kadar olan aralığı, aylık raporlar ise ayın ilk gününden son gününe kadar olan aralığı kapsamaktadır.

### 2.5.3 Socket.IO Gerçek Zamanlı Eventler

HTTP polling yerine WebSocket tabanlı Socket.IO protokolü kullanılması; düşük gecikme, sunucu kaynaklı bildirim desteği ve bağlantı durumunu yönetme kolaylığı sağlamaktadır. Sistem dört farklı Socket.IO eventi yayınlamaktadır:

**`new_alert`:** Dosya sistemi izleyicisi (`ResultsWatcher`) `results/` dizininde yeni bir JSON dosyası tespit ettiğinde yayınlanmaktadır. Payload; olay kimliğini, durumunu, zaman damgasını ve ihlal imzasını içermektedir. Bu event, frontend'in alarm geçmişini ve anlık bildirim panelini otomatik olarak güncellemesini sağlamaktadır.

**`event_closed`:** Bir olay kapatıldığında, manuel olarak çözüldüğünde veya yanlış tespit olarak işaretlendiğinde yayınlanmaktadır. Frontend bu eventi alarak ilgili kaydın görünümünü günceller.

**`camera_status`:** Tespit pipeline'ı kamera durumunu `POST /api/pipeline/camera-status` endpoint'ine bildirdiğinde backend bu veriyi Socket.IO üzerinden tüm istemcilere iletmektedir. Payload; durum bilgisini (`online`, `frozen`, `dark`, `offline`), kamera kimliğini ve bölge adını içermektedir.

**`report_llm_ready` / `report_llm_error`:** LLM raporu üretimi tamamlandığında `report_llm_ready`, bir hata oluştuğunda ise `report_llm_error` eventi yayınlanmaktadır. Bu eventler rapor üretiminin asenkron yapısını mümkün kılmaktadır; HTTP isteği anında yanıt dönerken rapor arka planda hazırlanmakta ve tamamlandığında istemci bilgilendirilmektedir.

### 2.5.4 Veritabanı Katmanı

Backend, veritabanı bağlantısının durumuna göre iki farklı okuma/yazma stratejisini desteklemektedir.

**Veritabanı modu:** `config.yaml` dosyasında `database.enabled: true` olduğunda ve bağlantı başarıyla kurulduğunda tüm okuma ve yazma işlemleri PostgreSQL üzerinden gerçekleştirilmektedir. `backend/database/reader.py` modülü olay listeleme, istatistik hesaplama ve rapor verisi çekme işlevlerini; `backend/database/writer.py` modülü ise olay kaydetme, kapatma ve LLM raporu yazma işlevlerini üstlenmektedir. Bağlantı havuzu (`connection pool`) yönetimi `backend/database/connection.py` modülünde merkezi olarak sağlanmaktadır; bu yaklaşım her istek için ayrı bağlantı kurmanın getireceği gecikmeyi ve bağlantı sayısı sınırlamalarını ortadan kaldırmaktadır.

**Dosya sistemi fallback modu:** Veritabanı devre dışıysa veya bağlantı kurulamazsa sistem `backend/event_reader.py` modülüne geçmektedir. Bu modül `results/` dizinindeki JSON dosyalarını okuyarak aynı arayüzü sunmaktadır. Dosya sistemi modu temel olay listeleme ve görsellere erişimi desteklemekte; ancak analitik özet endpoint'leri (`/api/reports/summary`) bu modda 503 döndürmektedir.

Bu iki katmanlı yapı sayesinde tespit pipeline'ı veritabanı durumundan bağımsız olarak çalışmaya devam edebilmektedir.

### 2.5.5 Dosya Sistemi İzleyici (Watcher)

Tespit pipeline'ı bir olay oluşturduğunda doğrudan veritabanına yazmak yerine `results/` dizinine JSON dosyası bırakmaktadır. `ResultsWatcher` sınıfı, bu dizini gerçek zamanlı olarak izleyerek yeni dosyaları algılamakta ve işlemektedir.

İzleme, Python'un Watchdog kütüphanesi [21] aracılığıyla gerçekleştirilmektedir. Windows işletim sisteminde Watchdog varsayılan olarak `WindowsApiObserver` kullanmaktadır; bu izleyici, NTFS dosya sistemi bildirimlerine dayanmaktadır.

**Windows NTFS deduplication:** Windows'ta NTFS dosya sistemi aynı dosya için `on_created` eventini birden fazla kez tetikleyebilmektedir. Bu sorunu önlemek amacıyla her işlenen dosya yolu bir sözlükte zaman damgasıyla birlikte kaydedilmektedir. Aynı dosya için gelen ikinci event, ilk işlemden bu yana geçen süre 10 saniyeden (`_DEDUP_TTL = 10.0`) kısa ise görmezden gelinmektedir.

**Yazma tamamlanma bekleme:** NTFS'te dosya oluşturma eventi, dosyaya yazma işlemi tamamlanmadan önce tetiklenebilmektedir. Bu durumda JSON dosyası henüz tam yazılmamış olabileceğinden 200 milisaniyelik bir bekleme süresi uygulanmaktadır. Ardından dosya okunmakta, veritabanına yazılmakta ve `new_alert` Socket.IO eventi yayınlanmaktadır.

### 2.5.6 Pipeline Yönetimi

Backend, tespit pipeline'ını bağımsız bir alt süreç olarak başlatmakta ve yönetmektedir. `POST /api/pipeline/start` isteği alındığında Python'un `subprocess.Popen` sınıfı kullanılarak `run_live_video.py` betiği ayrı bir işlem olarak başlatılmaktadır. Komut satırı argümanları kaynak türüne (`--camera` veya `--video`), çalışma moduna (`--mode crop|scene`), kamera kimliğine ve bölge adına göre dinamik olarak oluşturulmaktadır. CUDA mevcutsa `--device cuda`, değilse `--device cpu` argümanı otomatik olarak eklenmektedir.

`POST /api/pipeline/stop` isteği alındığında `Popen.terminate()` çağrısıyla süreç sonlandırılmaktadır. Pipeline durumu `GET /api/pipeline/status` endpoint'i aracılığıyla sorgulanabilmektedir; bu endpoint çalışma durumunu, kaynak bilgisini, kamera kimliğini, bölge adını ve aktif modu döndürmektedir.

**Windows dosya seçici:** Video dosyası kaynağı için kullanıcıya görsel dosya seçim iletişim kutusu sunulmaktadır. `GET /api/pipeline/browse` endpoint'i, Windows PowerShell üzerinden `System.Windows.Forms.OpenFileDialog` açarak kullanıcının seçtiği dosya yolunu döndürmektedir. Bu yöntem, Windows ortamına özgü yerel GUI entegrasyonu sağlamakta ve web arayüzünden dosya yolu girişini kolaylaştırmaktadır.

### 2.5.7 Güvenlik ve Doğrulama

Kullanıcı ve pipeline kaynaklı girdiler, işlenmeden önce düzenli ifade (regex) kalıplarıyla doğrulanmaktadır. Bu doğrulama katmanı; hatalı girdi nedeniyle oluşabilecek uygulama hatalarını ve kötü niyetli girdi saldırılarını önlemeyi amaçlamaktadır.

Tanımlanan doğrulama kalıpları şunlardır:
- **Olay kimliği:** `^evt_[a-zA-Z0-9_]+$` — yalnızca `evt_` önekiyle başlayan alfasayısal karakterlere izin verilmektedir.
- **Görsel dosya adı:** `^evt_[a-zA-Z0-9_]+(new|update_\d+|closed)\.jpg$` — yalnızca beklenen olay görsel adı formatına izin verilmektedir.
- **Dönem:** `^(daily|weekly|monthly)$` — yalnızca üç geçerli değer kabul edilmektedir.
- **Tarih:** `^\d{4}-\d{2}-\d{2}$` — ISO 8601 tarih formatı zorunlu tutulmaktadır.
- **İhlal tipi:** `^(helmet|vest|mask|fire)$` — yalnızca dört geçerli ihlal tipi kabul edilmektedir.

**Path traversal koruması:** Olay görsellerine erişim sağlayan `/api/images/<event_id>/<filename>` endpoint'inde `../` gibi dizin geçiş karakterlerinin kullanılmasını önlemek amacıyla ek bir kontrol uygulanmaktadır. Çözümlenen dosya yolunun `results/` dizini içinde kaldığı doğrulanmaktadır; dışarı taşan istekler 403 hatası ile reddedilmektedir.

---

## 2.6 Veritabanı Tasarımı

### 2.6.1 Tablo Yapıları

Sistem dört ana tablodan oluşan ilişkisel bir veritabanı şemasına sahiptir. Şema tasarımında iki temel gereksinim gözetilmiştir: olayların güncel durumuna hızlı erişim ve olayların tüm geçmişinin eksiksiz korunması. Bu iki gereksinim birbirinden farklı yazma stratejileri gerektirdiğinden `events` ve `event_timeline` tabloları birbirini tamamlayan farklı roller üstlenecek şekilde tasarlanmıştır.

**`events` tablosu:** Her olayın güncel ve en son durumunu tutan ana tablodur. Her olay bu tabloda tek bir satırla temsil edilmektedir. Bir olay durumu değiştiğinde yeni satır eklenmez; mevcut satır UPSERT (güncelle veya ekle) işlemiyle güncellenir. Bu tasarım, anlık durum sorgularında tablo taramasını minimize etmektedir.

Tablonun temel alanları şunlardır: `event_id` (benzersiz olay kimliği, `evt_0001` formatında), `event_status` (olayın güncel durumu: `new`, `active`, `closed`), `created_at` (ilk `new` geçişinin zamanı), `updated_at` (son güncelleme zamanı), `repeat_count` (ihlal tekrar sayısı), `duration_sec` (ihlal süresi), dört adet Boolean ihlal bayrağı (`helmet_violation`, `vest_violation`, `mask_violation`, `fire_detected`), `signature` ve `persons` JSONB alanları, `camera_id`, `zone` ve `false_positive` bayrağı.

**`event_timeline` tablosu:** Olayların tüm durum geçişlerini zaman sırasıyla kaydeden geçmiş tablosudur. Her durum değişikliğinde bu tabloya yeni bir satır eklenmekte, mevcut satırlar güncellenmemektedir. Böylece bir olayın `idle → new → active → closed` yolculuğunun her adımı, değişim nedeni (`change_reason`), görsel dosya adı ve kişi bazlı ihlal detaylarıyla birlikte saklanmaktadır. Bu tablo frontend'in olay zaman çizgisi görünümünü beslemektedir.

**`event_notes` tablosu:** Operatörlerin olaylara eklediği serbest metin notlarını barındırmaktadır. Her not; olay kimliğine yabancı anahtar ile bağlıdır ve oluşturulma zamanı damgasıyla birlikte saklanmaktadır. Bir olay silindiğinde ilgili notlar `ON DELETE CASCADE` kısıtı sayesinde otomatik olarak temizlenmektedir.

**`llm_reports` tablosu:** Dönemsel LLM güvenlik raporlarını saklamaktadır. Her kayıt; dönem türünü (`daily`, `weekly`, `monthly`), dönem başlangıç tarihini, LLM tarafından üretilen rapor metnini, oluşturulma zamanını ve otomatik mı yoksa kullanıcı isteğiyle mi üretildiğini gösteren bayrağı içermektedir. `(period, report_date)` çifti üzerindeki UNIQUE kısıtı, aynı döneme ait birden fazla rapor oluşmamasını sağlamakta ve UPSERT ile mevcut rapor üzerine yazılabilmesine imkân tanımaktadır.

### 2.6.2 JSONB Kullanımı

`signature` ve `persons` alanları PostgreSQL'in JSONB veri türüyle saklanmaktadır. JSONB, JSON verilerini ikili (binary) formatta depolamakta; sorgulama için ayrıştırma maliyeti olmaksızın alan erişimi ve indeksleme imkânı sunmaktadır.

**`signature` alanı:** Olayın hangi ihlal tiplerini içerdiğini tanımlayan yapıdır. İçeriği şu şekildedir: `{"helmet_violation": true, "vest_violation": false, "mask_violation": false, "fire_detected": false}`. Bu alan hem `events` hem de `event_timeline` tablolarında bulunmaktadır; böylece her geçişte o anki ihlal bileşimi tam olarak korunmaktadır.

**`persons` alanı:** Olayın tespit edildiği andaki kişi bazlı ihlal detaylarını içermektedir. Her kişi için takip kimliği, aktif ihlal tipleri ve ihlal süresi bu yapıda saklanmaktadır. Kişi bazlı verilerin ilişkisel tablolara normalleştirilmesi yerine JSONB kullanılmasının nedeni, kişi sayısının değişken olması ve bu verinin bağımsız sorgulanmasına nadiren ihtiyaç duyulmasıdır.

Hızlı filtreleme gerektiren ihlal bayrakları (`helmet_violation`, `vest_violation` vb.) ise ayrıca Boolean sütun olarak denormalize biçimde tutulmaktadır. Bu yaklaşım, aylık baret ihlali sayısı gibi yaygın sorguların JSONB ayrıştırması yapmadan yalnızca indekslenmiş Boolean sütunlar üzerinden çalışmasını sağlamaktadır.

### 2.6.3 İndeks Stratejisi ve İdempotent Schema

**İndeks stratejisi:** Şemada on iki indeks tanımlanmıştır. `events` tablosunda sık kullanılan filtreleme ve sıralama alanlarına indeks eklenmiştir: `event_status` (durum bazlı listeleme), `created_at DESC` ve `updated_at DESC` (zaman sıralı listeleme), dört ihlal bayrağı (ihlal tipine göre filtreleme) ve `false_positive` (yanlış tespit filtrelemesi). `event_timeline` tablosunda ise `event_id` (bir olayın tüm geçişlerini çekme) ve `ts DESC` (zaman sıralı geçmiş) alanlarına indeks eklenmiştir. `event_notes` tablosunda `event_id` indeksi bulunmaktadır.

**İdempotent schema:** Şema `CREATE TABLE IF NOT EXISTS` ifadeleri kullanılarak tasarlanmıştır. Bu sayede şema betiği her çalıştırıldığında hata üretmeksizin tamamlanmakta; tablolar zaten mevcutsa atlanmaktadır. Geliştirme sürecinde sonradan eklenen sütunlar (`camera_id`, `zone`, `false_positive`, `persons`) için ise `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` ifadesi kullanılmıştır. Bu yaklaşım, şema güncellemelerinin mevcut veritabanını bozmadan uygulanabilmesini sağlamaktadır.

**UPSERT yazma stratejisi:** `events` tablosuna yazma işlemi `INSERT ... ON CONFLICT (event_id) DO UPDATE` kalıbıyla gerçekleştirilmektedir. Bu kalıp; aynı olay kimliğiyle ilk kez veri geldiğinde yeni satır eklemekte, daha sonraki güncellemelerde ise mevcut satırı güncellemektedir. `camera_id` ve `zone` alanları için `COALESCE` kullanılmaktadır; bu sayede sonraki güncellemeler bu alanları NULL ile ezmemektedir. `event_timeline` tablosuna ise her zaman `INSERT` yapılmakta, böylece geçmiş korunmaktadır.

### 2.6.4 false_positive Soft-Delete Yaklaşımı

Operatörler bir olayı yanlış tespit olarak işaretleyebildiğinde bu işlemin veritabanında nasıl temsil edileceği önemli bir tasarım kararıdır. Kaydı fiziksel olarak silmek (`hard delete`) mümkün olmakla birlikte bu yaklaşım istatistiksel analizleri bozabilmekte ve geri alınamaz veri kaybına yol açmaktadır.

Bu sistemde `false_positive` Boolean sütunu kullanılarak yumuşak silme (`soft delete`) yaklaşımı tercih edilmiştir. Bir olay yanlış tespit olarak işaretlendiğinde `false_positive = TRUE` ve `event_status = 'closed'` olarak güncellenmekte; kayıt veritabanında kalmaya devam etmektedir. Rapor hesaplamaları ve istatistik sorguları `false_positive = FALSE` koşulunu içerecek şekilde tasarlanmıştır; böylece yanlış tespitler otomatik olarak analizden dışlanmaktadır.

Bu tasarım; veri bütünlüğünü korumakta, gerektiğinde işlemin geri alınabilmesine imkân tanımakta ve yanlış tespit oranının kendisinin de izlenebilir bir metrik haline gelmesini sağlamaktadır.

---

## 2.7 LLM Entegrasyonu

### 2.7.1 Ollama ile Yerel LLM Çalıştırma ve Model Seçimi

Büyük dil modelleri genellikle bulut tabanlı API servisler aracılığıyla kullanılmaktadır. Bu yaklaşım; ağ gecikmesi, kullanım başına maliyet ve hassas güvenlik verilerinin üçüncü taraf sunuculara gönderilmesi gibi dezavantajlar barındırmaktadır. Bu sistemde söz konusu dezavantajları ortadan kaldırmak amacıyla LLM çıkarımı yerel ortamda Ollama çerçevesi aracılığıyla gerçekleştirilmektedir.

Ollama, büyük dil modellerini yerel donanımda çalıştırmayı kolaylaştıran açık kaynaklı bir araçtır. HTTP tabanlı bir API sunucusu olarak çalışmakta ve istemci uygulamalarla `http://localhost:11434` adresinden iletişim kurmaktadır. Modeller Ollama deposundan indirilerek yerel GPU veya CPU üzerinde çalıştırılmaktadır.

Sistem için qwen3:8b modeli seçilmiştir. Bu tercih birkaç gerekçeye dayanmaktadır: Türkçe dil desteğinin güçlü olması, 8 milyar parametrelik yapısıyla RTX 3060 6 GB VRAM sınırında çalışabilmesi ve çıkarım hızının gerçek zamanlı raporlama için yeterli olması. Modelin sıcaklık parametresi (`temperature`) 0.3 olarak ayarlanmıştır; düşük sıcaklık değeri, modelin yaratıcı ifadelerden kaçınarak verilen istatistiklere sadık, tutarlı ve olgusal çıktılar üretmesini sağlamaktadır.

### 2.7.2 SafetyReportAgent Tasarımı

`SafetyReportAgent` sınıfı (`llm/safety_report_agent.py`), analitik servisler tarafından üretilen özet verisini alarak Ollama HTTP API'sine istek göndermekte ve model çıktısını işleyerek son raporu döndürmektedir.

**Prompt mühendisliği:** Raporun kalitesi büyük ölçüde modele verilen talimatların (prompt) yapısına bağlıdır. Bu sistemde prompt birkaç katmandan oluşmaktadır:

İlk katmanda modele rol tanımı verilmektedir: "Sen [Fabrika] iş güvenliği izleme sisteminin analiz asistanısın." Ardından ihlal türü karşılıkları açıklanmaktadır (`helmet_violation=baret ihlali`, `vest_violation=güvenlik yeleği ihlali` vb.), çünkü modelin JSON alanlarını Türkçe anlamlara doğru eşlemesi gerekmektedir.

İkinci katmanda analitik servislerden önceden hesaplanmış bağlam bilgisi eklenmektedir: en yüksek ihlal türü ve sayısı, en kritik lokasyon ve olay yoğunluğu, önceki döneme göre trend ve yüzde değişim, risk seviyesi ve normalize skor. Bu bilgiler modele önceden sunularak gereksiz hesaplama adımlarının önüne geçilmekte ve çıktının odak noktası belirlenmektedir.

Üçüncü katmanda rapor bölümleri ve kurallar tanımlanmaktadır. Bölümler: Genel Değerlendirme, Kritik Bulgular, Trend Analizi (veri mevcutsa) ve Eylem Önerileri. Kurallar arasında "Her iddianda veriden bir sayı veya lokasyon adı referans ver", "genel ifade kullanma, spesifik ol" ve "en fazla 3 öneri maddesi" yer almaktadır.

Son katmanda tam JSON özet verisi modele verilmekte ve "Rapor:" ifadesiyle model çıktısı başlatılmaktadır.

**`/think` token temizleme:** Qwen3 modeli varsayılan olarak yanıtını üretmeden önce `<think>...</think>` blokları içinde iç muhakeme gerçekleştirmektedir. Bu bloklar kullanıcıya gösterilmesi uygun olmayan ham düşünce süreçleri içermektedir. `_strip_thinking` metodu, düzenli ifade (`re.sub`) kullanarak bu blokları son çıktıdan temizlemektedir:

```python
re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

`re.DOTALL` bayrağı, düşünce bloklarının birden fazla satıra yayıldığı durumları da kapsayacak şekilde nokta karakterinin satır sonu dahil her karakterle eşleşmesini sağlamaktadır.

### 2.7.3 Rapor Analitik Servisleri

LLM'e veri göndermeden önce ham olay kayıtları iki servis sınıfı aracılığıyla anlamlı istatistiklere dönüştürülmektedir.

**EventAnalyticsService:** Olay listesi üzerinde temel hesaplamaları yapan servistir. Tarih aralığına göre filtreleme, ihlal sayılarını biriktirme ve risk skoru hesaplama işlevlerini üstlenmektedir.

Risk skoru, farklı ihlal tiplerinin ciddiyetini yansıtacak şekilde ağırlıklı bir formülle hesaplanmaktadır:

```
ham_risk = (baret_ihlali × 3) + (yelek_ihlali × 2) + (maske_ihlali × 1) + (yangın_tespiti × 10)
```

Yangın tespiti en yüksek ağırlığı taşımaktadır; çünkü can güvenliğine yönelik acil müdahale gerektiren bir durumdur. Baret ihlali kafa yaralanması riski nedeniyle yelek ihlaline kıyasla daha yüksek ağırlık almaktadır.

Ham risk skoru, 0-100 aralığına dört risk seviyesinin sınır değerlerini koruyacak şekilde normalize edilmektedir: düşük (0-19 → 0-25), orta (20-49 → 25-50), yüksek (50-99 → 50-75), kritik (100+ → 75-100).

**ReportSummaryService:** Analitik servis üzerine inşa edilmiş ve raporlama özelinde istatistikler hesaplayan servistir. Bu servis üç farklı özet üretmektedir:

- *Günlük özet:* Saatlik ihlal dağılımı ve önceki gün karşılaştırması.
- *Haftalık özet:* Günlük dağılım, en yoğun gün ve önceki hafta karşılaştırması.
- *Aylık özet:* Haftalık dağılım, en sık ihlal türü ve önceki ay karşılaştırması.

Üç özet türünde ortak olan alanlar şunlardır: toplam olay sayısı, ihlal türü sayıları, birden fazla ihlal içeren olay sayısı, lokasyon bazlı dağılım (kamera ve bölge adına göre, olay sayısına göre azalan sırada), süre ve tekrar istatistikleri ve risk özeti.

**Trend analizi:** Önceki döneme göre olay sayısı değişimi yüzde olarak hesaplanmakta; %5'in üzerinde artış `increasing`, %5'in altında azalış `decreasing`, bu aralıkta kalan değişimler `stable` olarak sınıflandırılmaktadır.

### 2.7.4 Periyodik Rapor Üretimi

LLM raporu üretimi iki yolla tetiklenebilmektedir: kullanıcı isteğiyle (`POST /api/reports/summary/llm`) veya otomatik zamanlayıcıyla.

**Asenkron üretim:** LLM çıkarım süresi modele ve donanıma bağlı olarak onlarca saniye sürebilmektedir. Bu süre boyunca HTTP bağlantısını açık tutmak istemci tarafında zaman aşımı sorunlarına yol açabilmektedir. Bu nedenle rapor üretimi asenkron olarak gerçekleştirilmektedir: HTTP isteği anında `{"ok": true, "pending": true}` yanıtıyla sonlandırılmakta, LLM çıkarımı arka planda ayrı bir Python thread'inde yürütülmekte ve tamamlandığında `report_llm_ready` Socket.IO eventi aracılığıyla bağlı tüm istemcilere iletilmektedir. Hata durumunda ise `report_llm_error` eventi yayınlanmaktadır.

**Otomatik zamanlayıcı:** Veritabanı modu etkin olduğunda backend başlangıcında bir arka plan thread'i başlatılmaktadır. Bu thread her 30 saniyede bir saat ve dakikayı kontrol etmekte; aşağıdaki koşullar sağlandığında ilgili raporu üretmektedir:

- Saat 23:55 olduğunda: her gün günlük rapor üretilmektedir.
- Saat 23:55 ve Pazar günü olduğunda: haftalık rapor üretilmektedir.
- Saat 23:55 ve içinde bulunulan günün ertesi günü farklı bir ay olduğunda: aylık rapor üretilmektedir.

Aynı gün için birden fazla tetiklenmeyi önlemek amacıyla son üretim tarihleri bir sözlükte takip edilmektedir. Üretilen raporlar `llm_reports` tablosuna UPSERT ile kaydedilmekte; aynı dönem ve tarih için önceki rapor varsa üzerine yazılmaktadır.

---

## 2.8 Frontend Sistemi

SafetyMonitor'ün kullanıcı arayüzü, React ve Vite teknolojileriyle geliştirilmiş tek sayfalık bir web uygulamasıdır (Single Page Application, SPA). Arayüz; gerçek zamanlı alarm bildirimleri, ihlal geçmişi, periyodik raporlar, pipeline kontrolü ve sistem ayarları gibi işlevleri tek bir çatı altında sunmaktadır.

### 2.8.1 React + Vite Uygulama Yapısı

Uygulama, React 18 ve Vite derleme aracı kullanılarak oluşturulmuştur. Vite, geliştirme modunda `localhost:5173` portunda çalışmakta; backend ile yapılan API ve Socket.IO bağlantıları Vite'ın proxy mekanizması aracılığıyla `localhost:5050`'ye yönlendirilmektedir. Bu sayede geliştirme sürecinde CORS sorunları yaşanmadan hem frontend hem de backend eş zamanlı çalışabilmektedir.

Uygulamanın en üst bileşeni olan `App.jsx`, aşağıdaki sorumlulukları üstlenmektedir:

- **Sayfa yönetimi:** `page` durum değişkeni, gösterilecek sayfayı belirlemektedir (`dashboard`, `alerts`, `reports`, `camera`, `settings`). Sayfa geçişleri URL yönlendirmesi olmaksızın koşullu render ile gerçekleştirilmektedir.
- **Tema yönetimi:** `theme` durumu `localStorage`'dan başlatılmakta; `toggleTheme()` fonksiyonu çağrıldığında `document.documentElement`'e `data-theme` niteliği atanarak CSS değişkenleri anında güncellenmektedir.
- **Kamera durum banner'ı:** Backend'den gelen `camera_status` socket olayları `camStatus` durumunda saklanmakta; `online` dışındaki durumlarda sayfa üstünde görsel uyarı banner'ı gösterilmektedir.
- **Bildirim sistemi:** `new_alert` socket olayları `ToastContainer` bileşenine iletilerek ekranın köşesinde 5 saniye boyunca kayan bildirim olarak gösterilmektedir. Aktif alarm sayısı navbar'da rozet olarak yansıtılmaktadır.
- **Dashboard → Alarm Geçmişi köprüsü:** Dashboard'dan bir olaya tıklandığında `pendingSelect` durumu güncellenmekte; `AlertHistory` sayfası açıldığında ilgili olay otomatik olarak seçili konuma getirilmektedir.

Socket.IO istemcisi `socket.js` modülünde tek örnek (singleton) olarak tanımlanmakta; `/` adresine Vite proxy üzerinden bağlanmakta ve `websocket` ile `polling` taşıma modlarını desteklemektedir.

Arayüz bileşen yapısı şu şekilde düzenlenmiştir:

```
App.jsx                     — kök bileşen, yönlendirme, tema, socket
├── Navbar.jsx              — üst gezinti çubuğu, tema toggle
├── cam-status-bar          — kamera uyarı banner'ı (inline)
├── Dashboard.jsx           — istatistik ve grafik sayfası
├── AlertHistory.jsx        — ihlal listesi ve olay detayı
│   ├── Sidebar.jsx         — sol panel: filtreli olay listesi
│   └── MainPanel.jsx       — sağ panel: timeline, notlar, yanlış tespit
├── Reports.jsx             — periyodik raporlar ve LLM raporu
├── CameraSetup.jsx         — pipeline kontrolü
├── Settings.jsx            — PPE ve detection ayarları
└── ToastContainer.jsx      — kayan bildirimler
```

Tüm API çağrıları `api.js` modülünde merkezi olarak tanımlanmış `_get`, `_post`, `_put`, `_patch` ve `_download` yardımcı fonksiyonları üzerinden gerçekleştirilmektedir.

### 2.8.2 Sayfalar

**Dashboard:** Sistemin genel durumuna ilişkin özet istatistikleri dört kart (aktif alarm, bugünkü ihlal, toplam olay, en sık ihlal türü) ve bir çubuk grafik üzerinden sunan sayfadır. Grafik, Recharts kütüphanesi kullanılarak oluşturulmakta; her ihlal türüne özgü renk (baret sarı `#ffd740`, yelek turuncu `#ff8c40`, maske mavi `#66bbff`, yangın kırmızı `#ff5f5f`) uygulanmaktadır. Sayfa, her mount işleminde ve yeni alarm / alarm kapanma olaylarında `GET /api/stats` uç noktasını çağırarak verileri güncel tutmaktadır. Ayrıca son alarmların listesi de gösterilmekte; bir olaya tıklandığında Alarm Geçmişi sayfasına yönlendirme gerçekleştirilmektedir.

**Alarm Geçmişi:** İki panelli master-detail düzenini benimseyen bu sayfada sol panel ihlal listesini, sağ panel seçili olayın ayrıntılarını göstermektedir.

Sol panelde üç adet filtre sunulmaktadır: tarih, ihlal türü (baret / yelek / maske / yangın) ve durum (yeni / aktif / kapalı). Filtreler değiştiğinde `GET /api/events` uç noktası güncel parametrelerle yeniden çağrılmaktadır. `new_alert` ve `event_closed` socket olayları geldiğinde liste otomatik olarak yenilenmektedir.

Sağ panelde (`MainPanel.jsx`) seçili olayın durum geçiş adımları Timeline bileşeni aracılığıyla görselleştirilmektedir. Aynı panel üzerinden operatörler olaya metin not ekleyebilmektedir; notlar `POST /api/events/<id>/note` uç noktasıyla kaydedilmekte ve anlık olarak listede görünmektedir.

*Yanlış Tespit İş Akışı:* Operatör bir olayı yanlış alarm olarak işaretlemek istediğinde "Yanlış Alarm" düğmesine tıklamaktadır. İki adımlı onay akışı devreye girmektedir: önce isteğe bağlı bir açıklama notu girilebilmekte, ardından "Onayla" düğmesiyle `PATCH /api/events/<id>/false-positive` isteği gönderilmektedir. Bu işlem arka tarafta olayı soft-delete ile işaretlemekte; olay listeden kalkmakta ve "Yanlış Tespit" rozeti gösterilmektedir.

**Raporlar:** Periyodik ihlal verilerini ve yapay zeka tarafından üretilen güvenlik raporlarını sunan bu sayfa, kullanıcıların günlük, haftalık veya aylık dönemler arasında geçiş yapmasına olanak tanımaktadır. Seçilen döneme göre uygun bir tarih seçici (date, week veya month tipi `<input>`) görüntülenmektedir.

Sayfanın üst bölümünde ihlal türlerine göre özetlenmiş kart satırı bulunmaktadır (toplam, baret, yelek, maske, yangın). Altında üç bilgi kartı yer almaktadır: risk seviyesi (düşük / orta / yüksek / kritik), önceki dönemle kıyaslama (değişim yüzdesi ve trend yönü) ve en kritik bölge (en fazla olay üretmiş kamera-bölge çifti). İhlal dağılım grafiği, seçili dönemin alt birimlerine (günlük: saatler, haftalık: günler, aylık: haftalar) göre gruplandırılmış yığılmış çubuk grafik olarak sunulmaktadır.

AI Güvenlik Raporu bölümünde kullanıcı "Raporu Oluştur" düğmesine bastığında `POST /api/reports/summary/llm` isteği gönderilmektedir. Rapor asenkron olarak üretilmekte; tamamlandığında gelen `report_llm_ready` socket olayı yakalanarak rapor metni sayfaya yansıtılmaktadır. Üretilen ve veritabanına kayıtlı raporlar "Kaydedilmiş Raporlar" bölümünde dönem-tarih etiketiyle listelenmekte; listeden herhangi bir rapora tıklandığında içeriği sağ panelde gösterilmektedir.

**Kamera Kurulumu:** Detection pipeline'ının başlatılıp durdurulduğu bu sayfa üç ana işlev sunmaktadır. Birincisi kaynak seçimidir: "Kamera" sekmesi seçildiğinde tarayıcı `MediaDevices.enumerateDevices()` API'si aracılığıyla bağlı kameraları listelemekte, kullanıcı indeks kartlarından birini seçmektedir; "Video Dosyası" sekmesi seçildiğinde ise metin girişiyle dosya yolu girilebilmekte ya da "Gözat" düğmesiyle `GET /api/pipeline/browse` üzerinden Windows OpenFileDialog penceresi açılmaktadır. İkincisi detection modu seçimidir: "Crop-Based" veya "Scene-Based" sekme butonları arasında geçiş yapılabilmektedir. Üçüncüsü kamera tanımıdır: isteğe bağlı Kamera ID ve Bölge alanları doldurulabilmektedir; bu bilgiler rapor lokasyon analizinde kullanılmaktadır.

"Sistemi Başlat" düğmesine basıldığında `POST /api/pipeline/start` isteği gönderilmektedir. Pipeline başlatıldıktan sonra tüm form alanları devre dışı bırakılmakta, etkin kaynak, kamera ID, bölge ve mod bilgileri bilgi çubuğunda görüntülenmektedir. Sayfa ayrıca 3 saniyede bir `GET /api/pipeline/status` çağrısıyla pipeline durumunu güncellemektedir.

**Ayarlar:** Backend `config.yaml` yapılandırmasının belirli alanlarının web arayüzü üzerinden güncellenebildiği bu sayfada dört yapılandırma grubu sunulmaktadır:

- *Hangi PPE'ler tespit edilsin:* Baret, yelek, maske ve yangın tespiti checkbox'larla etkinleştirilip devre dışı bırakılabilmektedir (`use_helmet`, `use_vest`, `use_mask`, `use_fire`).
- *Tespit eşikleri:* Her model için güven skoru eşiği (confidence threshold) kaydırıcı (slider) ile 0,05 - 0,95 aralığında ayarlanabilmektedir.
- *Zamanlama:* Alarm onay süresi (`new_confirm_sec`) ve temporal oylama penceresi (`temporal_window`) kaydırıcılarla ayarlanabilmektedir.
- *Yangın filtresi:* Minimum alan oranı, büyüme faktörü ve büyüme penceresi kaydırıcılarla yapılandırılabilmektedir.

"Kaydet" düğmesine basıldığında `PUT /api/config` isteğiyle değişiklikler backend'e gönderilmektedir. Ayarlar `config.yaml`'a yazılmakta; pipeline yeniden başlatıldığında aktif olmaktadır.

### 2.8.3 Gerçek Zamanlı Güncelleme ve Kamera Durum Banner'ı

Frontend ile backend arasındaki gerçek zamanlı iletişim Socket.IO aracılığıyla sağlanmaktadır. Farklı sayfa bileşenleri aynı socket örneğini kullanmakta; her bileşen kendi ihtiyaç duyduğu olaylara subscribe olmakta ve `useEffect` temizleme fonksiyonuyla ayrılmaktadır.

Aşağıdaki tablo frontend'in dinlediği socket olaylarını ve bu olayların tetiklediği arayüz güncellemelerini özetlemektedir:

| Socket Olayı | Dinleyen Bileşen | Tetiklenen Güncelleme |
|---|---|---|
| `new_alert` | App, Dashboard, AlertHistory | Toast bildirimi; istatistik ve olay listesi yenilenir |
| `event_closed` | App, Dashboard, AlertHistory | Aktif alarm sayacı azalır; liste yenilenir |
| `camera_status` | App | Kamera durum banner'ı güncellenir |
| `report_llm_ready` | Reports | LLM rapor metni görüntülenir; kaydedilmiş liste yenilenir |
| `report_llm_error` | Reports | Hata mesajı görüntülenir |

**Kamera durum banner'ı:** `camera_status` olayı alındığında `App.jsx`'teki `camStatus` durumu güncellenmektedir. Durum `online` dışında bir değer aldığında (`offline`, `frozen`, `dark`) sayfa içeriğinin üstünde tam genişlikte renkli bir uyarı şeridi belirmektedir. Bu şerit kameraya özgü Türkçe açıklama mesajını ve bir kapatma düğmesini içermektedir. Kapatma düğmesine basıldığında durum `online` sıfırlanmaktadır; ancak kameradan yeni bir uyarı gelirse şerit yeniden görünmektedir.

### 2.8.4 Tema Sistemi

Uygulama, aynı anda hem açık (light) hem de koyu (dark) görünümü desteklemekte ve kullanıcının tercihini oturumlar arasında korumaktadır.

**Tema durumu:** `App.jsx`'te başlatma anında `localStorage.getItem('theme')` ile önceki tercih okunmakta; yoksa `dark` varsayılan olarak atanmaktadır. `document.documentElement.setAttribute('data-theme', ...)` çağrısı React render döngüsü dışında senkron olarak yapıldığından sayfa yenilemelerinde beyaz ekran yanıp sönmesi önlenmektedir.

**CSS değişkenleri:** `global.css` dosyasında `:root` kapsamında koyu tema renkleri varsayılan olarak tanımlanmaktadır. `[data-theme="light"]` seçicisi bu değerleri açık tema renkleriyle geçersiz kılmaktadır. Tüm 16 CSS modülü `var(--bg)`, `var(--surface)`, `var(--text)` gibi değişkenleri kullanmakta; böylece tek bir `data-theme` niteliği değişimiyle tüm arayüz uyumlu biçimde güncellenmektedir.

**Grafik tema entegrasyonu:** Recharts tabanlı grafikler renk değişkenlerini CSS üzerinden otomatik olarak alamadığından, `Dashboard.jsx` ve `Reports.jsx`'te `CHART_STYLES = { dark: {...}, light: {...} }` nesnesi tanımlanmıştır. Bu nesne; eksen etiket rengi, tooltip arkaplan ve kenarlık rengi, ızgara rengi ve legend rengi gibi grafik stillerini içermektedir. `theme` prop'u ile doğru stil nesnesi seçilmekte ve grafikler tema değişiminde aynı render döngüsünde güncellenmektedir.

**Navbar:** Koyu ve açık mod arasında geçiş için Navbar'da bir toggle düğmesi bulunmaktadır. Koyu modda güneş ikonu, açık modda ay ikonu görüntülenmektedir. Navbar kendisi her iki modda da koyu arka plan rengiyle kalmakta; CSS değişkeni olarak `--bg-nav` ayrı tanımlanmaktadır.

### 2.8.5 Export Sistemi

Kullanıcılar, ihlal verilerini iki farklı formatta dışa aktarabilmektedir: CSV ve PDF. Export işlemleri Raporlar sayfasındaki düğmelerle tetiklenmekte; `api.js`'deki `_download()` yardımcı fonksiyonu kullanılmaktadır. Bu fonksiyon, ilgili uç noktaya GET isteği yapmakta, dönen ikili içeriği `Blob` olarak okumakta, `URL.createObjectURL()` ile geçici bir nesne URL'i oluşturmakta ve tarayıcı üzerinden otomatik indirme başlatmaktadır; ardından bellek sızıntısını önlemek için URL serbest bırakılmaktadır.

**CSV export:** `GET /api/reports/export/csv` uç noktası çağrıldığında backend, seçili dönem ve tarih parametrelerine göre veritabanından olay verilerini sorgulamaktadır. Oluşturulan CSV dosyasına UTF-8 BOM (byte order mark, `﻿`) eklenmektedir. Bu karakter, dosyanın Microsoft Excel'de açıldığında Türkçe karakterlerin (ç, ş, ı, ğ, ü, ö) doğru görüntülenmesini sağlamaktadır.

**PDF export:** `GET /api/reports/export/pdf` uç noktası çağrıldığında backend, Python'un ReportLab kütüphanesini kullanarak yapılandırılmış bir PDF belgesi oluşturmaktadır. Belge iki bölümden oluşmaktadır: ilk bölüm dönem özetini (toplam olay sayısı, ihlal türü dağılımı, risk skoru) içeren bir özet tablosudur; ikinci bölüm ise her olayın olay ID'si, tarih-saat, ihlal türleri, kamera ID ve bölge bilgilerini içeren ayrıntılı olay tablosudur. İçerik dinamik olarak hesaplandığından seçili dönem ve tarih değiştiğinde farklı bir PDF üretilmektedir.

---

# 3. BULGULAR VE DEĞERLENDİRME

## 3.1 Test Ortamı

Sistemin performansını değerlendirmek için kullanılan donanım ve yazılım ortamı aşağıdaki tabloda özetlenmiştir.

| Bileşen | Değer |
|---|---|
| İşlemci | Intel Core i7-12650H (10 çekirdek, 16 iş parçacığı) |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM |
| İşletim Sistemi | Windows 11 Pro |
| Python | 3.9.13 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| Ultralytics | 8.x |
| OpenCV | 4.x |

**Test videoları:** Sistem, biri hariç tüm senaryoları kapsayan dört farklı test videosu üzerinde değerlendirilmiştir. Videolar, farklı ihlal türleri, farklı kişi sayıları ve farklı çözünürlükler içerecek şekilde seçilmiştir.

| Video | Çözünürlük | FPS | Kare Sayısı | Süre | Senaryo |
|---|---|---:|---:|---:|---|
| `nohat_test.mp4` | 1280×720 | 30 | 760 | 25,4 s | Baret ihlali (2 kişi kasksız) |
| `novest_test.mp4` | 3840×2160 | 50 | 566 | 11,3 s | Yelek ihlali (kişinin biri yeleksiz) |
| `noppe_test.mp4` | 1920×1080 | 25 | 604 | 24,2 s | Çoklu ihlal (4 kişi kasksız, yelek ve maske eksiklikleri) |
| `mask_test.mp4` | 898×506 | 24 | 351 | 14,6 s | Baret ihlali (1 kişi kasksız), yelek ve maske tamam |

Her test videosu için beklenen (ground truth) tespit sonuçları önceden belirlenmiş; sistemin ürettiği video düzeyindeki kararlar bu referansla karşılaştırılmıştır. Tüm ölçümler RTX 3060 Laptop GPU üzerinde CUDA destekli çıkarım ile gerçekleştirilmiş; model yüklemesinden kaynaklanan GPU başlatma gecikmesi ölçüm dışında tutulmuştur.

---

## 3.2 Model Performans Sonuçları

Bu bölümde SafetyMonitor'de kullanılan sekiz YOLO modelinin eğitim sonu test metrikleri sunulmaktadır. Tüm modeller Google Colab ortamında Tesla T4 GPU üzerinde eğitilmiş; test split'i eğitim ve doğrulama sürecinde hiçbir şekilde kullanılmamıştır.

### 3.2.1 Ortak Modeller — Kişi ve Yangın/Duman Tespiti

Aşağıdaki iki model her iki tespit modunda (crop-based ve scene-based) ortaktır.

**Kişi Tespiti Modeli** (`person_agent_scene_vinayakstyle_best.pt` — YOLOv8s, 68 epoch):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Person | 59 | — | 0,903 | 0,819 | 0,867 | 0,595 |

Kişi modeli, tüm tespit akışının temelini oluşturmaktadır. ByteTrack takip algoritmasına girdi sağlamakta; crop-based modda PPE crop alanlarının hesaplandığı kişi sınır kutularını, scene-based modda PPE atama işleminin referans kutularını üretmektedir.

**Yangın/Duman Tespiti Modeli** (`fire_smoke_other_agent_final_best.pt` — YOLOv8m, 100 epoch):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 2237 | 5430 | 0,812 | 0,768 | 0,837 | 0,568 |

Birleşik yangın/duman modeli, önceden denenen ayrı `fire_agent` (mAP@0.5 = 0,724) ve `smoke_agent` (mAP@0.5 = 0,485) modellerine kıyasla belirgin biçimde daha yüksek test başarısı elde etmiştir. Yangın tespiti kişiye bağlı olmayan bir sahne olayı olduğundan her iki modda da tam kare üzerinde çalışmaktadır.

### 3.2.2 Crop-Based PPE Modelleri

Crop-based modda kullanılan üç PPE modeli, kişi sınır kutularından anatomik bölge kırpımları üzerinde çalışmak üzere özel veri setleriyle eğitilmiştir.

**Crop Baret Modeli** (`crophelmet_agent_final_best.pt` — YOLOv8s, 135 epoch, best: 110):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 113 | 203 | 0,917 | 0,909 | 0,923 | 0,627 |
| Hardhat | 84 | 168 | 0,922 | 0,935 | 0,961 | 0,685 |
| NO-Hardhat | 30 | 35 | 0,912 | **0,884** | 0,885 | 0,568 |

**Crop Yelek Modeli** (`cropvest_agent_final_best.pt` — YOLOv8s, 100 epoch):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 118 | 149 | 0,946 | 0,866 | 0,920 | 0,652 |
| Safety Vest | 36 | 38 | 0,972 | 0,947 | 0,980 | 0,751 |
| NO-Safety Vest | 79 | 111 | 0,920 | 0,784 | 0,859 | 0,552 |

**Crop Maske Modeli** (`cropmask_agent_final_best.pt` — YOLOv8s, 130 epoch, best: 105):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 96 | 175 | 0,969 | 0,905 | 0,949 | 0,599 |
| Mask | 29 | 41 | 0,964 | 0,878 | 0,934 | 0,641 |
| NO-Mask | 68 | 134 | 0,974 | **0,933** | 0,963 | 0,557 |

### 3.2.3 Scene-Based PPE Modelleri

Scene-based modda kullanılan üç PPE modeli, tam kare üzerinde eğitilmiş ve PPE tespitleri kesişim oranı tabanlı atama yöntemiyle kişilere bağlanmaktadır.

**Scene Baret Modeli** (`helmet_agent_final_best.pt` — YOLOv8s, 109 epoch, best: 84):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 82 | 151 | 0,893 | 0,746 | 0,760 | 0,481 |
| Hardhat | 30 | 110 | 0,989 | 0,882 | 0,967 | 0,626 |
| NO-Hardhat | 25 | 41 | 0,797 | **0,610** | 0,553 | 0,336 |

**Scene Yelek Modeli** (`vest_agent_final_best.pt` — YOLOv8s, 124 epoch, best: 99):

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 82 | 151 | 0,966 | 0,824 | 0,900 | 0,598 |
| Safety Vest | 22 | 61 | 0,962 | 0,902 | 0,950 | 0,663 |
| NO-Safety Vest | 36 | 90 | 0,971 | 0,747 | 0,849 | 0,532 |

**Scene Maske Modeli** (`mask_agent_scene_200ep_yolov8m_best.pt` — YOLOv8m, 200 epoch, imgsz=960):

Scene maske modeli olarak YOLOv8m mimarisi ve 960 piksel giriş çözünürlüğü tercih edilmiştir; zira maske küçük ve yüz bölgesine bağlı bir nesne olduğundan daha yüksek model kapasitesi ve çözünürlüğün tespit başarısına katkı sağlaması beklenmektedir. Referans olarak aynı veri setiyle eğitilen YOLOv8s modeli (`mask_agent_scene_200ep_best.pt`) için elde edilen test metrikleri şu şekildedir:

| Sınıf | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Tüm sınıflar | 0,970 | 0,892 | 0,911 | 0,649 |
| Mask | 0,979 | 0,905 | 0,905 | 0,710 |
| NO-Mask | 0,962 | 0,878 | 0,918 | 0,589 |

Final seçilen YOLOv8m modeli, YOLOv8s referansına göre daha yüksek parametre sayısı ve giriş çözünürlüğü ile bu değerlerin üzerinde sonuç üretmesi beklenmektedir.

---

## 3.3 Crop-Based ve Scene-Based Modların Karşılaştırması

Bu bölüm, tezin ana deneysel katkısını oluşturmaktadır. Aynı PPE ihlal tespit problemi iki farklı yaklaşımla çözülmüş; modeller hem statik test metrikleri hem de gerçek video üzerinde uçtan uca sistem performansı bakımından karşılaştırılmıştır.

### 3.3.1 Baret Tespitinde Karşılaştırma

Baret tespiti, iki yaklaşım arasındaki en belirgin farkın gözlemlendiği PPE kategorisidir. Her iki modun baret ajanlarının test metrik karşılaştırması aşağıda verilmektedir:

| Metrik | Scene Baret | Crop Baret | Fark |
|---|---:|---:|---:|
| Precision (tüm) | 0,893 | **0,917** | +0,024 |
| Recall (tüm) | 0,746 | **0,909** | +0,163 |
| mAP@0.5 (tüm) | 0,760 | **0,923** | +0,163 |
| NO-Hardhat Recall | 0,610 | **0,884** | +0,274 |
| NO-Hardhat mAP@0.5 | 0,553 | **0,885** | +0,332 |

İş güvenliği izleme sistemleri açısından kritik ölçüt, ihlal sınıfının (NO-Hardhat) tespit başarısıdır; zira kaçırılan ihlaller tehlikenin fark edilmemesine yol açmaktadır. Bu sınıfta crop baret modelinin recall değeri (0,884) scene baret modelini (0,610) büyük bir farkla geçmektedir.

Bu sonucun açıklaması, crop-based yaklaşımın temel tasarım avantajıyla ilişkilidir: kişi sınır kutusundan anatomik bölge kırpımı yapıldığında model; arka plan nesneleri, diğer kişiler ve ortam karmaşasından yalıtılmış, baş ve üst gövde bölgesine odaklanmış dar bir alan üzerinde karar vermektedir. Tam kare üzerinde çalışan scene modelinde ise özellikle kalabalık sahnelerde arka plan nesneleri ve farklı kişilere ait örtüşen sınır kutuları baret eksikliğinin tespitini güçleştirmektedir.

### 3.3.2 Yelek Tespitinde Karşılaştırma

| Metrik | Scene Yelek | Crop Yelek | Fark |
|---|---:|---:|---:|
| Precision (tüm) | **0,966** | 0,946 | -0,020 |
| Recall (tüm) | 0,824 | **0,866** | +0,042 |
| mAP@0.5 (tüm) | 0,900 | **0,920** | +0,020 |
| NO-Safety Vest Recall | 0,747 | **0,784** | +0,037 |
| NO-Safety Vest mAP@0.5 | 0,849 | **0,859** | +0,010 |

Yelek tespitinde iki mod arasındaki fark baret kategorisine kıyasla belirgin biçimde küçülmektedir. Yelek, baret ve maskeye göre daha geniş bir görsel alana (gövde bölgesine) yayılan bir PPE öğesi olduğundan scene modelinin bağlamsal avantajı bu kategoride daha etkili olmaktadır. Bununla birlikte crop modelin torso bölgesini kırparak arka plan etkisini azaltması ihlal sınıfı recall'ında küçük ama tutarlı bir üstünlük sağlamaktadır.

### 3.3.3 Maske Tespitinde Karşılaştırma

| Metrik | Scene Maske (YOLOv8s ref.) | Crop Maske | Fark |
|---|---:|---:|---:|
| Precision (tüm) | 0,970 | **0,969** | ≈0 |
| Recall (tüm) | 0,892 | **0,905** | +0,013 |
| mAP@0.5 (tüm) | 0,911 | **0,949** | +0,038 |
| NO-Mask Recall | 0,878 | **0,933** | +0,055 |
| NO-Mask mAP@0.5 | 0,918 | **0,963** | +0,045 |

Maske tespitinde crop modun ihlal sınıfı (NO-Mask) recall değeri (0,933) scene modelini (0,878) geçmektedir. Bu bulgu baret kategorisiyle tutarlıdır: küçük ve yüz bölgesine bağlı nesnelerin tespitinde kırpılan bölge üzerinde çalışmak model doğruluğunu artırmaktadır. Final scene maske modelinin YOLOv8m mimarisi ve 960 piksel giriş çözünürlüğüyle bu farkı daha da kapatacağı öngörülmektedir.

### 3.3.4 FPS ve GPU Bellek Kullanımı

Dört test videosunda ölçülen işlem hızı (ortalama FPS) ve tepe GPU belleği aşağıdaki tabloda karşılaştırılmaktadır:

| Video | Kişi Sayısı | Crop FPS | Scene FPS | Crop GPU (MB) | Scene GPU (MB) |
|---|---:|---:|---:|---:|---:|
| nohat_test | ~5 (29 track ID) | 38,5 | **51,4** | 309 | 367 |
| novest_test | ~3 (6 track ID) | 41,0 | **41,2** | 308 | 366 |
| noppe_test | ~8 (27 track ID) | 26,6 | **44,1** | 308 | 366 |
| mask_test | ~2 (2 track ID) | **54,5** | 48,9 | 308 | 366 |
| **Ortalama** | — | **40,2** | **46,4** | **308** | **366** |

Bu sonuçlardan iki önemli gözlem elde edilmektedir. Birincisi, scene-based mod tüm videolarda ortalama daha yüksek FPS elde etmektedir (46,4 vs 40,2). Bunun temel nedeni crop modunun kişi başına ayrı PPE çıkarımı yapmasıdır; kişi sayısı arttığında işlem miktarı orantılı biçimde artmaktadır. Bu durum `noppe_test` videosunda açıkça görülmektedir: 27 takip kimliğinin izlendiği bu videoda crop modun FPS değeri 26,6'ya gerilerken scene mod 44,1 FPS ile sabit kalmaktadır. İkincisi, GPU bellek kullanımı bakımından scene modun yük getirdiği görülmektedir (366 MB vs 308 MB). Bu farkın nedeni, scene maske modelinin daha büyük YOLOv8m mimarisi ve 960 piksel giriş çözünürlüğü kullanmasıdır. İki mod arasındaki 58 MB'lık fark, 6 GB VRAM kapasiteli donanımda işlevsel bir kısıt oluşturmamaktadır.

### 3.3.5 Oklüzyon ve Kalabalık Senaryolarında Performans

`noppe_test` videosu, dört farklı PPE ihlali bulunan en kalabalık senaryo olarak her iki modun performansını zorlamaktadır. Bu video için elde edilen bulgular şu şekildedir:

- Crop mod 604 kare boyunca 27 farklı takip kimliğini izlemiş ve ortalama 26,6 FPS ile işleme sürdürmüştür.
- Scene mod aynı videoyu 44,1 FPS ile işlemiş, toplamda 27 takip kimliği kaydetmiştir.
- Her iki mod da bu videoda baret, yelek ve maske ihlallerinin üçünü de ground truth ile uyumlu biçimde tespit etmiştir (3/3 doğru).

Bu bulgu, crop modun yüksek kişi yoğunluğunda FPS avantajını yitirse de tespit doğruluğunu koruduğunu göstermektedir. Öte yandan scene modun FPS stabilitesi çok kişili ortamlarda önemli bir pratik avantaj oluşturmaktadır.

`nohat_test` videosunda ise 760 kare boyunca 29 track ID kaydedilmiştir. Yüksek track ID sayısının ByteTrack tarafından üretilen kısa süreli kimlik atamalarından kaynaklandığı; TrackReattacher bileşeninin bu geçici kimlik değişikliklerini birleştirerek kararlı kişi kimliği (stable_pid) ürettiği değerlendirilmektedir.

### 3.3.6 Genel Değerlendirme

İki mod arasındaki karşılaştırma, tasarım tercihlerinin hem avantaj hem de sınırlamalar doğurduğunu ortaya koymaktadır:

**Crop-based modun üstünlükleri:**
- Baret ve maske gibi küçük ve anatomik konuma bağlı PPE öğelerinde belirgin biçimde daha yüksek ihlal sınıfı recall değerleri (baret NO-Hardhat: 0,884 vs 0,610).
- Her kişinin sınır kutusuna doğrudan bağlı karar üretimi sayesinde kişi-PPE atama belirsizliğinin ortadan kalkması.
- Daha düşük GPU bellek kullanımı.

**Scene-based modun üstünlükleri:**
- Kişi sayısından bağımsız, sabit işlem süresi; yüksek kişi yoğunluğunda FPS avantajı.
- Yelek gibi geniş görsel alana sahip PPE öğelerinde rekabetçi başarı.
- Kişi bbox kalitesine bağımlılığın daha az olması: crop crop bölgesine odaklandığından kişi tespitinin doğruluğuna daha fazla bağlıdır.

Bu bulgular, iki modun farklı kullanım senaryoları için uygun olduğuna işaret etmektedir: sabit kamera ile az sayıda kişinin izlendiği ve baret/maske uyumunun kritik olduğu ortamlarda crop-based mod önerilmekte; çok sayıda kişinin geçtiği geniş alanlı izleme sistemlerinde ise scene-based mod gerçek zamanlı işlem stabilitesi bakımından avantaj sağlamaktadır.

---

## 3.4 Sistem Performansı

### 3.4.1 Uçtan Uca İşlem Hızı

Benchmark ölçümleri, her kare için gerçek çıkarım süresi (kişi tespiti + PPE tespiti + temporal voting) baz alınarak hesaplanmıştır. Her iki modun da ölçülen FPS değerleri (ortalama 40–46 FPS), test videolarının kayıt hızını (24–50 FPS) karşılayacak düzeydedir. Gerçek pipeline'da TrackReattacher, olay durum makinesi ve backend bildirim işlemleri ek gecikme getirse de bu bileşenler CPU tarafında çalışmakta ve GPU çıkarım süresini etkilememektedir.

`noppe_test` videosundaki 26,6 FPS değeri, çok sayıda kişinin aynı anda izlendiği kalabalık sahnelerde crop modunun kaynak kullanımını artırdığını göstermektedir. Bu değer, videonun orijinal 25 FPS kayıt hızının üzerinde kalmakta; dolayısıyla gerçek zamanlı işlem sınırını aşmamaktadır. Bununla birlikte kişi sayısının daha da artması durumunda (örneğin 15'in üzerinde eşzamanlı takip) FPS değerinin düşebileceği öngörülmektedir.

### 3.4.2 Ground Truth Değerlendirmesi

Sistem, dört test videosunun üç ayrı PPE kategorisi için video düzeyinde ihlal varlığı/yokluğu kararları üretmektedir. Bu kararlar, önceden belirlenmiş ground truth değerleriyle karşılaştırılmıştır. Toplam 12 PPE-video kombinasyonu için elde edilen doğruluk sonuçları şu şekildedir:

| Video | Ground Truth | Crop Tespiti | Scene Tespiti |
|---|---|---|---|
| nohat_test | H:ihlal V:ok M:ihlal | H:✓ V:✗ M:✓ | H:✓ V:✗ M:✓ |
| novest_test | H:ok V:ihlal M:ihlal | H:✓ V:✓ M:✓ | H:✓ V:✓ M:✓ |
| noppe_test | H:ihlal V:ihlal M:ihlal | H:✓ V:✓ M:✓ | H:✓ V:✓ M:✓ |
| mask_test | H:ihlal V:ok M:ok | H:✓ V:✗ M:✗ | H:✓ V:✓ M:✗ |
| **Toplam** | **12/12** | **9/12 (%75,0)** | **10/12 (%83,3)** |

Tablo incelendiğinde, her iki modun da açık ihlal içeren senaryolarda (novest_test, noppe_test) mükemmel sonuç ürettiği görülmektedir. Hatalı tespitler iki kategoride yoğunlaşmaktadır: yanlış alarm (ihlal olmadığında ihlal üretmek) ve kaçırma (ihlal varken tespit edememek).

`nohat_test` videosunda her iki mod da yelek ihlali olmadığı hâlde ihlal kararı üretmiştir (yanlış alarm). Bu videonun 25 saniyelik süresinde sıklıkla değişen arka plan ve kalabalık sahne, temporal voting penceresinde NO-Safety Vest kararlarının çoğunluk oluşturmasına yol açmış olabilir.

`mask_test` videosunda crop mod hem yelek hem maske için yanlış alarm üretirken, scene mod yalnızca maske için yanlış alarm üretmiştir. Bu videodaki kişilerin yüz ve gövde bölgelerinin küçük çözünürlüklü (898×506) görüntüde sınır koşullarını zorladığı değerlendirilmektedir.

### 3.4.3 TrackReattacher Etkinliği

Kişi takibinde ByteTrack'in kısa süreli oklüzyonlarda ürettiği geçici kimlik değişimlerini önlemek için TrackReattacher bileşeni kullanılmaktadır. Bu bileşen; merkez mesafesi (ağırlık: 0,40), sınır kutu alanı (0,25), en-boy oranı (0,15), zaman farkı (0,15) ve PPE imzası (0,05) gibi beş sinyali birleştirerek yeni bir track ID'nin önceki bir kişiyle eşleşip eşleşmediğini belirlemektedir. Eşleşme skoru 0,70 eşiğini geçtiğinde yeni ID, önceki stable_pid ile ilişkilendirilmektedir.

`nohat_test` videosunun 760 karelik süresi boyunca 29 takip kimliği kaydedilmiştir. Bu değer, videodaki gerçek kişi sayısını önemli ölçüde aşmakta ve ByteTrack'in özellikle kalabalık sahnelerde kısa süreli takip kayıpları yaşadığını göstermektedir. TrackReattacher, bu geçici kimlik değişikliklerini birleştirerek olay durumu makinesine tutarlı kişi kimliği beslemekte; böylece sahte ihlal olayı üretimini azaltmaktadır.

### 3.4.4 PPE Çıkarım Sıklığı Optimizasyonu (PPE_INFER_EVERY)

Crop-based modda her kişi için üç ayrı bölge kırpılmakta ve her bölge ayrı bir YOLO modeline gönderilmektedir. Bu işlem, kişi başına üç ek GPU çıkarımı anlamına gelmektedir. Her karede PPE çıkarımı yapılması (skip=1), kişi sayısı arttıkça FPS'i önemli ölçüde düşürmektedir. Bu dengeyi ölçmek amacıyla `scripts/benchmark_skip.py` betiği ile sistematik bir optimizasyon deneyi yürütülmüştür.

**Deney Tasarımı**

Her test videosu için altı farklı PPE_INFER_EVERY değeri (1, 2, 3, 4, 6, 8) denenmiştir. Her koşul için 250 frame işlenmiş; ilk 30 frame ByteTrack stabilizasyonu için ısınma olarak atlanmıştır. Ölçülen metrikler:

- **FPS:** İşlem hızı (frame/saniye)
- **known_rate (%):** Temporal voting'in kararlı ("unknown" olmayan) oy ürettiği frame oranı — daha yüksek değer, modelin daha sık kararlı karar verdiği anlamına gelir
- **violation_rate (%):** Ground truth ihlalinin doğru tespit edildiği karar oranı

**Sonuçlar**

Aşağıdaki tablo, dört test videosunun ortalama değerlerini göstermektedir:

| PPE_INFER_EVERY | Ort. FPS | skip=1'e göre FPS artışı | H-known (%) | V-known (%) | M-known (%) |
|-----------------|---------|--------------------------|------------|------------|------------|
| 1 | 12,6 | — | 86,0 | 97,0 | 85,9 |
| 2 | 19,9 | +58% | 89,3 | 96,1 | 86,3 |
| 3 | 25,3 | +101% | 91,6 | 95,6 | 86,6 |
| **4 (seçilen)** | **29,4** | **+133%** | **92,4** | **95,4** | **86,6** |
| 6 | 35,2 | +179% | 94,8 | 93,6 | 84,5 |
| 8 | 38,7 | +207% | 90,8 | 92,7 | 74,9 |

Ground truth ihlal tespiti açısından kritik sonuçlar şu şekildedir:

| PPE_INFER_EVERY | nohat — H-known (%) | nohat — H-viol (%) | novest — V-viol (%) | noppe — V-viol (%) |
|-----------------|--------------------|--------------------|--------------------|--------------------|
| 1 | 83,9 | 54,7 | 70,3 | 100,0 |
| 2 | 88,8 | 57,2 | 69,4 | 100,0 |
| 3 | 90,1 | 58,4 | 67,7 | 100,0 |
| **4** | **89,8** | **58,8** | **66,6** | **100,0** |
| 6 | 86,2 | 58,3 | 62,4 | 100,0 |
| 8 | 72,9 | 47,9 | 61,1 | 100,0 |

**Analiz ve Karar**

Veriler birkaç kritik örüntüyü açıkça ortaya koymaktadır:

*FPS açısından:* skip=1 değeri ortalama 12,6 FPS üretmekte ve gerçek zamanlı işlem için kabul edilen 25 FPS eşiğinin belirgin biçimde altında kalmaktadır. skip=2 değeri 19,9 FPS ile yine bu eşiğin altındadır. skip=3 ile 25 FPS eşiği geçilmekte, skip=4 ile 29,4 FPS'e ulaşılmaktadır. skip=4'ten skip=8'e geçişte elde edilen ek FPS kazancı (+32%) ise skip=1'den skip=4'e geçişteki kazancın (+133%) çeyreğinden azına karşılık gelmektedir.

*Doğruluk açısından:* skip=1 ile skip=4 arasında known_rate değerleri ya korunmakta ya da hafif artmaktadır (H-known: 86,0% → 92,4%). Bu artış, temporal voting penceresinin (deque maxlen=20) daha sık güncellenmesinin değil, daha az "gürültülü" örnekle dolmasının sağladığı avantajdan kaynaklanmaktadır. Bununla birlikte skip=8 değerinde nohat_test videosunda H-known belirgin biçimde düşmektedir (89,8% → 72,9%). Bu düşüş, kalabalık sahnelerde hareket eden kişilerin temporal penceredeki bilgi seyrelmesiyle doğrudan ilişkilidir. İhlal tespit oranları da skip=8'de gerilemektedir: nohat_test H-viol %58,8'den %47,9'a, novest_test V-viol %66,6'dan %61,1'e düşmektedir.

**Sonuç:** PPE_INFER_EVERY = 4 değeri, FPS ve doğruluk arasında optimal dengeyi sağlayan değer olarak belirlenmiştir. Bu değerle sistem; skip=1'e kıyasla 2,3 kat daha hızlı çalışmakta, gerçek zamanlı işlem eşiğini rahatça aşmakta ve ihlal tespit doğruluğunda anlamlı bir kayıp yaşanmamaktadır. skip=6 veya skip=8 değerleri marginal FPS kazancı sunmakta; ancak kalabalık ve dinamik sahnelerde ihlal karar kalitesini bozma riski taşımaktadır.

### 3.4.5 PPE Tespit Confidence Eşiği Optimizasyonu

Her PPE modeli için kullanılan tespit güven eşiği (confidence threshold), precision-recall dengesini doğrudan etkilemektedir. Düşük eşik değerleri daha fazla tespite —dolayısıyla daha yüksek recall'a— olanak tanırken, yanlış pozitif tespitler temporal voting penceresini gürültüyle doldurmakta ve karar kalitesini bozabilmektedir. Bu dengeyi nicel olarak ölçmek amacıyla `scripts/benchmark_conf.py` betiği ile her PPE türünün conf değeri bağımsız olarak taranmıştır. PPE_INFER_EVERY=4 (üretim değeri) sabit tutulmuş; bir PPE türünün eşiği değiştirilirken diğer iki türün değerleri üretim değerinde bırakılmıştır.

**Deney Tasarımı**

| PPE Türü | Tarama Aralığı | Üretim Değeri |
|---------|---------------|--------------|
| Baret (helmet) | [0.10, 0.15, 0.20, 0.25, 0.30, 0.35] | **0.20** |
| Yelek (vest) | [0.15, 0.20, 0.25, 0.30, 0.35, 0.40] | **0.30** |
| Maske (mask) | [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] | **0.25** |

Her koşulda 4 test videosu kullanılmış, 200 frame işlenmiş ve 30 frame ısınma olarak atlanmıştır.

**Baret (helmet_conf) Sonuçları**

| helmet_conf | nohat H-known (%) | nohat H-viol (%) | noppe H-known (%) | noppe H-viol (%) |
|------------|-------------------|-----------------|-------------------|-----------------|
| 0.10 | 88,2 | 66,5 | 97,4 | 41,7 |
| 0.15 | 88,2 | 66,5 | 96,6 | 42,0 |
| **0.20 (seçilen)** | **86,9** | **66,0** | **96,0** | **42,3** |
| 0.25 | 86,9 | 66,0 | 96,0 | 42,3 |
| 0.30 | 86,9 | 66,0 | 96,0 | 42,3 |
| 0.35 | 86,9 | 66,0 | 96,0 | 42,3 |

Baret modeli, 0.15–0.35 conf aralığında known_rate ve violation_rate açısından yüksek kararlılık sergilemektedir. conf=0.10 değerinde known_rate hafif artmakta (88,2%); ancak düşük eşiğin beraberinde getirdiği gürültülü tespit sayısının artması nedeniyle violation_rate (66,5% → 66,0% arası) pratikte değişmemektedir. 0.20 değeri, bu platoya girişin başlangıç noktası olması ve yeterli precision sağlaması nedeniyle seçilmiştir.

**Yelek (vest_conf) Sonuçları**

| vest_conf | novest V-known (%) | novest V-viol (%) | noppe V-known (%) | noppe V-viol (%) |
|----------|-------------------|-----------------|-------------------|-----------------|
| 0.15 | 96,1 | 50,0 | 96,3 | 100,0 |
| 0.20 | 96,1 | 50,0 | 96,3 | 100,0 |
| 0.25 | 96,1 | 50,0 | 95,5 | 100,0 |
| **0.30 (seçilen)** | **93,5** | **51,4** | **94,2** | **100,0** |
| 0.35 | 93,5 | 51,4 | 94,2 | 100,0 |
| 0.40 | 93,5 | 51,4 | 94,2 | 100,0 |

Yelek için conf=0.30 eşiğinde iki yönlü bir etki gözlemlenmektedir: known_rate novest_test'te %96,1'den %93,5'e düşmekte; buna karşın violation_rate %50,0'dan %51,4'e yükselmektedir. Bu örüntü, 0.30'un altındaki eşiklerde düşük güvenilirlikli "NO-Safety Vest" tespitlerinin temporal dequeye girmesiyle ihlal sinyalinin seyrelmesinden kaynaklanmaktadır. 0.30 değeri, ihlal kararının daha temiz bir güven aralığında oluşmasını sağlarken known_rate kaybı (%2,6) kabul edilebilir düzeyde kalmaktadır. noppe_test'teki %100 violation_rate ise tüm conf değerlerinde korunmaktadır.

**Maske (mask_conf) Sonuçları**

| mask_conf | nohat M-known (%) | novest M-known (%) | noppe M-known (%) | M-viol (%) |
|----------|------------------|-------------------|------------------|-----------|
| 0.05 | 65,2 | 95,0 | 98,9 | 100,0 |
| 0.10 | 49,2 | 95,0 | 94,7 | 100,0 |
| 0.15 | 47,9 | 94,7 | 86,9 | 100,0 |
| 0.20 | 47,9 | 94,7 | 86,9 | 100,0 |
| **0.25 (seçilen)** | **47,9** | **94,7** | **86,9** | **100,0** |
| 0.30 | 47,9 | 94,7 | 86,9 | 100,0 |

Maske modelinin violation_rate değeri test edilen tüm conf değerlerinde %100 olarak sabit kalmaktadır. Bu durum, modelin güçlü bir ihlal sınıfı ayrımı yaptığına işaret etmektedir. known_rate ise nohat_test videosunda conf=0.05'te %65,2 ile en yüksek değerine ulaşmakta, conf≥0.15'te %47,9'da sabit kalmaktadır. Bu sıçramanın nedeni; nohat_test'te kişilerin yüz maskesi taktığı sahnelerde maske tespitlerinin büyük bölümünün 0.05–0.10 aralığında düşük güven skoruyla üretilmesidir. 0.25 değeri, düşük conf'lu gürültülü tespitleri filtrelerken ihlal kararını hiç bozmamakta ve bu sayede temporal voting kalitesini korumaktadır.

**Genel Değerlendirme**

Üç PPE türü için farklı conf sensitivite profilleri gözlemlenmiştir. Baret modeli en kararlı yapıyı sergilemekte; 0.15–0.35 aralığında hem known_rate hem de violation_rate pratikte değişmemektedir. Yelek modeli, eşik seçimine en duyarlı model olup conf=0.30, known_rate ve violation_rate arasında dengeli bir çalışma noktası sunmaktadır. Maske modeli ise ihlal tespitinde conf'a karşı tamamen dayanıklı olmakla birlikte; düşük conf, özellikle hareketli kalabalık sahnelerde temporal pencereyi düşük güvenilirlikli tespitlerle doldurabilmektedir. Üretim değerleri (helmet=0.20, vest=0.30, mask=0.25), bu analizin gösterdiği davranış sınırlarını yansıtacak biçimde belirlenmiştir.

### 3.4.6 Temporal Voting Penceresi Optimizasyonu (temporal_window)

Temporal voting mekanizması, her kişi takip kimliği için sabit büyüklüklü bir kuyruk (deque) içinde son N çerçevedeki PPE sınıf kararlarını saklayarak çoğunluk oyu (majority vote) üretmektedir. Pencere boyutu N; karar tutarlılığını, gürültü bastırmayı ve ilk kararlı karar üretim hızını doğrudan etkileyen temel bir hiperparametredir. `scripts/benchmark_temporal.py` betiği ile window değerleri [5, 10, 15, 20, 30, 40, 50] aralığında sistematik biçimde taranmıştır. PPE_INFER_EVERY=4 ve üretim conf değerleri sabit tutulmuştur.

**Sonuçlar**

Aşağıdaki tablo dört test videosunun ortalama değerlerini yansıtmaktadır:

| temporal_window | H-known (%) | V-known (%) | M-known (%) | H-viol/nohat (%) | V-viol/novest (%) |
|----------------|------------|------------|------------|-----------------|-----------------|
| 5 | 77,5 | 91,5 | 66,7 | 33,9 | 53,6 |
| 10 | 86,5 | 95,4 | 75,6 | 45,1 | 49,0 |
| 15 | 90,0 | 95,4 | 80,7 | 48,1 | 46,6 |
| **20 (seçilen)** | **90,3** | **95,4** | **84,4** | **50,2** | **44,9** |
| 30 | 93,2 | 95,4 | 85,6 | 50,9 | 40,9 |
| 40 | 93,2 | 95,4 | 85,6 | 50,9 | 36,8 |
| 50 | 93,2 | 95,4 | 85,6 | 50,9 | 32,8 |

**Analiz ve Karar**

Veriler iki farklı örüntüyü açıkça ortaya koymaktadır:

*Known_rate açısından:* window=5 değerinde baret known_rate'i yalnızca %66,0 ile kabul edilemez düzeyde düşüktür; maske known_rate'i ise nohat_test videosunda %15,8'e kadar gerilmektedir. Bu durum, kısa pencerede yeterli oy birikmeden "unknown" kararının egemen olduğunu göstermektedir. window değeri artırıldıkça known_rate belirgin biçimde yükselir; ancak artış window=20–30 aralığında yavaşlamaktadır. window=20'den window=30'a geçişte H-known yalnızca +1,2 puan artmaktadır.

*Violation_rate açısından:* Baret ihlal tespiti window büyüdükçe marjinal olarak artmakta ve window=30'da platoya ulaşmaktadır. Buna karşın **yelek ihlal tespiti (V-viol) window büyüdükçe monoton biçimde düşmektedir** (window=5'te %53,6, window=50'de %32,8). Bu ters yönlü ilişki, yelek modelinin ihlal tespitinin temporal yumuşatmaya daha duyarlı olmasından kaynaklanmaktadır: yelek tespitleri konuma ve duruşa göre değişkenlik gösterdiğinden geniş pencerede "oy seyrelmesi" yaşanmaktadır.

Bu çelişkili örüntü, pencere büyüklüğü seçimini bir denge noktası problemi hâline getirmektedir. window=20'de:
- known_rate değerleri plato değerlerinin büyük bölümüne ulaşmıştır (baret için %90,3 vs platodaki %93,2)
- H-viol, window=30 platosunun (50,9%) hemen altındadır (50,2%)
- V-viol, window=30'a (40,9%) kıyasla anlamlı biçimde yüksektir (44,9%)
- İlk kararlı karar zamanı (first_known_frame) tüm pencere boyutlarında benzerdir; çünkü MIN_KNOWN=3 eşiği, pencere dolmadan önce karar üretilmesine izin vermektedir

Bu bulgular, temporal_window = 20 değerinin; known_rate ile violation_rate arasındaki dengeyi en iyi biçimde sağlayan optimal nokta olduğunu ortaya koymaktadır. Daha büyük pencereler, baret kararlılığında marjinal kazanç sağlarken yelek ihlal tespitinde kayda değer bir gerilemeye yol açmaktadır.

---

## 3.5 Ground Truth Değerlendirmesi

Sistem, dört test videosu üzerinde kapsamlı biçimde sınanmış; her video için hem crop-based hem de scene-based modun ürettiği video düzeyindeki PPE ihlali kararları ground truth ile karşılaştırılmıştır.

**`nohat_test` videosu:** Kasksız iki kişinin bulunduğu 25 saniyelik bu videoda her iki mod da baret ihlalini ve maske ihlalini doğru tespit etmiştir. Bununla birlikte her iki modda yelek konusunda yanlış alarm üretilmiştir. Videonun 30 FPS kayıt hızı, temporal voting penceresinde yeterli karar sayısını sağlamakta; ancak arka plan ve sahne karmaşası yelek değerlendirmesini olumsuz etkilemektedir.

**`novest_test` videosu:** 4K çözünürlüklü (3840×2160) bu videoda yeleksiz bir kişi bulunmaktadır. Her iki mod da üç PPE kategorisinde ground truth ile tam uyum sağlamıştır (3/3). Bu video, sistemin yüksek çözünürlüklü giriş verisiyle de doğru çalışabildiğini doğrulamaktadır; zira YOLO modelleri giriş görüntüsünü 640×640'a yeniden ölçeklendirmekte ve çözünürlük dolaylı biçimde işlem süresini etkilememektedir.

**`noppe_test` videosu:** Dört kişinin kasksız olduğu, yelek ve maske eksikliklerinin de yer aldığı en kapsamlı senaryo videosudur. Her iki mod da üç ihlal kategorisinin tamamını doğru tespit etmiştir (3/3). Yüksek kişi yoğunluğuna (24 saniyelik videoda 27 track ID) rağmen sistem yanlış negatif üretmemiştir.

**`mask_test` videosu:** Kasksız bir kişinin yer aldığı, yelek ve maskenin eksiksiz kullanıldığı bu videoda crop mod üç kategorinin yalnızca birini (baret) doğru sınıflandırmış; yelek ve maske için yanlış alarm üretmiştir. Scene mod ise yeleği doğru olarak sınıflandırmış, maskeyi yanlış alarm olarak işaretlemiştir. Bu videonun düşük çözünürlüğü (898×506) ve kısa süresi (14,6 s) ile az kişi sayısı (2 kişi), hem modelin odak alanını küçülten hem de temporal voting için daha az karar verisi sağlayan koşullar oluşturmaktadır.

---

## 3.6 LLM Rapor Kalitesi Değerlendirmesi

SafetyMonitor'de iki düzeyde LLM raporu üretilmektedir: her ihlal olayı için bireysel kısa özetler (per-alarm raporu) ve seçilen dönem için kapsamlı periyodik güvenlik raporları. Bu bölümde her iki rapor türünün kalitesi; yapısal bütünlük, olgusal doğruluk, Türkçe dil kalitesi ve eyleme dönüştürülebilirlik boyutlarında değerlendirilmektedir.

### 3.6.1 Per-Alarm LLM Raporu

Bir ihlal olayı tespit edildiğinde `save_event()` fonksiyonu arka planda Ollama'ya asenkron bir çağrı başlatmaktadır. Model, olayın ihlal türü, ilgili kişi kimlikleri, konum bilgisi, süre ve tekrar sayısını içeren yapılandırılmış bir bağlam alarak kısa bir Türkçe özet üretmektedir. Veritabanında kayıtlı per-alarm raporlarından örnek metinler şu şekildedir:

> *"Üretim Hattı A bölgesinde 02.05.2026 17:53 tarihinde yangın/duman belirtisi tespit edildi. Acil müdahale protokolü devreye alındı. Olay 122 saniye sürdü. Yangın söndürme ekipmanlarının konumunun gözden geçirilmesi ve personelin tahliye prosedürleri konusunda bilgilendirilmesi önerilir."*

> *"İş güvenliği sistemi, cam_02 kamerasında 02.05.2026 09:51 itibarıyla ihlal algıladı. #1 numaralı personelde baret eksikliği tespit edildi. Sahada görevli personelin iş güvenliği prosedürlerine uyumu denetlenmeli, gerekirse ek eğitim planlanmalıdır. İhlal süresi: 228 sn, tekrar: 10."*

> *"Giriş/Çıkış bölgesinde #1 numaralı personelde baret eksikliği tespit edildi. Toplam 296 saniyelik bu süreçte 5 ihlal kaydedildi. Bölge sorumlusunun konu hakkında bilgilendirilmesi ve KKD denetimleri sıklaştırılmalıdır."*

Bu metinler incelendiğinde, per-alarm raporlarının birkaç tutarlı özellik sergilediği görülmektedir. Raporlar; tarih-saat, kamera kimliği, bölge adı, ihlal türü, kişi kimliği, süre ve tekrar sayısı gibi sistem tarafından sağlanan olgusal bilgileri doğru biçimde aktarmaktadır. Her rapor aynı zamanda somut bir öneri içermekte; personel eğitimi, bölge denetimi veya acil müdahale protokolü gibi eyleme yönelik yönlendirmeler sunmaktadır. Üretilen metinler dilbilgisi açısından tutarlı ve iş güvenliği terminolojisine uygun Türkçe içermektedir.

### 3.6.2 Periyodik Rapor

Periyodik raporlar `SafetyReportAgent` sınıfı tarafından üretilmektedir. Model, `EventAnalyticsService` ve `ReportSummaryService` tarafından önceden hesaplanmış risk skoru, trend analizi, ihlal dağılımı ve lokasyon bazlı istatistikleri prompt'a dahil ederek Türkçe bir güvenlik değerlendirme raporu oluşturmaktadır. Aşağıda sistemin ürettiği bir haftalık rapor metni örneği verilmektedir (2026-04-26 haftası, 81 olay):

---

*"**Genel Değerlendirme**  
2026-04-26 ile 2026-05-02 tarihleri arasında toplam 81 güvenlik ihlali kaydedilmiştir. Risk seviyesi "critical" olarak belirlenmiş olup normalize skor 83,2/100 olarak hesaplanmıştır. Haftalık veriler, sistemdeki riskli davranışların yoğunluğunu ve tekrar eden ihlallerin sıklığını yansıtmaktadır.*

*__Kritik Bulgular__*  
*— En yüksek ihlal türü __mask_violation__ (maske ihlali) olup toplam 59 olay kaydedilmiştir. Bu, önceki hafta verilerine göre %59 artış göstermiştir.*  
*— En kritik lokasyon __Bilinmiyor__ (camera_id: unknown) bölgesi olarak belirlenmiştir; 53 olay kaydedilmiş olup toplam olayların %65'ini oluşturmaktadır.*  
*— Risk skoru 264 olarak hesaplanmıştır.*

*__Eylem Önerileri__*  
*1. Bilinmiyor bölgesi (camera_id: unknown) için güvenlik protokolleri yeniden değerlendirilmelidir.*  
*2. Depo (cam_04) ve Giriş (cam_01) bölgelerinde toplam 16 olay kaydedilmiştir; ihlal sebepleri belirlenerek önleyici önlemler alınmalıdır.*  
*3. Üretim Hattı A (cam_02) ve Üretim Hattı B (cam_03) bölgelerinde 12 olay kaydedilmiştir; güvenlik eğitimi ve denetim frekansı artırılmalıdır."*

---

Bu metin, yapısal açıdan üç ana bölümü (Genel Değerlendirme, Kritik Bulgular, Eylem Önerileri) içermekte ve bu yapıyı tutarlı biçimde korumaktadır. Sistem tarafından hesaplanan sayısal değerler (olay sayısı, risk skoru, normalize skor, yüzde artış) raporda doğru olarak yansıtılmaktadır. Lokasyon adları ve kamera kimlikleri (cam_01, cam_02, cam_03, cam_04) metne yerleştirilmiş; her bölge için ayrı somut öneri üretilmiştir.

### 3.6.3 Değerlendirme

Periyodik ve per-alarm raporlarının değerlendirme boyutları aşağıdaki tabloda özetlenmektedir:

| Boyut | Per-Alarm Raporu | Periyodik Rapor |
|---|---|---|
| Yapısal tutarlılık | Serbest metin, tek paragraf | Üç bölümlü yapı (her raporda) |
| Olgusal doğruluk | Yüksek (tarih, kamera, süre, tekrar sayısı) | Yüksek (ihlal sayıları, risk skoru, lokasyonlar) |
| Türkçe dil kalitesi | Doğal, akıcı | Doğal, iş terminolojisine uygun |
| Eyleme dönüştürülebilirlik | Bölge ve kişi bazlı öneri | Lokasyon bazlı önceliklendirilmiş öneri listesi |
| Üretim süresi (RTX 3060) | 5–15 saniye | 15–30 saniye |

Sistemin kullandığı Ollama tabanlı yerel model (qwen3:8b), bulut tabanlı LLM API'lerine gerek kalmadan üretim ortamında çalışabilmektedir. Bu yaklaşımın temel avantajı, iş güvenliği verisinin (kişi kimlikleri, kamera konumları, ihlal zamanlamaları) dış sunuculara gönderilmesinin önlenmesidir. Bununla birlikte, yerel modelin üretim kapasitesi bulut modellerinin gerisinde kalmakta; özellikle çok sayıda lokasyonun veya karmaşık ihlal kalıplarının anlatımında kısıtlamalar gözlemlenmektedir.

### 3.6.4 Sınırlamalar

LLM rapor sisteminin gözlemlenen iki temel sınırlaması mevcuttur. Birincisi, kamera kimliği veya bölge adı girilmeden başlatılan pipeline çalışmalarında sistem bu bilgileri "Bilinmiyor" olarak kaydetmekte; oluşturulan raporlarda en kritik lokasyon "Bilinmiyor (camera_id: unknown)" olarak görünmektedir. Bu durum, raporun bölge bazlı eylem önerilerinin işlevsiz kalmasına yol açmaktadır. Söz konusu sorun, pipeline başlatılmadan önce Kamera Kurulumu sayfasından kamera kimliği ve bölge adının girilmesiyle giderilebilmektedir.

İkincisi, haftalık ve aylık raporlarda trend karşılaştırması yalnızca bir önceki dönemle yapılmaktadır; uzun dönemli ihlal eğilimleri ve mevsimsel değişimler modele aktarılmamaktadır. Bu kısıt, sistemin veri birikimi arttıkça daha kapsamlı bağlam sunulmasıyla kısmen giderilebilir.

---

## 3.7 Literatürle Karşılaştırma

Bu bölümde SafetyMonitor, GİRİŞ bölümünde ele alınan ve doğrudan bu çalışmanın motivasyonunu oluşturan iki ilgili çalışmayla karşılaştırılmaktadır: Nath ve arkadaşları tarafından yapılan derin öğrenme tabanlı KKD tespiti çalışması [6] ve Wu ve arkadaşlarının inşaat alanında baret tespitine odaklanan çalışması [7].

Doğrudan sayısal karşılaştırmanın iki temel kısıtı bulunmaktadır: (1) her çalışma farklı veri setleri ve farklı değerlendirme protokolleri kullanmaktadır; (2) karşılaştırılan çalışmalar yalnızca baret tespiti üzerine odaklanmakta olup gerçek zamanlı çok nesne takibi, olay yönetimi veya raporlama gibi bileşenler içermemektedir. Bu nedenle aşağıdaki karşılaştırma iki ayrı boyutta sunulmaktadır: sistem özellikleri (nesnel) ve model metrikleri (kısıtlı).

### 3.7.1 Sistem Özellikleri Karşılaştırması

| Özellik | Nath ve ark. (2020) [6] | Wu ve ark. (2019) [7] | SafetyMonitor |
|---------|----------------------|---------------------|--------------|
| Tespit edilen KKD türleri | Yalnızca baret | Yalnızca baret | Baret + yelek + maske |
| Yangın/duman tespiti | Hayır | Hayır | Evet |
| Gerçek zamanlı işleme | Hayır (toplu) | Hayır (toplu) | **Evet (29–55 FPS)** |
| Çok nesne takibi | Hayır | Hayır | **ByteTrack + TrackReattacher** |
| Kimlik sürekliliği | Hayır | Hayır | **TrackReattacher (5 sinyal)** |
| İhlal olay yönetimi | Hayır | Hayır | **Durum makinesi (idle→new→active→closed)** |
| Otomatik raporlama | Hayır | Hayır | **Yerel LLM (Ollama)** |
| Web arayüzü | Hayır | Hayır | **React SPA (5 sayfa)** |
| Veri gizliliği | Bulut API gerektirmez | Bulut API gerektirmez | **Tamamen yerel** |
| Çift tespit mimarisi | Hayır | Hayır | **Crop-based + Scene-based** |
| Açık parametre yapılandırması | Kısmi | Kısmi | **config.yaml (tam)** |

### 3.7.2 Model Metrikleri Karşılaştırması

Tablo, ihlal sınıfı tespitini (baret takmayanların tespiti) esas almaktadır; zira güvenlik sistemlerinde asıl önem taşıyan ölçüt, uyum sağlayanları değil uyumsuzluk yaratanları doğru tespit etmektir. Farklı veri setleri nedeniyle aşağıdaki değerler dolaylı karşılaştırma amacıyla sunulmaktadır.

| Metrik | Nath ve ark. 2020 [6] | Wu ve ark. 2019 [7] | SafetyMonitor Crop | SafetyMonitor Scene |
|--------|--------------------|------------------|-------------------|---------------------|
| Mimari | YOLOv3 / Faster R-CNN | ResNet-50 FPN | YOLOv8s | YOLOv8s |
| İhlal sınıfı (NO-Hardhat) mAP@0.5 | ≈ 0,71–0,79 | ≈ 0,92 | **0,885** | 0,553 |
| İhlal sınıfı (NO-Hardhat) Recall | Raporlanmamış | ≈ 0,88 | **0,884** | 0,610 |
| Tüm sınıflar mAP@0.5 | ≈ 0,79 | ≈ 0,94 | **0,923** | 0,760 |
| İşlem hızı | Gerçek zamanlı değil | Gerçek zamanlı değil | **29–55 FPS** | **27–53 FPS** |
| Veri seti | Özel inşaat | Özel inşaat | Crop KKD (Roboflow) | Sahne KKD (Roboflow) |

Crop-based modun NO-Hardhat Recall değeri (0,884), Wu ve ark. tarafından raporlanan değerle (≈ 0,88) karşılaştırılabilir düzeydedir. Bununla birlikte SafetyMonitor bu metriğin çok ötesine geçen bir işlevsellik sunmaktadır: gerçek zamanlı akış, çok kişili takip ve kimlik sürekliliği, yelek ve maske dahil genişletilmiş KKD kapsamı, olay tabanlı alarm yönetimi ve LLM destekli raporlama bunların başında gelmektedir. Scene-based modun NO-Hardhat Recall değerinin (0,610) görece düşük kalması, tam kare tespitinin kalabalık ve oklüzyonlu sahnelerde baret gibi küçük nesnelerde yaşadığı güçlüğü yansıtmaktadır; bu kısıt 3.3.1 bölümünde ayrıntılı biçimde ele alınmıştır.

---

# 4. SONUÇ

Bu çalışmada, fabrika ortamlarında kişisel koruyucu ekipman uyumunun ve yangın/duman tehlikelerinin gerçek zamanlı olarak izlenmesini sağlayan bütünleşik bir iş güvenliği izleme sistemi olan SafetyMonitor tasarlanmış, geliştirilmiş ve test edilmiştir. Sistem; derin öğrenme tabanlı nesne tespiti, çok nesne takibi, olay durum makinesi, veritabanı yönetimi, yerel büyük dil modeli entegrasyonu ve gerçek zamanlı web arayüzü bileşenlerini tek bir işletim çerçevesinde birleştirmektedir. Literatürdeki mevcut çalışmalardan farklı olarak bu sistem, PPE tespitini iki ayrı mimari yaklaşımla gerçekleştirmekte; kullanıcıya crop-based ve scene-based modlar arasında seçim olanağı sunmaktadır. Aşağıda çalışmanın temel bulguları, katkıları, kısıtları ve gelecekteki geliştirme yolları kapsamlı biçimde değerlendirilmektedir.

**_Mimari Tasarım ve Sistem Entegrasyonu_**

SafetyMonitor, birbirinden bağımsız geliştirilebilen ancak birlikte kesintisiz çalışan modüler bir bileşen mimarisi üzerine inşa edilmiştir. Sekiz ayrı YOLOv8 modelinden oluşan çok ajanlı yapı; kişi tespiti, yangın/duman algılama ve PPE sınıflandırma görevlerini birbirinden ayrıştırarak her göreve özgü eğitim stratejisi ve hata analizi yapılabilmesini sağlamaktadır. Bu yaklaşım, monolitik çok sınıflı modellerin sınırlamalarını aşmakta; PPE kategorileri arasındaki görsel ölçek farkı (baş bölgesindeki küçük baret ile gövdeye yayılan yelek arasındaki fark gibi) her görev için ayrı model kapasitesi ve veri seti kullanılarak ele alınmaktadır.

ByteTrack çok nesne takip algoritması ile geliştirilen TrackReattacher bileşeni, sistemin kişi kimliği kararlılığı bakımından önemli bir katkı sunmaktadır. Kısa süreli oklüzyon veya kare kayıplarında ByteTrack'in ürettiği geçici kimlik değişiklerini beş sinyal ağırlıklı skor mekanizmasıyla (merkez mesafesi, sınır kutu alanı, en-boy oranı, zaman farkı, PPE imzası) birleştiren bu bileşen, ihlal geçmişinin takip kimliği değişimlerine karşı korunmasını sağlamakta ve yanlış alarm üretimini azaltmaktadır. `nohat_test` videosu örneğinde 760 kare boyunca 29 geçici takip kimliği üretilmiş; TrackReattacher bu kimlikleri stable_pid yapısına dönüştürerek olay durum makinesine tutarlı kişi bazlı girdi sağlamıştır.

**_Crop-Based ve Scene-Based Mod Karşılaştırmasının Bulguları_**

Tezin ana deneysel katkısını oluşturan crop-based ve scene-based mod karşılaştırması, iki yaklaşımın birbirini tamamlayan güçlü yönlere sahip olduğunu ortaya koymaktadır. Model düzeyindeki test metrikleri incelendiğinde, crop-based modun ihlal sınıflarında tutarlı biçimde daha yüksek recall değerleri ürettiği görülmektedir: baret ihlali için NO-Hardhat recall değeri scene modda 0,610 iken crop modda 0,884'e yükselmektedir (fark: +0,274). Maske ihlali için bu fark daha sınırlı olmakla birlikte yine crop modun lehine seyretmektedir (NO-Mask recall: 0,878 → 0,933). Yelek ihlalinde iki mod arasındaki fark önemsiz düzeyde kalmaktadır; bu bulgu, büyük görsel alana sahip PPE öğelerinde tam sahne bağlamının da etkili olabildiğini göstermektedir.

Sistem düzeyindeki video testi sonuçları ise her iki modun da net ihlal içeren senaryolarda yüksek başarı sergilediğini doğrulamaktadır: çoklu ihlal ve yelek ihlali içeren `noppe_test` ve `novest_test` videolarında her iki mod 3/3 doğrulukla ihlalleri tespit etmiştir. Belirsiz veya sınır koşullarına sahip videolarda (düşük çözünürlüklü `mask_test`) yanlış alarm oranı artmaktadır. Bu durum, sistemin güvenilir çalışması için yeterli video kalitesi ve çözünürlüğünün önemini vurgulamaktadır.

İşlem hızı bakımından scene-based mod, kişi sayısından bağımsız tutarlı bir FPS aralığı (41–51 FPS) sunmaktadır. Crop-based modda FPS, sahnedeki kişi sayısıyla ters orantılı biçimde değişmektedir: az kişili `mask_test` videosunda 54,5 FPS elde edilirken, 27 takip kimliğinin izlendiği kalabalık `noppe_test` videosunda bu değer 26,6 FPS'ye gerilemektedir. GPU bellek kullanımı ise her iki modda düşük düzeyde kalmakta (crop: 308 MB, scene: 366 MB) ve 6 GB VRAM kapasiteli RTX 3060 Laptop GPU'sunda bol yedek kapasite bırakmaktadır.

Bu bulgular birlikte değerlendirildiğinde, iki modun kullanım senaryosuna göre farklı tercihler sunduğu görülmektedir. Baret ve maske gibi küçük ve anatomik konuma bağlı PPE öğelerinin kritik olduğu, kamera sayısının sınırlı ve kişi yoğunluğunun düşük olduğu ortamlarda crop-based modun ihlal tespit hassasiyeti bakımından daha uygun olduğu değerlendirilmektedir. Geniş alanlı izleme, yüksek kişi yoğunluğu ve gerçek zamanlı işlem istikrarının önceliklendirildiği durumlarda ise scene-based mod pratik avantaj sağlamaktadır. Sistemin her iki modu da desteklemesi ve kullanıcının pipeline başlatılırken modu seçebilmesi, farklı fabrika koşullarına uyarlanabilirlik açısından önemli bir işletimsel esneklik sunmaktadır.

**_Sistem Parametresi Optimizasyonu_**

Pipeline davranışını belirleyen üç kritik hiperparametre, sistematik kıyaslama deneyleriyle sayısal olarak doğrulanmıştır. PPE çıkarım sıklığı (PPE_INFER_EVERY) optimizasyonunda, her çerçevede çıkarım yapan referans konfigürasyonu (skip=1) ile farklı atlama değerleri karşılaştırılmış; PPE_INFER_EVERY=4 değerinin gerçek zamanlı işlem eşiğini aşan FPS sağlarken (referansa göre 2,3 kat artış, 12,6 FPS → 29,4 FPS) ihlal tespit doğruluğunda anlamlı bir kayıp oluşturmadığı belirlenmiştir. PPE_INFER_EVERY=8 ve üzeri değerlerde ise dinamik sahnelerde baret known_rate'in belirgin biçimde düştüğü gözlemlenmiştir (%89,8 → %72,9).

Tespit güven eşiği optimizasyonu; baret, yelek ve maske için bağımsız analizler gerektirmiştir. Baret modelinin eşik değerlerine düşük duyarlılık gösterdiği (0,15 üzerinde kararlı davranış) ve 0,20 değerinin yeterli olduğu saptanmıştır. Yelek modelinde 0,30 değerinin known_rate ve violation_rate arasında en iyi dengeyi sağladığı belirlenmiştir. Maske modelinde ise ihlal tespitinin tüm eşik aralıklarında (%100 violation_rate) yüksek kalması nedeniyle 0,25 değeri seçilmiştir.

Temporal oylama penceresi (temporal_window) optimizasyonunda, pencere boyutunun küçük değerlerde (5 çerçeve) kararsız kararlar ürettiği; 20 çerçeveden itibaren known_rate değerlerinin plato oluşturduğu ve 20 üzerindeki değerlerde ise yelek ihlali tespit oranının monoton biçimde gerilediği görülmüştür (%44,9 → %40,9 → %36,8 → %32,8, sırasıyla window=20, 30, 40, 50). Bu bulgular, temporal_window = 20 değerinin kararlılık-hassasiyet dengesi açısından optimal nokta olduğunu göstermektedir.

**_Literatür ile Karşılaştırmalı Konum_**

Bölüm 3.7'de gerçekleştirilen sistematik karşılaştırma, SafetyMonitor'ün literatürdeki benchmark sistemlerine (Nath 2020 [6], Wu 2019 [7]) göre farklılaştığı boyutları ortaya koymaktadır. İncelenen 11 özellik açısından değerlendirildiğinde, SafetyMonitor; çift mod PPE tespiti, kimlik kararlılığı (TrackReattacher), temporal oylama, kamera sağlık izleme ve yerel LLM entegrasyonu gibi işletimsel süreklilik ve raporlama gerektiren alanlarda belirgin biçimde farklılaşmaktadır. Nath 2020 ve Wu 2019 ile doğrudan metrik karşılaştırması farklı veri kümeleri nedeniyle metodolojik kısıtlara sahip olmakla birlikte, SafetyMonitor'ün uçtan uca sistem bütünleşmesi (tespit → olay yönetimi → DB → LLM → web arayüzü) boyutunda mevcut çalışmalardan ayrıştığı görülmektedir.

**_Olay Yönetimi ve Veri Altyapısı_**

İhlal tespiti ile olay yönetimi arasındaki katman, sistemin pratik kullanım değerini belirleyen kritik bir unsurdur. Durum makinesi tabanlı olay yöneticisi (idle → new → active → closed), her tespit çıktısının doğrudan alarm üretmesini önleyen bir onay mekanizması işlevi görmektedir. `new_confirm_sec` parametresiyle belirlenen 3 saniyelik onay süreci, tek karelik hataların alarmı tetiklemesinin önüne geçmektedir. Benzer biçimde `resolved_confirm_sec` (5 saniye) ve `gap_tolerance` (1 saniye) parametreleri, ihlal süresinin kesintili tespitlerden aşırı parçalanmasını engellemektedir. Tekrar sayısı (`repeat_count`) ve süre (`duration_sec`) metrikleri her olay için hesaplanarak veritabanına kaydedilmekte; bu veriler hem LLM raporlarına girdi sağlamakta hem de yöneticilerin ihlal örüntülerini analiz etmesine olanak tanımaktadır.

PostgreSQL tabanlı veri katmanı, olaylar ve zaman çizelgesi için idempotent şema ve UPSERT mantığı kullanarak veri tutarlılığını güvence altına almaktadır. JSONB sütunları (signature, persons) şema değişikliğine gerek kalmadan esnek veri yapıları depolanabilmesini sağlamakta; false_positive soft-delete yaklaşımı operatörlerin yanlış alarmları işaretlemesine imkân tanımaktadır.

**_LLM Entegrasyonunun Değeri ve Kısıtları_**

Yerel LLM entegrasyonu, SafetyMonitor'ün literatürdeki kamera tabanlı sistemlerden ayrıştığı özgün bileşenlerden biridir. Per-alarm raporları, tespit anında arka planda üretilerek kişi kimliği, konum, süre ve tekrar sayısı gibi yapılandırılmış verileri işleçlere anlaşılır Türkçe metne dönüştürmektedir. Periyodik raporlar ise önceden hesaplanmış analitik verileri (risk skoru, trend, lokasyon dağılımı) entegre ederek yöneticilerin manuel analiz yükünü azaltmaktadır.

Bu entegrasyonun en belirgin değeri gizlilik ve maliyet boyutlarında ortaya çıkmaktadır: iş güvenliği kayıtlarının (kişi kimlikleri, kamera konumları, olay zamanlamaları) bulut API'lerine gönderilmesine gerek kalmaksızın tüm çıkarım yerel donanımda gerçekleştirilmektedir. Bununla birlikte, yerel qwen3:8b modelinin kapasitesinin bulut tabanlı büyük dil modellerinin oldukça gerisinde kaldığı ve özellikle karmaşık bağlamsal akıl yürütme gerektiren durumlarda çıktı kalitesinin düştüğü gözlemlenmiştir. "Bilinmiyor" bölge sorunu gibi sistem yapılandırma hatalarının rapor kalitesini doğrudan etkilemesi, kullanıcı eğitimi ve sistem kurulumunun LLM çıktı kalitesi üzerindeki belirleyici etkisini ortaya koymaktadır.

**_Sistemin Güçlü Yönleri_**

SafetyMonitor'ün mevcut iş güvenliği izleme çözümlerine kıyasla öne çıkan güçlü yönleri şu şekilde sıralanabilir. Her şeyden önce sistem, ticari kamera veya özelleşmiş donanım gerektirmeksizin standart USB kamera ya da video dosyasıyla çalışabilmekte; bu durum kurulum ve işletim maliyetini önemli ölçüde düşürmektedir. İki modun desteklenmesi farklı ortam ve kapasitelere uyarlanabilirlik sağlamakta; konfigurasyon arayüzü aracılığıyla detection eşikleri, PPE etkinleştirme/devre dışı bırakma ve yangın filtre parametreleri operatör müdahalesi olmaksızın dinamik olarak değiştirilebilmektedir.

Gerçek zamanlı web arayüzü, Socket.IO tabanlı anlık bildirimlerle alarm durumunun herhangi bir tarayıcı üzerinden izlenmesine olanak tanımaktadır; yüklü yazılım veya özel istemci gerekmemektedir. Kamera izleme sistemi (donma, karanlık, çevrimdışı algılama) altyapı arızalarının tespit sürecini kesintiye uğratmasını önceden uyarı mekanizmasıyla azaltmaktadır. Export sistemi (CSV ve PDF) oluşturulan verilerin kurumsal raporlama süreçlerine aktarılmasını kolaylaştırmaktadır.

**_Kısıtlar ve Sınırlamalar_**

Çalışmanın açıkça kabul edilmesi gereken kısıtları da bulunmaktadır. Model eğitiminde kullanılan veri setleri belirli çekim koşullarını (aydınlatma, açı, kişi sayısı) yansıtmakta; sistematik olarak farklı koşullarda (gece çekimi, yoğun sis, çok düşük çözünürlük) kapsamlı değerlendirme yapılmamıştır. Test sürecinde kullanılan dört video, senaryolar bakımından çeşitlilik sunmakla birlikte gerçek fabrika ortamlarının tüm varyasyonlarını kapsayacak büyüklükte değildir.

Tek kamera mimarisi, sistemin temel tasarım kısıtıdır. Mevcut uygulama tek bir video akışını işleyebilmekte; çok sayıda kameradan aynı anda kesintisiz veri alınabilmesi için ek mimari geliştirme gerekmektedir. Bunun yanı sıra kişi tespiti modelinin başarısı, crop-based modda tüm PPE kararlarının doğruluğunu doğrudan etkilemektedir; kişi sınır kutusunun hatalı belirlenmesi PPE crop alanının da yanlış hesaplanmasına yol açmaktadır. Son olarak, LLM bileşeni internet bağlantısı gerektirmemekle birlikte çalıştırılabilmesi için Ollama ve model dosyasının ayrıca kurulması gerekmektedir; bu durum kolay kurulum sürecini biraz karmaşıklaştırmaktadır.

**_Gelecekteki Çalışma Yönleri_**

Bu tez çerçevesinde oluşturulan altyapı, çeşitli genişletme ve iyileştirme yönleri için sağlam bir temel oluşturmaktadır.

*Çok kamera desteği:* En doğrudan geliştirme yönü, sistemin birden fazla kamera akışını eş zamanlı işleyebilecek biçimde genişletilmesidir. Bu geliştirme, her kamera için bağımsız pipeline süreçleri ve merkezi bir olay yöneticisi arasında koordinasyon mekanizması gerektirmektedir. Mevcut PostgreSQL şeması ve kamera kimliği altyapısı bu genişlemeye hazır biçimde tasarlanmıştır.

*Kenar cihaz dağıtımı (edge deployment):* RTX 3060 GPU gereksinimini azaltmak ve sistemin daha yaygın donanımlara taşınmasını sağlamak amacıyla model sıkıştırma (quantization) veya küçültme (pruning) teknikleri uygulanabilir. NVIDIA Jetson serisi cihazlar veya Intel OpenVINO ekosistemi, fabrika ortamlarında merkezi sunucu gerektirmeyen dağıtık kurulum senaryoları için değerlendirilebilir.

*Mobil bildirim entegrasyonu:* Mevcut sistemde alarmlar web arayüzü ve Socket.IO bildirimleriyle iletilmektedir. Firebase Cloud Messaging veya benzeri bir push bildirim altyapısının eklenmesiyle güvenlik yöneticilerinin mobil cihazlarına anlık uyarı gönderilebilir; bu özellik özellikle denetimsiz vardiya saatlerinde büyük değer taşımaktadır.

*Çevrimiçi model güncelleme:* Sahadan toplanan görüntülerin insan etiketlemesiyle modellerin sürekli güncellenmesini sağlayan bir aktif öğrenme döngüsü, sistemin farklı fabrika koşullarına ve PPE türlerine zamanla uyum sağlamasını mümkün kılabilir. Bu yaklaşım, veri gizliliğini korumak amacıyla federe öğrenme yöntemleriyle de gerçekleştirilebilir.

*Eylem tanıma ve risk tahmini:* Mevcut sistem anlık PPE varlığı/yokluğunu tespit etmektedir. Gelecekte çalışanların hareketlerini ve davranışlarını inceleyen eylem tanıma modülleri eklenerek (tehlikeli bölgeye yaklaşma, ergonomik risk hareketleri gibi) proaktif iş güvenliği değerlendirmesi yapılabilir. Geçmiş ihlal örüntülerine dayalı risk tahmin modelleri de bu altyapı üzerine inşa edilebilir.

*LLM kalite iyileştirmesi:* Yerel LLM bileşeni, daha büyük model ağırlıkları (örneğin qwen3:14b veya qwen3:32b) veya özelleştirilmiş ince ayar (fine-tuning) yoluyla geliştirilebilir. Yapılandırılmış JSON çıktı formatı ile doğal dil anlatımının birlikte kullanılması, hem makine tarafından işlenebilir hem de insan tarafından okunabilir raporların üretilmesini sağlayabilir.

Bu çalışmada geliştirilen SafetyMonitor sistemi, yalnızca teknik bir prototip olarak değil; gerçek fabrika koşullarında konuşlandırılabilecek, bakımı yapılabilecek ve zaman içinde genişletilebilecek bir sistem mimarisi olarak tasarlanmıştır. Sistem, konfigürasyon dosyası tabanlı parametre yönetimi, idempotent veritabanı şeması, modüler bileşen yapısı ve kapsamlı API tasarımıyla uzun vadeli işletim ve geliştirme süreçleri gözetilerek oluşturulmuştur. İş güvenliği izleme alanında yapay zeka ve bilgisayarlı görü teknolojilerinin erişilebilir ve sürdürülebilir biçimde uygulanmasına yönelik bu çalışmanın, gelecekteki araştırma ve uygulamalar için yararlı bir referans oluşturması umulmaktadır.

---

# KAYNAKLAR

[1] International Labour Organization (ILO). (2023). *World Employment and Social Outlook: Trends 2023*. Geneva: International Labour Organization.

[2] Sosyal Güvenlik Kurumu (SGK). (2023). *2022 Yılı İş Kazası ve Meslek Hastalığı İstatistikleri*. Ankara: SGK Yayınları.

[3] Occupational Safety and Health Administration (OSHA). (2022). *Personal Protective Equipment: OSHA 3151-12R*. Washington: U.S. Department of Labor.

[4] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788. https://doi.org/10.1109/CVPR.2016.91

[5] Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Sürüm 8.0.0) [Bilgisayar yazılımı]. https://github.com/ultralytics/ultralytics

[6] Nath, N. D., Behzadan, A. H., & Paal, S. G. (2020). Deep learning for site safety: Real-time detection of personal protective equipment. *Automation in Construction*, 117, 103282. https://doi.org/10.1016/j.autcon.2020.103282

[7] Wu, J., Cai, N., Chen, W., Wang, H., & Wang, G. (2019). Automatic detection of hardhats worn by construction personnel: A deep learning approach and benchmark dataset. *Automation in Construction*, 106, 102894. https://doi.org/10.1016/j.autcon.2019.102894

[8] Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. *2016 IEEE International Conference on Image Processing (ICIP)*, 3464–3468. https://doi.org/10.1109/ICIP.2016.7533003

[9] Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. *2017 IEEE International Conference on Image Processing (ICIP)*, 3645–3649. https://doi.org/10.1109/ICIP.2017.8296962

[10] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *European Conference on Computer Vision (ECCV)*, Cilt 13682, 1–21. https://doi.org/10.1007/978-3-031-20047-2_1

[11] OpenAI. (2023). *GPT-4 Technical Report*. arXiv ön baskı arXiv:2303.08774.

[12] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., & diğerleri. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv ön baskı arXiv:2307.09288*.

[13] Ollama. (2024). *Ollama: Get up and running with large language models* [Bilgisayar yazılımı]. https://ollama.com

---

# EKLER

## EK-1: Fizibilite Raporu

**Proje Adı:** FABRİKA İŞ GÜVENLİĞİ İZLEME SİSTEMİ (SafetyMonitor)  
**Tarih:** Mayıs 2026

---

### 1. Proje Tanımı ve Kapsam

**1.1 Problem Tanımı**

Fabrika ortamlarında iş güvenliği denetimi büyük ölçüde manuel gözleme dayanmaktadır. Bu yaklaşım; büyük üretim alanlarının anlık izlenememesi, insan kaynaklı dikkat dağınıklığı ve sürekli gözetlemenin yüksek maliyeti nedeniyle yetersiz kalmaktadır. ILO verilerine göre her yıl dünya genelinde yaklaşık 2,3 milyon çalışan iş kazası veya meslek hastalığı nedeniyle hayatını kaybetmekte; bu kazaların önemli bir bölümünün kişisel koruyucu ekipman (KKD) kullanım uyumsuzluğundan kaynaklandığı bilinmektedir.

**1.2 Proje Hedefleri**

- Baret, güvenlik yeleği ve yüz maskesi kullanım uyumluluğunu gerçek zamanlı olarak tespit etmek
- Yangın ve duman riskini otomatik olarak algılayıp ilgili personele anlık bildirmek
- İhlal olaylarını zaman damgası, kişi kimliği ve görsel kanıt ile veritabanında kayıt altına almak
- Haftalık ve aylık güvenlik raporlarını yerel LLM aracılığıyla otomatik olarak üretmek
- Tüm bu işlevleri operatörün kolaylıkla kullanabileceği web tabanlı bir arayüz üzerinden sunmak

**1.3 Kapsam Tablosu**

| Konu | Kapsam İçi | Kapsam Dışı |
|------|-----------|------------|
| KKD tespiti | Baret, güvenlik yeleği, yüz maskesi | Eldiven, koruyucu gözlük, emniyet kemeri |
| Tehlike tespiti | Yangın, duman | Gaz sızıntısı, akustik uyarı |
| Kamera sayısı | Tek kamera kaynağı (USB / video dosyası) | Eş zamanlı çoklu kamera akışı |
| Kullanıcı arayüzü | Web tarayıcı tabanlı (masaüstü) | Mobil uygulama |
| Dağıtım ortamı | Yerel sunucu (LAN) | Bulut dağıtımı, edge cihazı |
| İşletim sistemi | Windows 11 | Gömülü Linux / NVIDIA Jetson |

---

### 2. Teknik Fizibilite

**2.1 Donanım Gereksinimleri**

| Bileşen | Minimum Gereksinim | Kullanılan Donanım | Değerlendirme |
|---------|------------------|-------------------|---------------|
| GPU | CUDA destekli, ≥4 GB VRAM | NVIDIA RTX 3060 Laptop (6 GB GDDR6) | 8 modelin eş zamanlı yüklenmesi için yeterli; peak GPU 308–366 MB |
| CPU | ≥4 çekirdek | Intel i7-12650H (10 çekirdek, 16 iş parçacığı) | ByteTrack, Flask worker'ları ve SocketIO I/O için yeterli |
| RAM | ≥16 GB | 32 GB DDR5 | Veri tampon belleği, pipeline ve backend eş zamanlı çalışması için yeterli |
| Depolama | ≥50 GB | NVMe SSD | Hızlı model yükleme ve PostgreSQL veri yazımı için uygun |
| İşletim Sistemi | Windows 10/11 veya Linux | Windows 11 Pro | Windows dosya seçici API'si ve NTFS deduplication desteği |
| CUDA | ≥11.x | CUDA 11.8 | PyTorch 2.7.1+cu118 ile uyumlu |

**2.2 Yazılım Gereksinimleri**

| Kütüphane / Araç | Gereksinim | Kullanılan Sürüm | Lisans |
|-----------------|-----------|-----------------|--------|
| Python | ≥3.9 | 3.9.13 | PSF |
| PyTorch + CUDA | ≥2.0.0 | 2.7.1+cu118 | BSD |
| Ultralytics (YOLOv8) | ≥8.0.0 | 8.x | AGPL-3.0 |
| OpenCV | ≥4.8.0 | 4.8.x | Apache-2.0 |
| Flask | ≥3.0.0 | 3.x | BSD |
| Flask-SocketIO | ≥5.3.0 | 5.3.x | MIT |
| psycopg2-binary | ≥2.9.0 | 2.9.x | LGPL |
| ReportLab | ≥4.0.0 | 4.x | BSD |
| Ollama Python SDK | ≥0.1.0 | 0.x | MIT |
| PostgreSQL | ≥13 | 15.x | PostgreSQL |
| React | ≥18.0 | 18.3.1 | MIT |
| Recharts | ≥3.0 | 3.8.1 | MIT |
| Vite | ≥5.0 | 5.4.2 | MIT |

Tüm bağımlılıklar açık kaynaklıdır. Ultralytics YOLOv8, akademik ve araştırma kullanımı için AGPL-3.0 lisansı kapsamında ücretsiz kullanılabilmektedir.

**2.3 Teknoloji Alternatifleri Karşılaştırması**

*Nesne Tespiti:*

| Yöntem | Hız | Doğruluk | Gerekçe |
|--------|-----|----------|---------|
| **YOLOv8 (seçilen)** | Yüksek (40–55 FPS) | Yüksek | Gerçek zamanlı tespit, esnek model boyutu (n/s/m), aktif topluluk desteği |
| Faster R-CNN | Düşük (5–10 FPS) | Çok yüksek | Gerçek zamanlı kullanım için hız yetersiz |
| SSD | Orta (20–30 FPS) | Orta | YOLOv8'e kıyasla daha düşük doğruluk |

*Çok Nesne Takibi:*

| Yöntem | Kimlik Sürekliliği | Düşük Güven Tespiti | Gerekçe |
|--------|-------------------|-------------------|---------|
| **ByteTrack (seçilen)** | Yüksek | Destekli | Eşik altı adayları izleme sürecine dahil eder; kimlik kaybını azaltır |
| SORT | Orta | Desteksiz | Oklüzyon sonrası kimlik kaybı yaygın |
| DeepSORT | Orta–Yüksek | Kısmen | Ek görünüm modeli gerektirir; gecikme artar |

*Veritabanı:*

| Seçenek | JSONB Desteği | Eş Zamanlı Erişim | Gerekçe |
|---------|-------------|-----------------|---------|
| **PostgreSQL (seçilen)** | Yerel JSONB | Güçlü | `signature` ve `persons` alanları için native JSONB; bağlantı havuzu |
| MySQL | Sınırlı JSON | Orta | JSONB indeks ve sorgu kabiliyeti yetersiz |
| SQLite | Yok | Zayıf | Pipeline ve backend eş zamanlı yazımında kilit çakışması |

*LLM Entegrasyonu:*

| Seçenek | Gizlilik | Maliyet | Gecikme | Gerekçe |
|---------|---------|---------|---------|---------|
| **Ollama — qwen2.5:7b (seçilen)** | Tam | Sıfır | Orta (30–60 s) | Fabrika verisi dışarı çıkmaz; internet bağımsız |
| OpenAI API (GPT-4) | Yok | API ücreti | Düşük (~3–5 s) | Fabrika verisi buluta gönderilir; gizlilik riski |
| HuggingFace Inference | Kısmi | API ücreti | Değişken | İnternet bağımlılığı; SLA garantisi yok |

---

### 3. Operasyonel Fizibilite

**3.1 Kullanıcı Profili**

| Kullanıcı Tipi | Teknik Düzey | Kullanacağı İşlevler |
|---------------|-------------|---------------------|
| Güvenlik görevlisi | Düşük–orta | Dashboard anlık izleme, alarm geçmişi inceleme, not ekleme |
| Fabrika yöneticisi | Orta | Haftalık/aylık raporlar, CSV/PDF dışa aktarma, trend analizi |
| Sistem yöneticisi | Yüksek | Kamera kurulumu, ayarlar paneli, pipeline başlatma/durdurma |

**3.2 Operasyonel Gereksinimler**

- Modern web tarayıcı (Chrome, Firefox, Edge sürüm ≥90) — istemci tarafında kurulum gerekmez
- Yerel ağ (LAN) erişimi — internet bağlantısı zorunlu değildir
- PostgreSQL ve Ollama servislerinin çalışır durumda olması
- Kamera: USB webcam veya video dosyası (.mp4, .avi vb.)

**3.3 Bakım ve Güncelleme Kolaylığı**

Tüm sistem parametreleri `config.yaml` dosyasından yönetilmektedir; kod değişikliği gerekmeksizin tespit eşikleri, olay onay süreleri, kamera izleme hassasiyeti ve LLM modeli güncellenebilmektedir. Veritabanı şeması idempotent `CREATE TABLE IF NOT EXISTS` ve `ALTER TABLE ADD COLUMN IF NOT EXISTS` yapısıyla her yeniden başlatmada güvenle çalıştırılabilmektedir. Model dosyaları, `config.yaml` içindeki yollar güncellenerek değiştirilebilir; yeni model denemesi için kod düzenlenmesine gerek yoktur.

---

### 4. Süre Fizibilitesi

| Aşama | Tahmini Süre | Gerçekleşen Süre | Notlar |
|-------|-------------|----------------|--------|
| Veri seti hazırlama ve model eğitimi | 4 hafta | 5 hafta | 8 ayrı model + compare_vest.py karşılaştırması ek süre aldı |
| Detection pipeline geliştirme | 3 hafta | 4 hafta | TrackReattacher'ın 5 sinyalli tasarımı planlananın ötesine geçti |
| Backend ve veritabanı geliştirme | 3 hafta | 3 hafta | — |
| Frontend geliştirme | 2 hafta | 2 hafta | — |
| LLM entegrasyonu | 1 hafta | 1 hafta | — |
| Test ve iyileştirme | 2 hafta | 2 hafta | — |
| **Toplam** | **~15 hafta** | **~17 hafta** | **%13 sapma** |

**Kritik Yol:** Model eğitimi → Detection pipeline → Backend API → Frontend → Entegrasyon testi

Model eğitimi aşaması kritik yolun başını oluşturmakta; bu aşamadaki gecikme sonraki tüm aşamaları doğrudan etkilemektedir. Fiilen yaşanan 2 haftalık sapma, baret için crop-based ve scene-based olmak üzere iki ayrı model mimarisinin paralel geliştirilmesinden ve yelek model karşılaştırma sürecinden kaynaklanmıştır. Bu kapsam genişlemesi kabul edilebilir bir aralıkta değerlendirilmiştir.

---

### 5. Maliyet Fizibilitesi

**5.1 Proje Geliştirme Maliyeti**

| Kalem | Maliyet | Açıklama |
|-------|---------|---------|
| Donanım | 0 ₺ | Mevcut geliştirici donanımı kullanıldı; ek satın alma yapılmadı |
| Yazılım lisansları | 0 ₺ | Tüm bileşenler açık kaynak veya ücretsiz lisans kapsamında |
| Bulut / API ücreti | 0 ₺ | Yerel LLM (Ollama) tercih edildi; OpenAI API kullanılmadı |
| Veri seti | 0 ₺ | Roboflow ücretsiz plan kapsamında erişildi |
| **Toplam** | **0 ₺** | — |

**5.2 Alternatif Çözümlerle Tahmini Maliyet Karşılaştırması**

| Alternatif Çözüm | Tahmini Maliyet |
|-----------------|----------------|
| Ticari KKD izleme yazılımı (SaaS abonelik) | 500–2.000 $/ay |
| Bulut Vision API (Google Cloud Vision / Azure CV) | 1–5 $/1.000 görüntü analizi |
| Bulut LLM API (GPT-4 Turbo tabanlı raporlama) | 10–30 $/1 milyon token |
| **SafetyMonitor (bu proje)** | **0 ₺** |

---

### 6. Risk Analizi

| Risk | Olasılık | Etki | Azaltma Stratejisi |
|------|---------|------|-------------------|
| GPU bellek yetersizliği | Düşük | Yüksek | YOLOv8n crop modeli; `temporal_window` ve `mask_imgsz` parametresiyle bellek kontrolü |
| ByteTrack kimlik kaybı (oklüzyon) | Orta | Orta | TrackReattacher: 5 sinyalli ağırlıklı benzerlik fonksiyonu, MIN_SCORE=0.70 eşiği |
| LLM yanıt gecikmesi veya sunucu hatası | Orta | Düşük | Async akış (POST anında döner); HTTP 500 durumunda kullanıcı bildirim socket'i |
| Veritabanı bağlantı kesintisi | Düşük | Orta | `results/` dizini dosya sistemi fallback modu; DB kapalıyken sistem çalışmaya devam eder |
| Düşük ışık veya kirli lens (karanlık frame) | Orta | Orta | `cam_dark_thresh` + `cam_dark_frames` kamera izleme; `camera_status` banner bildirimi |
| Donmuş kamera akışı | Düşük | Orta | `cam_freeze_diff` + `cam_freeze_frames` kamera izleme; otomatik bildirim |
| Yanlış pozitif alarmlar | Orta | Orta | Temporal voting (20 frame penceresi), grace period; operatör onaylı FP işaretleme akışı |
| Model performans düşüşü (ortam değişikliği) | Düşük | Yüksek | `config.yaml` üzerinden model değişimi; Roboflow entegrasyonu ile yeniden eğitim kolaylığı |

---

### 7. Sonuç

SafetyMonitor projesi; teknik, operasyonel, süre, maliyet ve risk yönetimi boyutlarının tamamında uygulanabilir niteliktedir. Mevcut donanım altyapısı gerçek zamanlı çok-model inferans için yeterli kapasiteye sahip olup tüm yazılım bileşenleri açık kaynak lisansları kapsamında sıfır maliyetle temin edilmiştir. Sistem, web tarayıcı erişimi dışında herhangi bir özel istemci yazılımı gerektirmemekte; tüm parametreler merkezi bir konfigürasyon dosyası aracılığıyla yönetilebilmektedir. Proje süresinde yaşanan %13'lük sapma, kapsam genişlemesinden (iki modun paralel geliştirilmesi) kaynaklanmakta olup kabul edilebilir bir aralıkta kalmaktadır. Belirlenen risklerin büyük bölümü için sistem içinde otomatik azaltma mekanizmaları geliştirilmiş, projenin sürdürülebilir ve genişletilebilir biçimde tamamlanması sağlanmıştır.

---

## EK-2: Sistem Test Raporu

**Proje Adı:** SafetyMonitor — FABRİKA İŞ GÜVENLİĞİ İZLEME SİSTEMİ  
**Test Dönemi:** Nisan–Mayıs 2026  
**Test Ortamı:** Intel i7-12650H, 32 GB RAM, NVIDIA RTX 3060 Laptop 6 GB, CUDA 11.8, Python 3.9.13

---

**1. Birim Test Sonuçları**

| Bileşen | Test Edilen İşlev | Sonuç |
|---------|-----------------|-------|
| TrackReattacher | MIN_SCORE=0.70 eşiğinin altında kalan adayların reddedilmesi | BAŞARILI |
| TrackReattacher | Yüksek benzerlikli yeniden bağlama sonrası stable_pid korunması | BAŞARILI |
| Event State Machine | idle→new geçişi (new_confirm_sec=3.0 s) | BAŞARILI |
| Event State Machine | active→closed geçişi (resolved_confirm_sec=5.0 s) | BAŞARILI |
| Temporal Voting | 20 frame penceresi ile çoğunluk kararı | BAŞARILI |
| DB Writer | close_event — duration_sec ve repeat_count doğru yazımı | BAŞARILI |
| API `/api/events` | Filtre parametreleri (tarih, ihlal türü, durum) | BAŞARILI |
| API `/api/pipeline/start` | mode=crop ve mode=scene ile subprocess başlatma | BAŞARILI |
| Socket.IO | new_alert eventi — frontend anlık bildirimi | BAŞARILI |
| LLM async akışı | POST dönüşü anlık, rapor Socket.IO ile iletim | BAŞARILI |

**2. Entegrasyon Test Sonuçları**

| Test Senaryosu | Beklenen Davranış | Sonuç |
|---------------|-----------------|-------|
| Canlı kamera başlatma | Pipeline başlat → kamera görüntüsü akışı | BAŞARILI |
| Baret ihlali tespiti → alarm | Tespit → new_alert socket → Dashboard güncellemesi | BAŞARILI |
| Event otomatik kapanışı | 5 s ihlalsiz → event_closed socket → AlertHistory güncellemesi | BAŞARILI |
| Yanlış pozitif işaretleme | İki adımlı onay akışı → false_positive=true DB yazımı | BAŞARILI |
| CSV dışa aktarma | UTF-8 BOM kodlama — Excel uyumlu açılış | BAŞARILI |
| PDF dışa aktarma | ReportLab özet + tablo — Türkçe karakter desteği | BAŞARILI |
| LLM haftalık rapor | POST → async üretim → report_llm_ready socket iletimi | BAŞARILI |
| DB kapalı fallback | results/ dizininden dosya okuma modu aktif | BAŞARILI |
| Dark/Light mod geçişi | Tüm sayfalar — CSS değişkenleri anlık güncelleme | BAŞARILI |
| Kamera donması (freeze) | cam_freeze_frames aşımı → camera_status banner görünümü | BAŞARILI |

**3. Performans Test Sonuçları**

Detaylı model ve sistem performans ölçümleri için Bölüm 3.2 ve Bölüm 3.3'e bakılabilir. Özet olarak: crop modunda ortalama 40.2 FPS, scene modunda 46.4 FPS elde edilmiştir. Her iki mod da RTX 3060 Laptop GPU üzerinde gerçek zamanlı işleme eşiği olan 25 FPS'in belirgin biçimde üzerinde çalışmaktadır.

**4. Bilinen Sınırlamalar**

- Mevcut mimari tek kamera kaynağını desteklemektedir; eş zamanlı çoklu kamera henüz eklenmemiştir.
- Ollama yerel sunucu çalışmıyorsa LLM raporu üretilemez (HTTP 500 döner).
- Windows'a özgü dosya yolu işlemleri (path traversal koruması, Windows dosya seçici diyaloğu) Linux ortamında test edilmemiştir.

---

## EK-3: config.yaml Tam İçeriği

```yaml
backend:
  cors_origins:
  - http://localhost:5173
  - http://localhost:5174
  - http://127.0.0.1:5173
  - http://127.0.0.1:5174
  host: 0.0.0.0
  port: 5050
database:
  enabled: true
  host: localhost
  name: ppe_db
  password: '1234'
  port: 5432
  user: postgres
detection:
  fire_confidence: 0.5
  min_hits: 3
  person_confidence: 0.4
  ppe_confidence: 0.35
event_manager:
  fire_clear_frames: 10
  fire_confirm_frames: 20
  new_confirm_sec: 3.0
  resolved_confirm_sec: 5.0
llm:
  base_url: http://localhost:11434
  enabled: true
  model: qwen2.5:7b
  temperature: 0.3
  timeout: 120
models:
  crop:
    helmet_model: models/bera/crophelmet_agent_final_best.pt
    mask_imgsz: 640
    mask_model: models/bera/cropmask_agent_final_best.pt
    vest_model: models/bera/cropvest_agent_final_best.pt
  device: cuda
  fire_model: models/bera/fire_smoke_other_agent_final_best.pt
  person_model: models/person_agent_scene_vinayakstyle_best.pt
  scene:
    helmet_model: models/vinayak_trained_byBera/helmet_agent_final_best.pt
    mask_model: models/mask_agent_scene_200ep_yolov8m_best.pt
    vest_model: models/vinayak_trained_byBera/vest_agent_final_best.pt
ppe_pipeline:
  cam_dark_frames: 60
  cam_dark_thresh: 0.03
  cam_freeze_diff: 0.002
  cam_freeze_frames: 60
  fire_conf: 0.5
  fire_growth_factor: 1.5
  fire_growth_window: 10
  fire_min_area_ratio: 0.027
  helmet_conf: 0.2
  mask_conf: 0.25
  new_confirm_sec: 3
  person_conf: 0.25
  resolved_confirm_sec: 5
  temporal_window: 20
  use_fire: true
  use_helmet: true
  use_mask: true
  use_vest: true
  vest_conf: 0.3
results_dir: results
results_keep_events: 50
```

---

## EK-4: Veritabanı Şeması (schema.sql)

```sql
-- ============================================================
-- PPE Güvenlik Sistemi — PostgreSQL Şeması
-- Idempotent: CREATE TABLE IF NOT EXISTS kullanılır
-- ============================================================

-- Her event'in güncel (en son) durumu
CREATE TABLE IF NOT EXISTS events (
    id               SERIAL       PRIMARY KEY,
    event_id         VARCHAR(20)  UNIQUE NOT NULL,
    event_status     VARCHAR(20)  NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL,
    updated_at       TIMESTAMPTZ  NOT NULL,
    repeat_count     INT          NOT NULL DEFAULT 0,
    duration_sec     FLOAT        NOT NULL DEFAULT 0.0,
    helmet_violation BOOLEAN      NOT NULL DEFAULT FALSE,
    vest_violation   BOOLEAN      NOT NULL DEFAULT FALSE,
    mask_violation   BOOLEAN      NOT NULL DEFAULT FALSE,
    fire_detected    BOOLEAN      NOT NULL DEFAULT FALSE,
    signature        JSONB,
    llm_report       TEXT,
    persons          JSONB,
    false_positive   BOOLEAN      NOT NULL DEFAULT FALSE
);

-- Her event'in tüm durum geçişleri (zaman çizgisi)
CREATE TABLE IF NOT EXISTS event_timeline (
    id             SERIAL       PRIMARY KEY,
    event_id       VARCHAR(20)  NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    event_status   VARCHAR(20)  NOT NULL,
    ts             TIMESTAMPTZ  NOT NULL,
    repeat_count   INT          NOT NULL DEFAULT 0,
    duration_sec   FLOAT        NOT NULL DEFAULT 0.0,
    change_reason  TEXT,
    signature      JSONB,
    llm_report     TEXT,
    image_filename VARCHAR(200),
    recorded_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    persons        JSONB
);

-- Operatör notları
CREATE TABLE IF NOT EXISTS event_notes (
    id          SERIAL       PRIMARY KEY,
    event_id    VARCHAR(20)  NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    note_text   TEXT         NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ALTER TABLE migrasyonları (idempotent)
ALTER TABLE events          ADD COLUMN IF NOT EXISTS persons        JSONB;
ALTER TABLE event_timeline  ADD COLUMN IF NOT EXISTS persons        JSONB;
ALTER TABLE events          ADD COLUMN IF NOT EXISTS camera_id      VARCHAR(20);
ALTER TABLE events          ADD COLUMN IF NOT EXISTS zone           VARCHAR(50);
ALTER TABLE events          ADD COLUMN IF NOT EXISTS false_positive BOOLEAN NOT NULL DEFAULT FALSE;

-- Otomatik ve manuel oluşturulan LLM raporları
CREATE TABLE IF NOT EXISTS llm_reports (
    id             SERIAL       PRIMARY KEY,
    period         VARCHAR(10)  NOT NULL,
    report_date    VARCHAR(10)  NOT NULL,
    llm_text       TEXT         NOT NULL,
    generated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    auto_generated BOOLEAN      NOT NULL DEFAULT FALSE,
    UNIQUE (period, report_date)
);

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_events_status         ON events(event_status);
CREATE INDEX IF NOT EXISTS idx_events_updated_at     ON events(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_created_at     ON events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_helmet         ON events(helmet_violation);
CREATE INDEX IF NOT EXISTS idx_events_vest           ON events(vest_violation);
CREATE INDEX IF NOT EXISTS idx_events_mask           ON events(mask_violation);
CREATE INDEX IF NOT EXISTS idx_events_fire           ON events(fire_detected);
CREATE INDEX IF NOT EXISTS idx_timeline_event_id     ON event_timeline(event_id);
CREATE INDEX IF NOT EXISTS idx_timeline_ts           ON event_timeline(ts DESC);
CREATE INDEX IF NOT EXISTS idx_notes_event_id        ON event_notes(event_id);
CREATE INDEX IF NOT EXISTS idx_events_false_positive ON events(false_positive);
```

---

## EK-5: API Endpoint Referans Tablosu

| Yöntem | Uç Nokta | Açıklama |
|--------|----------|---------|
| GET | `/api/events` | Filtrelenmiş event listesi (tarih, ihlal türü, durum parametreleri) |
| GET | `/api/events/<id>` | Tek event — timeline + operatör notları |
| POST | `/api/events/<id>/note` | Event'e operatör notu ekle |
| PATCH | `/api/events/<id>/close` | Event'i kapat (status → closed, repeat_count güncelle) |
| PATCH | `/api/events/<id>/resolve` | Event'i kapat — `/close` ile işlevsel olarak özdeş |
| PATCH | `/api/events/<id>/llm` | Per-alarm LLM raporu DB'ye yaz |
| GET | `/api/stats` | Dashboard istatistikleri (toplam, aktif, tür dağılımı) |
| GET | `/api/reports` | Grafik verisi (daily / weekly / monthly) |
| GET | `/api/reports/summary` | Risk skoru, trend analizi, lokasyon dağılımı (DB gerekli) |
| POST | `/api/reports/summary/llm` | LLM raporu üret (async — sonuç Socket.IO ile iletilir) |
| GET | `/api/reports/saved` | Kayıtlı LLM raporları listesi |
| GET | `/api/reports/saved/<id>` | Tek kayıtlı LLM raporu |
| GET | `/api/config` | Mevcut PPE pipeline konfigürasyonu |
| PUT | `/api/config` | PPE pipeline konfigürasyonunu güncelle |
| GET | `/api/pipeline/status` | Pipeline durumu (`running`, `source`, `camera_id`, `zone`, `mode`) |
| POST | `/api/pipeline/start` | Pipeline başlat (`source`, `camera_id`, `zone`, `mode`) |
| POST | `/api/pipeline/stop` | Çalışan pipeline'ı durdur |
| GET | `/api/pipeline/browse` | Windows dosya seçici diyaloğu (video dosyası seçimi) |
| POST | `/api/pipeline/camera-status` | Kamera durum bildirimi → `camera_status` socket eventi |
| GET | `/api/images/<event_id>/<fname>` | Event'e ait kayıtlı görüntü dosyası |

---

## EK-6: Kesinleşmiş Model Dosyaları ve Seçim Kriterleri

**Tablo EK-6.1: Kullanılan Model Dosyaları**

| Model | Dosya Yolu | Kullanım Modu | Notlar |
|-------|-----------|-------------|--------|
| Kişi tespiti | `models/person_agent_scene_vinayakstyle_best.pt` | Her iki mod | ByteTrack için temel kişi tespiti |
| Yangın/Duman | `models/bera/fire_smoke_other_agent_final_best.pt` | Her iki mod | YOLOv8n; fire + smoke + other sınıfları |
| Baret (crop) | `models/bera/crophelmet_agent_final_best.pt` | Yalnızca crop | NO-Hardhat recall=0.884 |
| Yelek (crop) | `models/bera/cropvest_agent_final_best.pt` | Yalnızca crop | compare_vest.py karşılaştırması ile seçildi |
| Maske (crop) | `models/bera/cropmask_agent_final_best.pt` | Yalnızca crop | YOLOv8m, mask_imgsz=640 |
| Baret (scene) | `models/vinayak_trained_byBera/helmet_agent_final_best.pt` | Yalnızca scene | NO-Hardhat recall=0.610 |
| Yelek (scene) | `models/vinayak_trained_byBera/vest_agent_final_best.pt` | Yalnızca scene | Tam kare yelek tespiti |
| Maske (scene) | `models/mask_agent_scene_200ep_yolov8m_best.pt` | Yalnızca scene | YOLOv8m, 200 epoch eğitimi |

**Tablo EK-6.2: Crop Modu Yelek Model Seçim Süreci**

Crop modu için yelek modeli seçiminde `compare_vest.py` betiği çalıştırılmıştır. Karşılaştırma; mAP@0.5, Precision, Recall ve F1 metrikleri temel alınarak adaylar arasında gerçekleştirilmiş, `cropvest_agent_final_best.pt` modeli en yüksek bütünleşik performansı sergileyerek nihai kullanım için seçilmiştir.

**Not:** Tüm model dosyaları ağırlık boyutları nedeniyle git deposuna dahil edilmemiştir. Yeniden eğitim için Roboflow üzerinden erişilebilen veri setleri ve standart YOLOv8 eğitim betikleri kullanılabilir.
