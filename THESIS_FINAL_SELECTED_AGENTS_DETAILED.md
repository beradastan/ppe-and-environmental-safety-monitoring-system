# Final Seçilen Ajanlar - Tez İçin Detaylı Açıklama

Bu doküman, projede **net olarak final seçilen ajanları** ve bu ajanların tezde nasıl anlatılması gerektiğini detaylı şekilde açıklar. Buradaki amaç, model listesini yalnızca dosya adı olarak vermek değil; her ajanın sistemdeki rolünü, neden seçildiğini, hangi modda çalıştığını, hangi veri ve eğitim yaklaşımıyla üretildiğini ve tezde hangi bilimsel gerekçeyle savunulacağını netleştirmektir.

Final sistem, YOLOv8 tabanlı çok ajanlı bir PPE ihlal tespit mimarisidir. Sistem iki farklı kullanım modunu destekler:

1. **Scene mode:** PPE ajanları tüm video karesi üzerinde çalışır.
2. **Crop mode:** Önce kişi tespit/takip edilir, sonra PPE ajanları kişi veya kişi alt-bölgeleri üzerinde çalışır.

Bu iki modun karşılaştırılması tezdeki ana deneysel omurgayı oluşturur.

---

## 1. Final Ajan Listesi

| Model | Kullanım | Sistem içindeki görevi |
|---|---|---|
| `person_agent_scene_vinayakstyle_best.pt` | Kişi tespiti, her iki mod | Kişileri bulur ve takip sistemine temel sağlar |
| `bera/fire_smoke_other_agent_final_best.pt` | Yangın/duman, her iki mod | Fire/smoke/other sahne olaylarını algılar |
| `bera/crophelmet_agent_final_best.pt` | Baret, crop modu | Kişi crop/head bölgesinden Hardhat/NO-Hardhat kararı üretir |
| `bera/cropvest_agent_final_best.pt` | Yelek, crop modu | Kişi torso crop'ından Safety Vest/NO-Safety Vest kararı üretir |
| `bera/cropmask_agent_final_best.pt` | Maske, crop modu | Kişi/face crop'ından Mask/NO-Mask kararı üretir |
| `vinayak_trained_byBera/helmet_agent_final_best.pt` | Baret, scene modu | Tam karede Hardhat/NO-Hardhat tespiti yapar |
| `vinayak_trained_byBera/vest_agent_final_best.pt` | Yelek, scene modu | Tam karede Safety Vest/NO-Safety Vest tespiti yapar |
| `mask_agent_scene_200ep_yolov8m_best.pt` | Maske, scene modu | Tam karede Mask/NO-Mask tespiti yapar |

Bu seçimle sistem iki ayrı PPE tespit yaklaşımını aynı altyapı üzerinde karşılaştırabilir hale gelmiştir:

| Bileşen | Scene mode | Crop mode |
|---|---|---|
| Person | Ortak | Ortak |
| Fire/smoke | Ortak scene ajanı | Ortak scene ajanı |
| Helmet | Scene helmet ajanı | Crop helmet ajanı |
| Vest | Scene vest ajanı | Crop vest ajanı |
| Mask | Scene mask ajanı | Crop mask ajanı |

---

## 2. Neden Çok Ajanlı Mimari Seçildi?

Bu projede tek bir büyük PPE modeli yerine birden fazla uzman ajan kullanılmıştır. Bunun temel sebebi, PPE ihlal tespit probleminin tek tip bir nesne algılama problemi olmamasıdır. Kişi, baret, yelek, maske ve yangın/duman farklı görsel ölçeklere, farklı konum özelliklerine ve farklı hata davranışlarına sahiptir.

Örneğin:

| Görev | Görsel özellik | Neden ayrı ajan mantıklı? |
|---|---|---|
| Person | Büyük nesne, tüm vücut | Takip ve kişi bazlı event üretimi için temel gereklidir |
| Helmet | Küçük nesne, baş bölgesi | Crop ile baş bölgesine odaklanmak avantaj sağlayabilir |
| Vest | Gövde bölgesi, orta/büyük nesne | Torso bağlamı önemlidir; scene ve crop karşılaştırmaya uygundur |
| Mask | Küçük nesne, yüz bölgesi | Hem crop hem scene yaklaşımı test edilmeye uygundur |
| Fire/smoke | Kişiye bağlı değil, sahne olayı | Tam kare bağlamı gerekir; crop kullanmak mantıklı değildir |

Çok ajanlı yapı sayesinde her problem için farklı eğitim stratejisi uygulanabilmiştir. Ayrıca hata analizi daha okunabilir hale gelmiştir; örneğin sistemin helmet hatası mı, person assignment hatası mı, yoksa temporal voting kaynaklı bir hata mı yaptığı ayrıştırılabilmektedir.

Tezde kullanılabilecek ifade:

> Bu çalışmada tek bir monolitik PPE modeli yerine görev bazlı uzman ajanlardan oluşan modüler bir mimari tercih edilmiştir. Böylece kişi tespiti, PPE sınıflandırması ve yangın/duman algılama görevleri birbirinden ayrılmış; her görev kendi veri seti, eğitim stratejisi ve çalışma modu ile optimize edilmiştir. Bu yaklaşım, hem scene-based hem crop-based kullanım modlarının aynı altyapı üzerinde karşılaştırılmasına olanak sağlamıştır.

---

## 3. Ortak Ajanlar

## 3.1 Person Agent - `person_agent_scene_vinayakstyle_best.pt`

### Sistemdeki rolü

Person agent, sistemin kişi merkezli çalışmasını sağlayan temel ajandır. Her iki modda da ortaktır. Scene mode'da PPE tespitlerinin kişilere atanması için kişi kutularını üretir. Crop mode'da ise PPE ajanlarına verilecek kişi crop'larının çıkarılmasını sağlar.

Bu ajan olmadan sistem yalnızca sahnedeki nesneleri bulabilir; fakat ihlalin hangi kişiye ait olduğunu güvenilir şekilde raporlayamaz.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Tam video karesi |
| Çıktı | `Person` bounding box'ları |
| Takip | ByteTrack ile `track_id` üretimi |
| Kullanım | Scene ve crop modda ortak |

Person agent çıktısı daha sonra şu işlemler için kullanılır:

1. Her kişiye benzersiz takip kimliği atanır.
2. Scene mode'da PPE tespitleri kişi kutularına atanır.
3. Crop mode'da kişi/head/torso/face crop'ları çıkarılır.
4. Temporal voting her `track_id` için ayrı yürütülür.
5. Event manager kişi bazlı ihlal üretir.

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `person_agent_scene_vinayakstyle_best.pt` |
| Base model | `yolov8s.pt` |
| Veri seti | Construction Site Safety/Vinayak tarzı person dataset |
| Sınıf sayısı | 1 |
| Sınıf | `Person` |
| Train image | 2506 |
| Valid image | 84 |
| Test image | 59 |
| Epoch hedefi | 100 |
| Erken durma | 68. epoch |
| Best epoch | 48 |
| Image size | 640 |
| Batch | 16 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Patience | 20 |

Validation sonucu:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.903 | 0.819 | 0.867 | 0.595 |

### Neden seçildi?

Bu model, projede kendi eğitilmiş person agent ihtiyacını karşılamak için seçildi. Daha önce Vinayak kaynaklı person/person-anchor yaklaşımıyla karşılaştırıldı. Dedicated person agent her videoda açık üstünlük göstermese de sistemin kendi eğitilmiş ajanlardan oluşması hedefi için uygun bulundu.

Video bazlı karşılaştırma özetinde dedicated person agent:

| Video | Gözlem |
|---|---|
| `nohat_test` | Vinayak'a göre daha az kişi tespit etti |
| `novest_test` | Vinayak ile benzer kişi sayısı verdi |
| `noppe_test` | Biraz daha fazla kişi tespit etti |
| `mask_test` | Benzer kişi sayısı verdi |

Son karar:

> Dedicated person agent belirgin bir pratik üstünlük sağlamasa da, kendi eğitilmiş modüler sistem şartını karşıladığı ve crop-based pipeline'ın zorunlu temelini oluşturduğu için final sistemde ortak person ajanı olarak kullanılmıştır.

### Tez için yorum

Person agent, PPE tespitinden bağımsız olarak sistemin kişi merkezli olay üretmesini sağlayan ana bileşendir. Crop-based yöntemde PPE ajanlarının doğru kişi bölgesine uygulanabilmesi tamamen bu ajanın ürettiği bounding box'lara bağlıdır. Scene-based yöntemde ise PPE tespitlerinin kişiye atanması için person kutuları referans olarak kullanılır. Bu nedenle person agent, her iki yaklaşımın da ortak altyapısıdır.

---

## 3.2 Fire/Smoke Agent - `bera/fire_smoke_other_agent_final_best.pt`

### Sistemdeki rolü

Fire/smoke agent, yangın ve duman gibi kişi üzerinde taşınmayan sahne olaylarını tespit eder. Bu ajan her iki modda da ortaktır ve crop-based PPE yapısından bağımsız şekilde tam kare üzerinde çalışır.

Fire/smoke tespiti PPE tespitinden farklıdır çünkü:

1. Yangın veya duman bir kişiye bağlı değildir.
2. Sahnenin herhangi bir bölgesinde ortaya çıkabilir.
3. Kişi crop'ı içine almak anlamlı değildir.
4. Tam sahne bağlamı korunmalıdır.

Bu yüzden fire/smoke ajanının crop-based sistemde bile scene-based çalışması mimari olarak doğrudur.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Tam video karesi |
| Çıktı | `fire`, `smoke`, `other` tespitleri |
| Mod | Her iki sistemde scene-based |
| Event etkisi | `fire_detected` ihlal imzasını etkiler |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `fire_smoke_other_agent_final_best.pt` |
| Base model | `yolov8m.pt` |
| Veri seti | Costi fire/smoke/other dataset |
| Sınıflar | `fire`, `other`, `smoke` |
| Train image | 12813 |
| Valid image | 6068 |
| Test image | 2237 |
| Epoch hedefi | 100 |
| Image size | 640 |
| Batch | Başlangıç 64, OOM sonrası etkili batch 32 |
| Optimizer | `auto`, pratikte MuSGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Patience | 30 |
| close_mosaic | 15 |
| Seed | 0 |

Final test sonucu:

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 2237 | 5430 | 0.812 | 0.768 | 0.837 | 0.568 |

### Neden seçildi?

Yangın/duman için önce ayrı `fire_agent_final` ve `smoke_agent_final` modelleri denenmişti. Ancak bu iki ayrı model özellikle smoke tarafında zayıf kaldı. Birleşik `fire/smoke/other` modeli daha dengeli sonuç verdi.

| Model | Precision | Recall | mAP50 | mAP50-95 | Karar |
|---|---:|---:|---:|---:|---|
| Ayrı fire modeli | 0.726 | 0.649 | 0.724 | 0.363 | Final için zayıf |
| Ayrı smoke modeli | 0.544 | 0.455 | 0.485 | 0.222 | Final için zayıf |
| Birleşik fire/smoke/other | 0.812 | 0.768 | 0.837 | 0.568 | Final seçildi |

Son karar:

> Yangın ve duman algılama için ayrı modeller yerine tek bir `fire/smoke/other` modeli kullanılmıştır. Bu model hem daha yüksek test başarısı vermiş hem de tek ajanla sahne olaylarını takip etmeyi sağlamıştır.

### Tez için yorum

Fire/smoke ajanı, PPE ajanlarından farklı olarak kişi merkezli değil sahne merkezli çalışır. Bu nedenle scene/crop karşılaştırmasında ortak bileşen olarak tutulmuştur. Böylece iki sistem arasındaki karşılaştırma helmet, vest ve mask ajanlarının çalışma biçimine odaklanırken, yangın/duman algılama sabit bir yardımcı güvenlik bileşeni olarak korunmuştur.

---

## 4. Crop Mode Ajanları

Crop mode'da önce person agent kişi kutularını üretir. Daha sonra her kişi için ilgili crop alanı çıkarılır ve PPE ajanlarına verilir.

Genel akış:

```text
Frame
  |
  v
Person Agent + ByteTrack
  |
  v
Tracked person bbox
  |
  +--> head/person crop  --> Crop Helmet Agent
  +--> torso crop        --> Crop Vest Agent
  +--> face/person crop  --> Crop Mask Agent
  |
  v
Temporal voting
  |
  v
PersonEventManager
```

Crop mode'un temel amacı, PPE ajanlarının tüm sahne yerine kişiyle ilişkili daha odaklı görsel alan üzerinde çalışmasıdır.

---

## 4.1 Crop Helmet Agent - `bera/crophelmet_agent_final_best.pt`

### Sistemdeki rolü

Crop helmet agent, crop-based modda baret durumunu belirler. Person agent tarafından bulunan kişi kutusundan baş/üst bölge çıkarılır ve bu crop, helmet ajanına verilir.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Kişi/head crop |
| Çıktı | `Hardhat` veya `NO-Hardhat` |
| Mod | Crop mode |
| Event etkisi | `helmet_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `crophelmet_agent_final_best.pt` |
| Deney adı | `crophelmet_agent_200_retry` |
| Base model | `yolov8s.pt` |
| Veri seti | `crophelmet_dataset` |
| Train image | 4057 |
| Valid image | 112 |
| Test image | 113 |
| Sınıflar | `Hardhat`, `NO-Hardhat` |
| Epoch hedefi | 140 |
| Gerçek tamamlanan epoch | 135 |
| Best epoch | 110 |
| Image size | 640 |
| Batch | 16 |
| Patience | 25 |
| Learning rate | 0.001 |
| Optimizer | `auto` |
| Dropout | 0.025 |
| Seed | 88 |
| Eğitim süresi | 3.140 saat |

Validation sonucu:

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 112 | 162 | 0.880 | 0.856 | 0.867 | 0.571 |

Test sonucu:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 113 | 203 | 0.917 | 0.909 | 0.923 | 0.627 |
| Hardhat | 84 | 168 | 0.922 | 0.935 | 0.961 | 0.685 |
| NO-Hardhat | 30 | 35 | 0.912 | 0.884 | 0.885 | 0.568 |

### Neden seçildi?

Helmet sınıfı küçük ve baş bölgesine bağlı bir nesnedir. Tam sahne üzerinde helmet/no-helmet ayrımı yapmak kalabalık ve karmaşık görüntülerde zorlaşabilir. Crop yaklaşımı, modelin arka plan yerine baş/üst vücut bölgesine odaklanmasını sağlar.

Scene helmet modelinde `NO-Hardhat` sınıfı daha zayıf kalırken crop helmet modelinde `NO-Hardhat` için daha dengeli sonuç elde edilmiştir.

| Model | NO-Hardhat Recall | NO-Hardhat mAP50 | NO-Hardhat mAP50-95 |
|---|---:|---:|---:|
| Scene helmet | 0.610 | 0.553 | 0.336 |
| Crop helmet | 0.884 | 0.885 | 0.568 |

Son karar:

> Crop mode için baret ajanı olarak `crophelmet_agent_final_best.pt` seçilmiştir. Bu model özellikle baret yokluğu sınıfında daha dengeli başarı verdiği için crop-based yaklaşımın helmet kategorisindeki temel temsilcisidir.

### Tez için yorum

Crop helmet ajanı, crop-based mimarinin en güçlü gerekçelerinden biridir. Helmet küçük bir nesne olduğu için tam sahnede arka plan, kişi kalabalığı ve uzaklık etkileri hatayı artırabilir. Kişi/head crop'ı kullanıldığında model daha dar ve anlamlı bir görsel alanda karar verir. Bu nedenle crop helmet modelinin `NO-Hardhat` sınıfındaki başarısı, crop-based yaklaşımın küçük ve lokal PPE öğelerinde avantaj sağlayabileceğini göstermektedir.

---

## 4.2 Crop Vest Agent - `bera/cropvest_agent_final_best.pt`

### Sistemdeki rolü

Crop vest agent, crop-based modda güvenlik yeleği durumunu belirler. Person agent tarafından bulunan kişi kutusundan gövde/torso bölgesi çıkarılır ve model bu bölge üzerinde `Safety Vest` veya `NO-Safety Vest` tahmini yapar.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Kişi torso crop'ı |
| Çıktı | `Safety Vest` veya `NO-Safety Vest` |
| Mod | Crop mode |
| Event etkisi | `vest_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `cropvest_agent_final_best.pt` |
| Deney adı | `cropvest_agent_final` |
| Base model | `yolov8s.pt` |
| Veri seti | `cropvest_dataset` |
| Train image | 4879 |
| Valid image | 111 |
| Test image | 118 |
| Sınıflar | `Safety Vest`, `NO-Safety Vest` |
| Epoch hedefi | 100 |
| Gerçek tamamlanan epoch | 100 |
| Image size | 640 |
| Batch | 16 |
| Patience | 30 |
| Learning rate | 0.001 yazıldı, `optimizer=auto` tarafından override edildi |
| Optimizer | AdamW, lr=0.001667, momentum=0.9 |
| Weight decay | 0.0005 |
| Dropout | 0.025 |
| Seed | 88 |
| Eğitim süresi | 2.680 saat |

Validation sonucu:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 111 | 149 | 0.946 | 0.866 | 0.920 | 0.652 |
| Safety Vest | 36 | 38 | 0.972 | 0.947 | 0.980 | 0.751 |
| NO-Safety Vest | 79 | 111 | 0.920 | 0.784 | 0.859 | 0.552 |

### Neden 100 epoch yeterli görüldü?

Crop vest eğitimi 100 epoch tamamlanmıştır. Eğitim boyunca metrikler özellikle 60. epoch sonrası anlamlı şekilde iyileşmiş, 80-100 epoch aralığında ise artış daha sınırlı hale gelmiştir.

Özet gözlem:

| Epoch aralığı | Gözlem |
|---|---|
| 1-30 | Model temel ayrımı öğrenmeye başladı |
| 30-60 | mAP50 ve mAP50-95 belirgin yükseldi |
| 60-80 | İyileşme devam etti |
| 80-100 | Plato etkisi başladı, küçük artışlar görüldü |

Bu yüzden 100 epoch, final crop vest modeli için yeterli ve savunulabilir bir eğitim süresi olarak kabul edilmiştir.

### Neden seçildi?

Tezin ana amacı artık crop vs scene karşılaştırması olduğu için crop vest ajanı final crop set içinde tutulmuştur. Scene vest modeli güçlü olsa da, crop vest modeli crop-based yaklaşımın yelek kategorisindeki karşılığıdır.

Son karar:

> Crop mode için yelek ajanı olarak `cropvest_agent_final_best.pt` seçilmiştir. Bu model, kişi torso crop'ı üzerinde eğitim aldığı için crop-based sistemde scene vest modelinin doğrudan karşılaştırma alternatifi olarak kullanılacaktır.

### Tez için yorum

Yelek, helmet ve maskeye göre daha büyük bir görsel alana yayılır. Bu nedenle scene-based model de gövde bağlamını kullanarak güçlü sonuç verebilir. Buna rağmen crop vest ajanı, kişi torso bölgesine odaklanarak arka plan etkisini azaltır. Bu kategori, crop ve scene yaklaşımlarının birbirine en yakın sonuç verebileceği PPE sınıfı olarak yorumlanabilir.

---

## 4.3 Crop Mask Agent - `bera/cropmask_agent_final_best.pt`

### Sistemdeki rolü

Crop mask agent, crop-based modda maske durumunu belirler. Maske küçük ve yüz bölgesine bağlı bir PPE öğesi olduğu için crop-based kullanım mantığına uygundur.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Kişi/face crop'ı |
| Çıktı | `Mask` veya `NO-Mask` |
| Mod | Crop mode |
| Event etkisi | `mask_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `cropmask_agent_final_best.pt` |
| Deney adı | `cropmask_agent_final` |
| Base model | `yolov8s.pt` |
| Sınıflar | `Mask`, `NO-Mask` |
| Epoch hedefi | 200 |
| Gerçek tamamlanan epoch | 130 |
| Best epoch | 105 |
| Image size | 640 |
| Batch | 16 |
| Patience | 25 |
| Learning rate | 0.001 |
| Optimizer | `auto` |
| Dropout | 0.025 |
| Seed | 88 |
| Eğitim süresi | 3.636 saat |

Validation sonucu:

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 91 | 111 | 0.982 | 0.861 | 0.899 | 0.622 |

Test sonucu:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 96 | 175 | 0.969 | 0.905 | 0.949 | 0.599 |
| Mask | 29 | 41 | 0.964 | 0.878 | 0.934 | 0.641 |
| NO-Mask | 68 | 134 | 0.974 | 0.933 | 0.963 | 0.557 |

### Neden seçildi?

Mask sınıfı küçük ve lokal bir nesnedir. Bu nedenle crop-based yaklaşımın avantaj sağlayabileceği PPE öğelerinden biridir. Crop mask modeli test split üzerinde yüksek precision, recall ve mAP50 değerleri üretmiştir.

Son karar:

> Crop mode için maske ajanı olarak `cropmask_agent_final_best.pt` seçilmiştir. Model, mask/no-mask ayrımında yüksek test başarısı verdiği ve crop-based sistemin maske bileşenini temsil ettiği için final sete dahil edilmiştir.

### Tez için yorum

Maske tespiti küçük nesne, yüz görünürlüğü, açı ve uzaklık gibi faktörlerden etkilenir. Crop-based yaklaşım, modelin tüm sahne yerine kişi/face bölgesine odaklanmasını sağladığı için maske tespitinde anlamlı bir stratejidir. Crop mask ajanının güçlü test sonucu, crop-based PPE algılamanın küçük ve lokal ekipmanlarda uygulanabilir olduğunu destekler.

---

## 5. Scene Mode Ajanları

Scene mode'da PPE ajanları tüm frame üzerinde çalışır. Daha sonra bulunan PPE kutuları person kutularına atanır.

Genel akış:

```text
Frame
  |
  +--> Person Agent + ByteTrack
  +--> Scene Helmet Agent
  +--> Scene Vest Agent
  +--> Scene Mask Agent
  |
  v
PPE bbox -> person bbox assignment
  |
  v
Temporal voting
  |
  v
PersonEventManager
```

Atama mantığı:

```text
inside_frac = area(ppe_bbox ∩ person_bbox) / area(ppe_bbox)
```

Kural:

| Koşul | Karar |
|---|---|
| `inside_frac >= 0.40` | PPE tespiti kişiye atanabilir |
| `inside_frac < 0.40` | PPE tespiti o kişiye ait sayılmaz |
| Birden fazla uygun tespit varsa | En yüksek confidence seçilir |

---

## 5.1 Scene Helmet Agent - `vinayak_trained_byBera/helmet_agent_final_best.pt`

### Sistemdeki rolü

Scene helmet agent, scene mode'da tam kare üzerinde baret tespiti yapar. Çıktıları person kutularına atanarak her kişi için helmet durumu belirlenir.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Tam video karesi |
| Çıktı | `Hardhat`, `NO-Hardhat` bounding box'ları |
| Mod | Scene mode |
| Atama | `inside_frac` ile kişi kutusuna bağlanır |
| Event etkisi | `helmet_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `helmet_agent_final_best.pt` |
| Deney adı | `helmet_agent_normal_final` |
| Base model | `yolov8s.pt` |
| Veri seti | `helmet_dataset` |
| Train image | 2605 |
| Valid image | 114 |
| Test image | 82 |
| Sınıflar | `Hardhat`, `NO-Hardhat` |
| Epoch hedefi | 200 |
| Gerçek tamamlanan epoch | 109 |
| Best epoch | 84 |
| Image size | 640 |
| Batch | 16 |
| Patience | 25 |
| Learning rate | 0.001 |
| Optimizer | `auto` |
| Dropout | 0.025 |
| Seed | 88 |
| Eğitim süresi | 1.749 saat |

Validation sonucu:

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 114 | 148 | 0.927 | 0.778 | 0.846 | 0.542 |

Test sonucu:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 82 | 151 | 0.893 | 0.746 | 0.760 | 0.481 |
| Hardhat | 30 | 110 | 0.989 | 0.882 | 0.967 | 0.626 |
| NO-Hardhat | 25 | 41 | 0.797 | 0.610 | 0.553 | 0.336 |

### Neden seçildi?

Scene mode için scene-trained bir helmet ajanına ihtiyaç vardı. Bu model, tam kare üzerinde eğitilmiş helmet ajanı olduğu için scene-based karşılaştırmanın helmet bileşeni olarak seçildi.

Crop helmet modelinin statik metrikleri daha güçlü olsa da bu durum, iki yaklaşımın karşılaştırılması açısından avantajdır. Scene helmet modeli, tam kare bağlamıyla çalışan yöntemi temsil ederken crop helmet modeli kişi/head crop yaklaşımını temsil eder.

Son karar:

> Scene mode için baret ajanı olarak `helmet_agent_final_best.pt` seçilmiştir. Model, tam kare üzerinde helmet/no-helmet tespiti yapar ve scene-based yaklaşımın helmet bileşenini temsil eder.

### Tez için yorum

Scene helmet ajanı, tam sahne üzerinde PPE tespiti yapmanın avantaj ve zorluklarını temsil eder. Tam kare bağlamı korunur; ancak küçük bir nesne olan baret için kalabalık sahnelerde kişi-PPE eşleştirme ve `NO-Hardhat` sınıfı daha zor hale gelir. Bu nedenle scene helmet ve crop helmet sonuçları, tezde scene/crop farkını göstermek için kritik bir karşılaştırma oluşturur.

---

## 5.2 Scene Vest Agent - `vinayak_trained_byBera/vest_agent_final_best.pt`

### Sistemdeki rolü

Scene vest agent, scene mode'da tam kare üzerinde güvenlik yeleği tespiti yapar. Tespit edilen `Safety Vest` ve `NO-Safety Vest` kutuları person kutularına atanır.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Tam video karesi |
| Çıktı | `Safety Vest`, `NO-Safety Vest` bounding box'ları |
| Mod | Scene mode |
| Atama | `inside_frac` ile kişi kutusuna bağlanır |
| Event etkisi | `vest_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `vest_agent_final_best.pt` |
| Deney adı | `vest_agent_normal_final` |
| Base model | `yolov8s.pt` |
| Veri seti | `vest_dataset` |
| Train image | 2605 |
| Valid image | 114 |
| Test image | 82 |
| Sınıflar | `Safety Vest`, `NO-Safety Vest` |
| Epoch hedefi | 200 |
| Gerçek tamamlanan epoch | 124 |
| Best epoch | 99 |
| Image size | 640 |
| Batch | 16 |
| Patience | 25 |
| Learning rate | 0.001 |
| Optimizer | `auto` |
| Dropout | 0.025 |
| Seed | 88 |
| Eğitim süresi | 1.969 saat |

Validation sonucu:

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 114 | 147 | 0.959 | 0.870 | 0.904 | 0.637 |

Test sonucu:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 82 | 151 | 0.966 | 0.824 | 0.900 | 0.598 |
| Safety Vest | 22 | 61 | 0.962 | 0.902 | 0.950 | 0.663 |
| NO-Safety Vest | 36 | 90 | 0.971 | 0.747 | 0.849 | 0.532 |

### Neden seçildi?

Scene vest modeli, final testte güçlü ve dengeli sonuç verdi. Özellikle önceki 30 epoch vest modeline göre belirgin iyileşme sağladı.

Önceki modele göre iyileşme:

| Metrik | 30 epoch vest | Final scene vest |
|---|---:|---:|
| Precision | 0.882 | 0.966 |
| Recall | 0.730 | 0.824 |
| mAP50 | 0.819 | 0.900 |
| mAP50-95 | 0.514 | 0.598 |

Son karar:

> Scene mode için yelek ajanı olarak `vest_agent_final_best.pt` seçilmiştir. Model, tam kare üzerinde güçlü test başarısı verdiği için scene-based yelek tespitinin final temsilcisi olarak kullanılmıştır.

### Tez için yorum

Yelek, baret ve maskeye göre daha büyük bir alan kapladığı için scene-based yaklaşımda daha başarılı çalışabilir. Scene vest ajanının yüksek precision ve mAP değerleri, tam sahne bağlamının yelek tespiti için faydalı olabileceğini göstermektedir. Crop vest modeli ile karşılaştırıldığında, bu kategori crop ve scene yaklaşımlarının en dengeli rekabet ettiği PPE sınıfıdır.

---

## 5.3 Scene Mask Agent - `mask_agent_scene_200ep_yolov8m_best.pt`

### Sistemdeki rolü

Scene mask agent, scene mode'da tam kare üzerinde maske tespiti yapar. Model `Mask` ve `NO-Mask` nesnelerini tüm sahnede bulur; ardından bu tespitler person kutularına atanır.

### Girdi ve çıktı

| Alan | Açıklama |
|---|---|
| Girdi | Tam video karesi |
| Çıktı | `Mask`, `NO-Mask` bounding box'ları |
| Mod | Scene mode |
| Atama | `inside_frac` ile kişi kutusuna bağlanır |
| Event etkisi | `mask_violation` |

### Eğitim özeti

| Özellik | Değer |
|---|---|
| Model | `mask_agent_scene_200ep_yolov8m_best.pt` |
| Deney adı | `mask_agent_scene_200ep_yolov8m` |
| Base model | `yolov8m.pt` |
| Veri seti | `mask_dataset` |
| Train image | 2035 |
| Valid image | 61 |
| Test image | 44 |
| Sınıflar | `Mask`, `NO-Mask` |
| Epoch hedefi | 200 |
| Image size | 960 |
| Batch | 16 |
| Patience | 30 |
| Optimizer | AdamW |

### Neden YOLOv8m seçildi?

Önceki scene mask eğitiminde `yolov8s.pt` ile güçlü sonuç alınmıştı. Ancak maske küçük bir nesne olduğu ve yüz bölgesi ayrıntı gerektirdiği için final scene mask modelinde daha büyük kapasiteye sahip `yolov8m.pt` backbone'u tercih edildi.

Bu seçim şu nedenlerle savunulabilir:

1. Maske küçük ve ayrıntı gerektiren bir nesnedir.
2. Scene mode'da model tüm kare üzerinde çalıştığı için küçük objeler daha zor algılanır.
3. `imgsz=960` kullanılması küçük nesne çözünürlüğünü artırır.
4. `yolov8m` modeli `yolov8s` modeline göre daha yüksek temsil kapasitesine sahiptir.
5. Final scene mask ajanı, scene-based sistemin maske bileşenini daha güçlü temsil eder.

Önceki `yolov8s` referans sonucu:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `mask_agent_scene_200ep_best.pt` | 0.970 | 0.892 | 0.911 | 0.649 |

Class-level referans:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Mask | 0.979 | 0.905 | 0.905 | 0.710 |
| NO-Mask | 0.962 | 0.878 | 0.918 | 0.589 |

Son karar:

> Scene mode için maske ajanı olarak `mask_agent_scene_200ep_yolov8m_best.pt` seçilmiştir. Model, scene-based maske tespitinde daha yüksek model kapasitesi ve daha büyük görüntü boyutu ile final temsilci olarak kullanılmıştır.

### Tez için güvenli not

Bu modelin final `.pt` dosyası arşivlenmiştir. Tezde sayısal sonuç tablosu verilecekse, Drive run klasöründen `results.csv`, `args.yaml` ve varsa test summary dosyası eklenmelidir. Eğer bu dosya eklenmeden yazılacaksa, tezde kesin metrik olarak önceki `yolov8s` referans tablosu verilip `yolov8m` modelinin final seçilen checkpoint olduğu açıkça belirtilmelidir.

### Tez için yorum

Scene mask ajanı, tam kare üzerinde maske/no-mask tespiti yaparak scene-based sistemin maske bileşenini oluşturur. Maske küçük ve yüz bölgesine bağlı bir nesne olduğu için model kapasitesi ve giriş çözünürlüğü önemlidir. Bu nedenle final seçimde `yolov8m` ve `imgsz=960` kombinasyonu tercih edilmiştir. Crop mask ajanı ile karşılaştırıldığında, scene mask ajanı tam sahne bağlamını korurken crop mask ajanı daha odaklı yüz/kişi bölgesi üzerinde karar verir.

---

## 6. Veri Seti Hazırlama Süreci

Final ajanların eğitimi tek bir genel veri setiyle değil, ajan görevine göre ayrılmış veri setleriyle yapılmıştır. Bu tercih, çok ajanlı mimarinin temelidir. Her ajan yalnızca kendi karar vermesi gereken sınıfları görür; böylece model karmaşıklığı azalır ve hata analizi daha açık yapılabilir.

### 6.1 Ana kaynak veri seti

PPE ajanları için ana kaynak veri seti, Construction Site Safety sınıf yapısına sahip YOLO formatındaki veri setidir. Bu veri seti Vinayakmane PPE modelinin kullandığı sınıf yapısıyla uyumludur.

Kaynak sınıf listesi:

| Kaynak class id | Sınıf |
|---:|---|
| 0 | `Hardhat` |
| 1 | `Mask` |
| 2 | `NO-Hardhat` |
| 3 | `NO-Mask` |
| 4 | `NO-Safety Vest` |
| 5 | `Person` |
| 6 | `Safety Cone` |
| 7 | `Safety Vest` |
| 8 | `machinery` |
| 9 | `vehicle` |

Bu çalışmada bütün sınıflar tek modelde kullanılmamıştır. Bunun yerine her ajan için gerekli sınıflar filtrelenmiştir.

### 6.2 Scene dataset filtreleme mantığı

Scene tabanlı ajanlarda görüntünün tamamı korunmuş, yalnızca etiket sınıfları filtrelenmiştir.

| Hedef dataset | Kaynak sınıflar | Hedef sınıflar | Amaç |
|---|---|---|---|
| `helmet_dataset` | `Hardhat`, `NO-Hardhat` | `0: Hardhat`, `1: NO-Hardhat` | Scene helmet eğitimi |
| `vest_dataset` | `Safety Vest`, `NO-Safety Vest` | `0: Safety Vest`, `1: NO-Safety Vest` | Scene vest eğitimi |
| `mask_dataset` | `Mask`, `NO-Mask` | `0: Mask`, `1: NO-Mask` | Scene mask eğitimi |
| `person_dataset` | `Person` | `0: Person` | Person agent eğitimi |

Scene dataset üretiminde kullanılan temel işlem:

```text
1. Kaynak YOLO label dosyası okunur.
2. Sadece hedef ajanın ihtiyaç duyduğu class id'leri tutulur.
3. Tutulan class id'leri yeni ajan class id'lerine remap edilir.
4. Hedef sınıf kalmayan görüntüler background olarak tutulmayacaksa çıkarılır.
5. Yeni data.yaml dosyası hedef sınıf isimleriyle yazılır.
```

Örnek remap:

```text
mask_dataset:
  kaynak 1 Mask    -> hedef 0 Mask
  kaynak 3 NO-Mask -> hedef 1 NO-Mask

person_dataset:
  kaynak 5 Person -> hedef 0 Person
```

Bu işlem `scripts/filter_yolo_classes.py`, `scripts/prepare_scene_mask_dataset.py` ve `scripts/prepare_scene_person_dataset.py` mantığıyla yapılmıştır.

### 6.3 Final veri seti sayıları

| Dataset | Kullanım | Train | Valid | Test | Sınıflar |
|---|---:|---:|---:|---:|---|
| `helmet_dataset` | Scene helmet | 2605 | 114 | 82 | `Hardhat`, `NO-Hardhat` |
| `vest_dataset` | Scene vest | 2605 | 114 | 82 | `Safety Vest`, `NO-Safety Vest` |
| `mask_dataset` | Scene mask | 2035 | 61 | 44 | `Mask`, `NO-Mask` |
| `person_dataset` | Person | 2506 | 84 | 59 | `Person` |
| `crophelmet_dataset` | Crop helmet | 4057 | 112 | 113 | `Hardhat`, `NO-Hardhat` |
| `cropvest_dataset` | Crop vest | 4879 | 111 | 118 | `Safety Vest`, `NO-Safety Vest` |
| `cropmask_dataset` | Crop mask | 4401 | 91 | 96 | `Mask`, `NO-Mask` |
| `firesmoke_dataset` | Fire/smoke | 12813 | 6068 | 2237 | `fire`, `other`, `smoke` |

### 6.4 Crop dataset üretim mantığı

Crop tabanlı ajanlar için doğrudan orijinal tam görüntüler kullanılmamıştır. Önce kaynak scene dataset üzerinde kişi tespiti yapılmış, sonra her kişi için person crop üretilmiştir. PPE label'ları bu crop koordinat sistemine dönüştürülmüştür.

Crop dataset üretiminde kullanılan person modeli:

```text
models/pretrained/person/person_yolov8s-seg.pt
```

Ortak crop üretim ayarları:

| Parametre | Değer | Açıklama |
|---|---:|---|
| Person model confidence | 0.25 | Person kutusu üretmek için minimum güven |
| Person model image size | 640 | Person inference çözünürlüğü |
| Crop padding | 0.30 | Kişi kutusu genişlik/yüksekliğinin %30'u kadar genişletme |
| Crop clipping | Var | Crop görüntü sınırları dışına taşarsa kırpılır |
| Boş crop | Atılır | Geçersiz genişlik/yükseklik varsa kullanılmaz |
| Label dönüşümü | Var | Orijinal PPE bbox crop koordinatlarına normalize edilir |

PPE label'ının crop içine dahil edilmesi için kullanılan eşikler:

| Crop dataset | Minimum label IoA | Anlamı |
|---|---:|---|
| `crophelmet_dataset` | 0.60 | Helmet label alanının en az %60'ı crop içinde olmalı |
| `cropvest_dataset` | 0.60 | Vest label alanının en az %60'ı crop içinde olmalı |
| `cropmask_dataset` | 0.35 | Mask label alanının en az %35'i crop içinde olmalı |

Burada IoA şu şekilde hesaplanır:

```text
label_ioa = area(label_bbox ∩ crop_bbox) / area(label_bbox)
```

Bir label crop'a yeterince girmiyorsa o crop label'ı olarak kullanılmaz. Hiç label kalmayan crop'lar da eğitim setine eklenmez.

### 6.5 Fire/smoke dataset

Fire/smoke ajanı PPE veri setinden türetilmemiştir. Yangın ve duman problemi için ayrı bir `firesmoke_dataset` kullanılmıştır. Bu dataset `fire`, `other` ve `smoke` sınıflarından oluşur.

| Özellik | Değer |
|---|---|
| Dataset | `firesmoke_dataset` |
| Sınıflar | `fire`, `other`, `smoke` |
| Train image | 12813 |
| Valid image | 6068 |
| Test image | 2237 |
| Kullanım | Her iki modda ortak scene fire/smoke ajanı |

Fire/smoke datasetinin ayrı tutulmasının nedeni, yangın ve dumanın kişi üzerinde taşınan PPE nesneleri olmamasıdır. Bu görev sahne olayı olarak ele alınmıştır.

### 6.6 Tez için veri seti açıklama metni

> Bu çalışmada her ajan için görev odaklı ayrı veri setleri hazırlanmıştır. Helmet, vest, mask ve person ajanları için Construction Site Safety sınıf yapısına sahip YOLO formatındaki veri setinden ilgili sınıflar filtrelenmiş ve hedef class id'lerine yeniden eşlenmiştir. Scene-based ajanlarda tam görüntüler korunurken, crop-based ajanlarda önce kişi tespiti yapılarak kişi crop'ları oluşturulmuş ve ilgili PPE etiketleri crop koordinat sistemine dönüştürülmüştür. Bu sayede scene-based ve crop-based yaklaşımlar aynı temel veri kaynağından türetilmiş, fakat farklı görsel temsil biçimleriyle eğitilmiştir.

---

## 7. Crop Üretim Algoritması

Crop işlemi iki farklı bağlamda değerlendirilmelidir:

1. Eğitim veri seti üretiminde yapılan crop işlemi
2. Runtime/pipeline sırasında yapılan crop işlemi

Bu ayrım önemlidir çünkü eğitimde crop dataset üretmek için person detector ile kişi kutuları çıkarılmıştır; runtime sırasında ise canlı karede takip edilen kişi kutuları üzerinden crop alınır.

### 7.1 Eğitim veri seti için crop üretimi

Eğitim veri seti üretim algoritması:

```text
Girdi:
  - Orijinal frame
  - Orijinal YOLO label dosyası
  - Person detector çıktısı

Adımlar:
  1. Frame üzerinde person modeli çalıştırılır.
  2. Her person bbox için bbox %30 padding ile genişletilir.
  3. Genişletilmiş bbox görüntü sınırları içine clip edilir.
  4. Orijinal PPE label'ları okunur.
  5. Her PPE label için label_ioa hesaplanır.
  6. label_ioa eşik üzerindeyse label crop içine alınır.
  7. PPE bbox koordinatları crop koordinat sistemine dönüştürülür.
  8. Crop image ve yeni YOLO label dosyası kaydedilir.
  9. Hiç label içermeyen crop'lar atılır.
```

Kullanılan temel formüller:

```text
padding_x = person_width  * 0.30
padding_y = person_height * 0.30

crop_x1 = person_x1 - padding_x
crop_y1 = person_y1 - padding_y
crop_x2 = person_x2 + padding_x
crop_y2 = person_y2 + padding_y
```

Görüntü sınırı düzeltmesi:

```text
crop_x1 = max(0, crop_x1)
crop_y1 = max(0, crop_y1)
crop_x2 = min(image_width, crop_x2)
crop_y2 = min(image_height, crop_y2)
```

Label dahil etme formülü:

```text
label_ioa = area(label_bbox ∩ crop_bbox) / area(label_bbox)
```

YOLO label dönüşümü:

```text
new_x_center = ((clipped_x1 + clipped_x2) / 2) / crop_width
new_y_center = ((clipped_y1 + clipped_y2) / 2) / crop_height
new_width    = (clipped_x2 - clipped_x1) / crop_width
new_height   = (clipped_y2 - clipped_y1) / crop_height
```

Bu işlem sayesinde crop image ile label dosyası aynı koordinat sistemine taşınmıştır.

### 7.2 Crop datasetlere özel farklar

| Dataset | Kaynak dataset | Crop kaynağı | Label dahil eşiği |
|---|---|---|---:|
| `crophelmet_dataset` | `helmet_dataset` | Person bbox + %30 padding | 0.60 |
| `cropvest_dataset` | `vest_dataset` | Person bbox + %30 padding | 0.60 |
| `cropmask_dataset` | `mask_dataset` | Person bbox + %30 padding | 0.35 |

Mask için eşik daha düşüktür çünkü maske label'ları küçük olabilir ve yüz bölgesi kişi kutusunun kenarlarına yakın kalabilir. Bu nedenle maske label'ının crop içinde kalma koşulu helmet/vest'e göre daha esnek tutulmuştur.

### 7.3 Runtime crop kullanımı

Runtime sırasında crop işlemi eğitim verisi üretiminden farklı olarak takip edilen kişi kutuları üzerinden yapılır.

Genel runtime mantığı:

```text
1. Person agent frame üzerinde kişi bbox'larını üretir.
2. ByteTrack her kişiye track_id atar.
3. Her track_id için PPE'ye uygun crop bölgesi çıkarılır.
4. Crop ilgili PPE ajanına gönderilir.
5. PPE sonucu track_id geçmişine eklenir.
6. Temporal voting sonrası kişi durumu belirlenir.
```

Tezde güvenli şekilde anlatılabilecek runtime crop bölgeleri:

| PPE | Runtime crop fikri | Gerekçe |
|---|---|---|
| Helmet | Kişinin üst/head bölgesi | Baret baş bölgesinde bulunur |
| Vest | Kişinin torso/gövde bölgesi | Yelek gövde üzerinde bulunur |
| Mask | Kişinin üst/face bölgesi | Maske yüz bölgesinde bulunur |

Uygulama notlarında kullanılan pratik crop yaklaşımı:

| PPE | Kullanılan/planlanan crop bölgesi |
|---|---|
| Helmet | Kişi kutusunun üst yaklaşık %40'lık bölgesi |
| Vest | Kişi kutusunun yaklaşık %10-%90 dikey aralığı, yani torso ağırlıklı bölge |
| Mask | Kişi üst/face bölgesi veya person crop üzerinden mask ajanı |

Bu bölüm tezde şu şekilde ifade edilebilir:

> Crop-based modda PPE ajanları tam kare yerine takip edilen kişiye ait bölgesel görüntüler üzerinde çalıştırılmıştır. Helmet için kişinin baş/üst bölgesi, vest için gövde bölgesi, mask için ise yüz/üst kişi bölgesi kullanılmıştır. Eğitim veri setleri ise person detector ile üretilen kişi crop'ları üzerinden hazırlanmış; ilgili PPE etiketleri crop koordinat sistemine dönüştürülmüştür.

### 7.4 Crop yönteminin avantajı

Crop tabanlı yöntemin beklenen avantajları:

| Avantaj | Açıklama |
|---|---|
| Arka plan azalır | Model tüm sahne yerine kişi bölgesine odaklanır |
| Küçük nesneler büyür | Helmet ve mask gibi küçük PPE öğeleri crop içinde daha görünür hale gelir |
| Kişi bazlı karar kolaylaşır | Model çıktısı doğrudan ilgili `track_id` ile ilişkilidir |
| False association azalabilir | Scene mode'daki kişi-PPE atama belirsizliği azalır |

Dezavantajlar:

| Dezavantaj | Açıklama |
|---|---|
| Person hatasına bağımlılık | Kişi bbox yanlışsa crop da yanlıştır |
| Crop kesme riski | PPE bölgesi crop dışında kalabilir |
| Ek inference maliyeti | Her kişi için ayrı PPE inference gerekebilir |
| Çok kişi varsa FPS düşebilir | Kişi sayısıyla işlem sayısı artar |

---

## 8. Scene Assignment Algoritması

Scene mode'da PPE ajanları tam kare üzerinde çalışır. Bu nedenle model çıktısı doğrudan kişi bazlı değildir. Bir PPE tespitinin hangi kişiye ait olduğunu belirlemek için person bbox ile PPE bbox arasında geometrik atama yapılır.

### 8.1 Temel inside fraction formülü

Scene assignment için kullanılan temel ölçüt:

```text
inside_frac = area(ppe_bbox ∩ person_bbox) / area(ppe_bbox)
```

Bu oran, PPE kutusunun ne kadarının kişi kutusunun içinde kaldığını ölçer.

### 8.2 Atama eşiği

| Koşul | Karar |
|---|---|
| `inside_frac >= 0.40` | PPE tespiti kişiye atanabilir |
| `inside_frac < 0.40` | PPE tespiti o kişiye ait kabul edilmez |

Eşik olarak 0.40 kullanılması, PPE kutusunun tamamının kişi içinde olmasını şart koşmaz. Çünkü bbox'lar model hatası, hareket bulanıklığı veya perspektif nedeniyle tam örtüşmeyebilir. Ancak PPE kutusunun anlamlı bir kısmının kişi içinde olmasını şart koşar.

### 8.3 Kişi bazlı en iyi tespit seçimi

Her kişi için aynı kategoride birden fazla uygun PPE tespiti olabilir. Bu durumda en yüksek confidence değerine sahip tespit seçilir.

Pseudocode:

```text
for each tracked_person:
    best_detection = None

    for each ppe_detection:
        frac = area(ppe_bbox ∩ person_bbox) / area(ppe_bbox)

        if frac >= 0.40:
            if best_detection is None:
                best_detection = ppe_detection
            else if ppe_detection.conf > best_detection.conf:
                best_detection = ppe_detection

    if best_detection exists:
        assign best_detection.label to tracked_person
    else:
        assign "unknown"
```

Bu işlem helmet, vest ve mask tespitleri için ayrı ayrı uygulanır.

### 8.4 Scene assignment sonrası temporal voting

Scene assignment tek karelik karar üretir. Ancak tek karelik kararlar gürültülü olabilir. Bu nedenle her `track_id` için geçmiş kararlar tutulur.

```text
track_id -> helmet_deque(maxlen=20)
track_id -> vest_deque(maxlen=20)
track_id -> mask_deque(maxlen=20)
```

Voting mantığı:

| Durum | Karar |
|---|---|
| En sık değer bilinen bir sınıfsa | O sınıf döner |
| En sık değer `unknown` ise | Bilinen sınıflar arasında majority aranır |
| Bilinen karar sayısı yetersizse | `unknown` korunur |

Bu yapı scene assignment hatalarının doğrudan alarm üretmesini azaltır.

### 8.5 Tez için scene assignment metni

> Scene-based modda PPE ajanları tüm kare üzerinde çalıştığı için tespit edilen PPE kutularının hangi kişiye ait olduğunu belirlemek amacıyla geometrik bir atama yöntemi kullanılmıştır. Her PPE kutusu için kişi kutusuyla kesişim oranı hesaplanmış ve PPE kutusunun en az %40'ı kişi kutusu içinde kalıyorsa bu tespit ilgili kişiye atanabilir kabul edilmiştir. Aynı kişi için birden fazla uygun tespit olması durumunda en yüksek confidence değerine sahip tespit seçilmiştir. Bu yöntem, tam sahne bağlamını korurken kişi bazlı PPE durumu üretmeyi sağlamıştır.

---

## 9. Eğitim Prosedürü ve Augmentation Ayarları

Tüm YOLO eğitimleri Google Colab üzerinde GPU ile yürütülmüştür. Eğitimlerde Ultralytics YOLOv8 kullanılmış ve her ajan için ayrı `data.yaml`, run adı ve Drive yedekleme yapısı oluşturulmuştur.

### 9.1 Ortak eğitim ortamı

| Bileşen | Değer |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4 |
| Ultralytics | 8.4.39 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| AMP | Aktif |
| Pretrained transfer | Aktif |
| Validation | Her epoch sonunda |
| Early stopping | Aktif |
| Plot çıktıları | Aktif |

YOLO eğitim başlangıcında modelin COCO sınıf sayısı hedef veri setine göre override edilmiştir. Örneğin 80 sınıflı pretrained model, ilgili ajan için 1, 2 veya 3 sınıfa düşürülmüştür.

### 9.2 Model bazlı ana eğitim parametreleri

| Model | Base | Epoch hedefi | Gerçek epoch | imgsz | batch | optimizer | patience |
|---|---|---:|---:|---:|---:|---|---:|
| `person_agent_scene_vinayakstyle_best.pt` | `yolov8s.pt` | 100 | 68 | 640 | 16 | AdamW | 20 |
| `fire_smoke_other_agent_final_best.pt` | `yolov8m.pt` | 100 | Notlarda final test var | 640 | 32/64 | auto/MuSGD | 30 |
| `crophelmet_agent_final_best.pt` | `yolov8s.pt` | 140 | 135 | 640 | 16 | auto | 25 |
| `cropvest_agent_final_best.pt` | `yolov8s.pt` | 100 | 100 | 640 | 16 | auto/AdamW | 30 |
| `cropmask_agent_final_best.pt` | `yolov8s.pt` | 200 | 130 | 640 | 16 | auto | 25 |
| `helmet_agent_final_best.pt` | `yolov8s.pt` | 200 | 109 | 640 | 16 | auto | 25 |
| `vest_agent_final_best.pt` | `yolov8s.pt` | 200 | 124 | 640 | 16 | auto | 25 |
| `mask_agent_scene_200ep_yolov8m_best.pt` | `yolov8m.pt` | 200 | CSV eklenmeli | 960 | 16 | AdamW | 30 |

### 9.3 Ortak augmentation ayarları

Ultralytics eğitim loglarında kullanılan tipik augmentation ayarları:

| Ayar | Değer | Açıklama |
|---|---:|---|
| `mosaic` | 1.0 | Mosaic augmentation aktif |
| `close_mosaic` | Genelde 10, fire/smoke için 15 | Son epochlarda mosaic kapatma |
| `hsv_h` | 0.015 | Hue değişimi |
| `hsv_s` | 0.7 | Saturation değişimi |
| `hsv_v` | 0.4 | Value/brightness değişimi |
| `fliplr` | 0.5 | Yatay flip |
| `flipud` | 0.0 | Dikey flip kapalı |
| `scale` | 0.5 | Ölçek augmentation |
| `translate` | 0.1 | Konum kaydırma |
| `degrees` | 0.0 | Rotasyon kullanılmadı |
| `shear` | 0.0 | Shear kullanılmadı |
| `perspective` | 0.0 | Perspective kullanılmadı |
| `erasing` | 0.4 | Random erasing |
| `auto_augment` | `randaugment` | Otomatik augment |

Albumentations tarafında loglarda görülen ek işlemler:

| İşlem | Olasılık | Açıklama |
|---|---:|---|
| Blur | 0.01 | Hafif bulanıklık |
| MedianBlur | 0.01 | Median blur |
| ToGray | 0.01 | Gri tonlama |
| CLAHE | 0.01 | Lokal kontrast iyileştirme |

### 9.4 Optimizer ve learning rate notları

Bazı eğitimlerde `optimizer='auto'` kullanılmıştır. Bu durumda Ultralytics verilen `lr0` ve momentum değerlerini doğrudan kullanmak yerine kendi seçimini yapabilir.

Örnek:

```text
cropvest_agent_final:
  Kullanıcı ayarı: lr0=0.001, optimizer=auto
  Ultralytics seçimi: AdamW(lr=0.001667, momentum=0.9)
```

Bu nedenle tezde optimizer bilgisi yazılırken yalnızca notebook'taki parametre değil, training logunda görülen gerçek optimizer seçimi de dikkate alınmalıdır.

### 9.5 Early stopping mantığı

Eğitimlerde `patience` kullanılmıştır. Bu, validation metriği belirli sayıda epoch boyunca iyileşmezse eğitimin durdurulması anlamına gelir.

Örnekler:

| Model | Patience | Best epoch | Eğitim durma noktası |
|---|---:|---:|---:|
| Scene helmet | 25 | 84 | 109 |
| Scene vest | 25 | 99 | 124 |
| Crop helmet | 25 | 110 | 135 |
| Crop mask | 25 | 105 | 130 |
| Person | 20 | 48 | 68 |
| Scene mask `yolov8s` referans | 30 | 128 | 158 |

### 9.6 Tez için eğitim prosedürü metni

> Tüm ajanlar Google Colab ortamında Ultralytics YOLOv8 kullanılarak eğitilmiştir. Eğitimlerde pretrained YOLOv8 ağırlıkları başlangıç noktası olarak kullanılmış ve model başlığı her ajanın sınıf sayısına göre yeniden yapılandırılmıştır. Eğitim sürecinde AMP etkinleştirilmiş, validation her epoch sonunda yapılmış ve aşırı öğrenmeyi azaltmak için early stopping uygulanmıştır. Augmentation tarafında mosaic, renk uzayı değişimleri, yatay çevirme, ölçekleme, konum kaydırma ve düşük olasılıklı blur/gray/CLAHE işlemleri kullanılmıştır. Her eğitim sonunda `best.pt`, `last.pt`, `results.csv` ve konfigürasyon dosyaları Google Drive'a yedeklenmiştir.

---

## 10. Model Dosyası / Notebook / Drive Kayıt Tablosu

Bu bölüm, final tez arşivinde hangi modelin hangi notebook ve hangi run ile ilişkili olduğunu gösterir.

| Model | Notebook | Run adı | Drive model yolu | Arşiv yolu |
|---|---|---|---|---|
| `person_agent_scene_vinayakstyle_best.pt` | `notebooks/scene_based/person_agent_scene_colab_drive_safe.ipynb` | `person_agent_scene_vinayakstyle` | `/content/drive/MyDrive/ppe_training_datasets/models/person_agent_scene_vinayakstyle_best.pt` | `models/scene_based/person_agent_scene_vinayakstyle_best.pt`, `models/crop_based/person_agent_scene_vinayakstyle_best.pt` |
| `fire_smoke_other_agent_final_best.pt` | `notebooks/scene_based/costi_fire_smoke_yolov8m_colab_drive.ipynb` | `fire_smoke_other_agent_final` | `/content/drive/MyDrive/ppe_training_datasets/models/fire_smoke_other_agent_final_best.pt` | `models/scene_based/fire_smoke_other_agent_final_best.pt`, `models/crop_based/fire_smoke_other_agent_final_best.pt` |
| `crophelmet_agent_final_best.pt` | `notebooks/crop_based/crophelmet_colab_drive.ipynb` | `crophelmet_agent_200_retry` | `/content/drive/MyDrive/ppe_training_datasets/models/crophelmet_agent_final_best.pt` | `models/crop_based/crophelmet_agent_final_best.pt` |
| `cropvest_agent_final_best.pt` | `notebooks/crop_based/cropvest_train_colab_drive.ipynb` | `cropvest_agent_final` | `/content/drive/MyDrive/ppe_training_datasets/models/cropvest_agent_final_best.pt` | `models/crop_based/cropvest_agent_final_best.pt` |
| `cropmask_agent_final_best.pt` | `notebooks/crop_based/cropmask_final_colab_drive.ipynb` | `cropmask_agent_final` | `/content/drive/MyDrive/ppe_training_datasets/models/cropmask_agent_final_best.pt` | `models/crop_based/cropmask_agent_final_best.pt` |
| `helmet_agent_final_best.pt` | `notebooks/scene_based/helmet_final_colab_drive.ipynb` | `helmet_agent_normal_final` | `/content/drive/MyDrive/ppe_training_datasets/models/helmet_agent_final_best.pt` | `models/scene_based/helmet_agent_final_best.pt` |
| `vest_agent_final_best.pt` | `notebooks/scene_based/vest_final_colab_drive.ipynb` | `vest_agent_normal_final` | `/content/drive/MyDrive/ppe_training_datasets/models/vest_agent_final_best.pt` | `models/scene_based/vest_agent_final_best.pt` |
| `mask_agent_scene_200ep_yolov8m_best.pt` | `notebooks/scene_based/mask_agent_scene_yolov8m_colab_drive_safe.ipynb` | `mask_agent_scene_200ep_yolov8m` | `/content/drive/MyDrive/ppe_training_datasets/models/mask_agent_scene_200ep_yolov8m_best.pt` | `models/scene_based/mask_agent_scene_200ep_yolov8m_best.pt` |

Kalıcı arşiv:

| Arşiv | Yol |
|---|---|
| Klasör | `C:\Users\berat\Desktop\Bitirme\ppe_yolo_training\thesis_scene_vs_crop_desktop_exports_2026-05-02` |
| Zip | `C:\Users\berat\Desktop\Bitirme\ppe_yolo_training\thesis_scene_vs_crop_desktop_exports_2026-05-02.zip` |
| Ek zip kopyası | `C:\Users\berat\Desktop\kesin\thesis_scene_vs_crop_desktop_exports_2026-05-02.zip` |

Tez için kayıt notu:

> Final model dosyaları, eğitimde kullanılan Colab notebook kopyalarıyla birlikte ayrı bir tez arşivinde saklanmıştır. Böylece her modelin hangi veri seti, hangi eğitim parametreleri ve hangi run çıktısıyla üretildiği izlenebilir hale getirilmiştir.

---

## 11. Final Kararların Bilimsel Gerekçesi

Final ajan seçimi yalnızca en yüksek tekil metriklere göre yapılmamıştır. Ajanlar, tezdeki scene/crop karşılaştırma amacını karşılayacak şekilde seçilmiştir.

| Kriter | Açıklama |
|---|---|
| Kendi eğitilmiş model kullanımı | Final sistem dış hazır model yerine kendi eğitilen ajanlara dayanır |
| Scene/crop karşılaştırma | Helmet, vest ve mask için iki farklı çalışma modu korunur |
| Ortak person altyapısı | İki modun adil karşılaştırılması için aynı person agent kullanılır |
| Ortak fire/smoke altyapısı | Fire/smoke PPE crop problemi olmadığı için sabit tutulur |
| Video pipeline uygunluğu | Modeller sadece dataset metriğiyle değil, proje videolarındaki davranışla değerlendirilir |
| Tez anlatılabilirliği | Her ajan ayrı rol ve deneysel gerekçeyle savunulabilir |

---

## 12. Tezde Kullanılabilecek Ana Açıklama Metni

Aşağıdaki metin doğrudan tezde "Final Model Seçimi" veya "Önerilen Sistem Mimarisi" bölümünde kullanılabilir.

> Bu çalışmada final sistem, YOLOv8 tabanlı çok ajanlı bir mimari olarak tasarlanmıştır. Sistemde kişi tespiti, PPE tespiti ve yangın/duman algılama görevleri ayrı ajanlara bölünmüştür. Kişi tespiti için `person_agent_scene_vinayakstyle_best.pt` modeli her iki çalışma modunda ortak olarak kullanılmıştır. Bu model, video karelerinde kişileri tespit ederek ByteTrack ile takip edilmesini sağlar ve PPE durumlarının kişi bazlı değerlendirilmesine temel oluşturur.

> Yangın ve duman algılama için `fire_smoke_other_agent_final_best.pt` modeli kullanılmıştır. Bu ajan, `fire`, `smoke` ve `other` sınıflarını tam sahne üzerinde tespit eder. Yangın ve duman olayları kişi üzerinde taşınan PPE öğeleri olmadığı için crop-based modda bile bu ajan scene-based olarak çalıştırılmıştır.

> PPE tespiti için iki farklı yaklaşım korunmuştur. Scene-based modda helmet, vest ve mask ajanları tam kare üzerinde çalışmakta; ürettikleri tespitler kişi kutularına kesişim oranı tabanlı bir atama yöntemiyle bağlanmaktadır. Bu modda helmet için `helmet_agent_final_best.pt`, vest için `vest_agent_final_best.pt` ve mask için `mask_agent_scene_200ep_yolov8m_best.pt` kullanılmıştır. Crop-based modda ise önce kişi tespiti yapılmakta, ardından kişi veya kişi alt-bölgeleri kırpılarak PPE ajanlarına verilmektedir. Bu modda helmet için `crophelmet_agent_final_best.pt`, vest için `cropvest_agent_final_best.pt` ve mask için `cropmask_agent_final_best.pt` kullanılmıştır.

> Bu yapı sayesinde çalışma, yalnızca tek bir modelin başarısını değil, aynı PPE ihlal probleminin iki farklı algılama stratejisiyle nasıl çözüldüğünü karşılaştırmaktadır. Scene-based yaklaşım tam sahne bağlamını korurken, crop-based yaklaşım kişi odaklı bölgeler üzerinde karar vererek arka plan karmaşasını azaltmayı hedefler. Böylece özellikle küçük ve lokal PPE öğeleri olan baret ve maske için crop-based yaklaşımın etkisi, yelek gibi daha geniş görsel alana sahip PPE öğeleri için ise scene-based yaklaşımın rekabetçi performansı incelenebilir.

---

## 13. Final Ajanların Kısa Savunma Tablosu

| Ajan | Neden final? |
|---|---|
| `person_agent_scene_vinayakstyle_best.pt` | Her iki modda kişi bazlı takip ve event üretimi için ortak temel sağlar |
| `fire_smoke_other_agent_final_best.pt` | Ayrı fire/smoke modellerinden daha dengeli sonuç verdi; sahne olayı olduğu için ortak kullanılır |
| `crophelmet_agent_final_best.pt` | Crop modda `NO-Hardhat` sınıfında güçlü ve dengeli sonuç verdi |
| `cropvest_agent_final_best.pt` | Crop modun yelek temsilcisi; 100 epoch final eğitimde güçlü validation sonucu verdi |
| `cropmask_agent_final_best.pt` | Crop modda Mask/NO-Mask ayrımında güçlü test sonucu verdi |
| `helmet_agent_final_best.pt` | Scene modun baret temsilcisi; tam kare helmet/no-helmet tespiti için kullanılır |
| `vest_agent_final_best.pt` | Scene modda güçlü final test sonucu verdi; yelek için sahne bağlamı etkili |
| `mask_agent_scene_200ep_yolov8m_best.pt` | Scene modun final maske temsilcisi; daha büyük YOLOv8m backbone ve 960 img size ile seçildi |

---

## 14. Sonuç Cümlesi

Tezde final karar şu şekilde net ifade edilebilir:

> Final sistemde kişi tespiti ve yangın/duman algılama ajanları her iki modda ortak tutulmuş; helmet, vest ve mask tespitleri ise scene-based ve crop-based olmak üzere iki ayrı ajan setiyle karşılaştırılmıştır. Bu seçim, sistemin hem modüler ve kendi eğitilmiş modellerden oluşmasını sağlamış hem de tez kapsamında PPE algılama stratejisinin etkisini analiz etmeye olanak vermiştir.
