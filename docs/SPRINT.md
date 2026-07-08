# Sprint: Gerçek CV Modellerinin Entegrasyonu (Stub → Production)

> Bu doküman bir coding agent'a (Claude Code) verilmek üzere yazılmıştır.
> Her görev; hedef dosya, teknik yönerge ve kabul kriteri içerir.
> **Değişmez mimari kararlara** (Bölüm 2) dokunma; onları yeniden tartışma.

---

## 1. Bağlam

Bu modül, İTÜ Bilgisayar Mühendisliği tanıtım robotunun **görüntü analizi (CV)** bileşenidir. Kamera akışını alır, kişinin davranışsal/yüz analizini yapar ve yapılandırılmış JSON üretir. LLM ve RAG bileşenleri **bu modülün kapsamı dışındadır** — onlara sadece JSON gönderilir.

Mevcut durumda çalışan bir **iskelet** var: WebSocket ingest, çoklu-kiosk oturum yönetimi, state machine, ring buffer / baseline / EMA, iki çıktı kanalı ve Dockerfile hazır ve test edilmiş. Tek eksik: gerçek CV modelleri. Şu an `app/processing.py` içindeki `extract_signals()` **stub**'dır (rastgele değer döner). Bu sprint'in amacı stub'ı gerçek MediaPipe + duygu modelleriyle değiştirmektir.

### Mevcut dosya yapısı

```
cv-pipeline/
├── app/
│   ├── main.py          # FastAPI: /stream, /profile, /focus, /health + arka plan görevleri
│   ├── manager.py       # session_id -> SessionData dict + stale GC
│   ├── session.py       # SessionData, RawSignals, state machine sabitleri
│   ├── processing.py    # FrameSlot (drop-stale) + extract_signals() [STUB]
│   └── scoring.py       # update_session + build_initial_profile + compute_focus
├── Dockerfile
├── requirements.txt
└── test_client.py       # sahte kamera akışı üreten test istemcisi
```

### Çıktı kanalları (mevcut, korunacak)

- `WS /profile/{session_id}` — kişi başına **tek seferlik** zengin profil (duygu, dikkat, açıklık, enerji). Yeterli veri toplanır toplanmaz gönderilir; yeni kişi gelince otomatik yeniden tetiklenir.
- `WS /focus/{session_id}` — **her ~2.5 sn** `is_focused` + `focus_time` (kesintisiz odaklanma streak'i; dikkat dağılınca 0'a sıfırlanır).

---

## 2. Değişmez mimari kararlar

Bunlar önceki tasarım turlarında kararlaştırıldı. Claude Code bunları **değiştirmemeli**, yalnızca üzerine inşa etmeli:

1. **Ağ-stream tüketici model.** Modül kamera donanımına erişmez; istemci (robot) JPEG kareleri `/stream` üzerinden push eder. Container donanımdan izole kalır. `cv2.VideoCapture(0)` gibi cihaz erişimi **eklenmeyecek**.
2. **Çoklu-kiosk.** State her zaman `session_id` ile ayrılır (`SessionManager`). Global tekil state yok.
3. **Drop-stale.** Kareler kuyruğa alınmaz; yalnızca en güncel kare işlenir (`FrameSlot`). Sınırsız kuyruk eklenmeyecek.
4. **Stateful analiz.** Ring buffer, baseline (delta), EMA korunur.
5. **Doküman yerine iyileştirilmiş metodlar** (Bölüm 3). Orijinal teknik dokümandaki eski yöntemlere geri dönülmeyecek.

---

## 3. Metod kararları (dokümandan sapmalar — bilerek)

Orijinal teknik doküman bazı eski/ağır araçlar öneriyordu; bu sprint aşağıdaki **iyileştirilmiş** yöntemleri kullanır. Claude Code eski yönteme dönmemeli:

| Sinyal | Dokümandaki eski yöntem | Kullanılacak yöntem |
|---|---|---|
| Göz teması | Sadece iris'in göz içi yatay konumu | **Baş pozu (transformation matrix) + göz blendshape'leri** birleşik; baş dönükse override |
| Öne eğilme (lean) | Image `landmark.z` (gürültülü monoküler) | **Pose Landmarker WORLD landmarks** z farkı |
| Omurga/postür | `shoulder_width / hip_width` (dönmeyle karışır) | World landmarks geometrisi + baş pitch |
| Duygu | DeepFace (ağır, runtime indirme) | **EmotiEffLib / HSEmotion ONNX** (hafif, worker thread) |
| Yüz landmark | `mp.solutions.face_mesh` (eski Solutions API) | **MediaPipe Tasks — Face Landmarker** (landmark + 52 blendshape + head pose) |

> Not: MediaPipe Tasks çağrıları için **VIDEO running mode** kullan (monoton artan timestamp'lerle senkron çağrı). LIVE_STREAM'in async callback'i, mevcut senkron worker thread yapısını gereksiz karmaşıklaştırır. IMAGE mode da kabul edilebilir ama VIDEO, zamansal tutarlılık verir.

---

## 4. Sprint hedefi (Definition of Done)

Sprint tamamlandığında:

- `extract_signals()` gerçek modelleri kullanıyor; `RawSignals`'ın tüm alanları gerçek verilerle doluyor.
- Göz teması baş pozuyla düzeltilmiş; lean/spine world landmarks üzerinden.
- Duygu EmotiEffLib ile, ana işleme yolunu bloklamadan (ayrı thread) hesaplanıyor.
- Birden fazla yüz olduğunda tek birincil kişi seçiliyor.
- Modeller Docker image'ına **build-time'da gömülü**; runtime indirme yok.
- Mevcut iskelet davranışı bozulmadan çalışıyor (profil tek sefer, focus periyodik, çoklu-kiosk izolasyonu).
- Temel testler (`pytest`) geçiyor.

---

## 5. Görevler

Görevler sıralıdır; bağımlılıklar belirtildi. Her görev ayrı commit olabilir.

### T1 — Model varlıklarının hazırlanması ve Docker'a gömülmesi
**Dosyalar:** `scripts/fetch_models.sh` (yeni), `Dockerfile`, `models/` (yeni, .gitignore'a eklenebilir)

- `scripts/fetch_models.sh`: MediaPipe **Face Landmarker** (`.task`) ve **Pose Landmarker** (`.task`) model dosyalarını resmi MediaPipe model deposundan indirir. Güncel URL'leri MediaPipe'ın resmi model sayfasından doğrula (URL'yi ezberden yazma; hardcode edilen link bayatlamış olabilir). EmotiEffLib ONNX ağırlığını da indir/yerleştir.
- `Dockerfile`: mevcut `# TODO(gercek-model)` satırını gerçek `COPY models/ /srv/models/` ile değiştir. Modeller build-time'da image'a girmeli.
- Model dosyalarının yolu bir config/env değişkeninden okunmalı (örn. `MODELS_DIR=/srv/models`).

**Kabul kriteri:**
- `docker build` başarılı; image içinde `/srv/models/` altında üç model dosyası mevcut.
- Container ilk çalıştığında hiçbir model **indirmeye çalışmıyor** (offline çalışabilir).

---

### T2 — Face Landmarker sarmalayıcısı
**Dosyalar:** `app/detectors/__init__.py` (yeni), `app/detectors/face.py` (yeni)
**Bağımlılık:** T1

- `mediapipe` Tasks API ile Face Landmarker'ı **VIDEO** modunda yükle; `output_face_blendshapes=True`, `output_facial_transformation_matrixes=True`, `num_faces` ≥ 2 (birincil kişi seçimi için).
- Bir sınıf/fonksiyon: girdi = BGR frame + monoton artan timestamp (ms); çıktı = her yüz için `{landmarks, blendshapes(dict), transform_matrix(4x4), bbox}`.
- `transform_matrix`'ten baş pozunu (yaw/pitch/roll derece) çıkaran yardımcı fonksiyon ekle.
- Yüz yoksa boş liste dön.

**Kabul kriteri:**
- Yüz içeren bir test karesi → en az bir yüz, blendshape sözlüğü dolu, head pose (yaw/pitch/roll) makul aralıkta.
- Yüzsüz kare → boş liste, exception yok.

---

### T3 — Pose Landmarker sarmalayıcısı (world landmarks)
**Dosyalar:** `app/detectors/pose.py` (yeni)
**Bağımlılık:** T1

- Pose Landmarker'ı VIDEO modunda yükle; **world landmarks** çıktısını kullan (`pose_world_landmarks`).
- Çıktı: ilgili landmark indeksleri (11,12 omuz; 23,24 kalça; 13-16 kol/bilek) + her birinin visibility/presence skoru.
- Poz yoksa `None` dön.

**Kabul kriteri:**
- Gövde içeren kare → world landmarks (metre ölçeğinde) döner; omuz/kalça mevcut.
- Bilekler kadraj dışındaysa ilgili landmark'ların visibility düşük işaretli.

---

### T4 — Göz teması (baş pozuyla düzeltilmiş)
**Dosyalar:** `app/detectors/gaze.py` (yeni)
**Bağımlılık:** T2

- Girdi: Face Landmarker çıktısı (blendshape'ler + head pose).
- `eyeLookIn/Out/Up/Down` blendshape'lerini ve baş yaw/pitch'i birleştirerek `eye_contact ∈ [0,1]` üret.
- **Kapı (gating):** baş yaw veya pitch bir eşiği (örn. |yaw| > 25°) aşarsa, göz ortada olsa bile `eye_contact` düşürülür.
- `head_yaw_deg` değerini de dışarı ver (JSON'a giriyor).

**Kabul kriteri:**
- Frontal yüz + merkezî göz → `eye_contact > 0.75`.
- Baş ~40° yana dönük + göz merkezî → `eye_contact < 0.45` (eski iris-only yöntemin aksine).

---

### T5 — Lean (world z) + baseline delta
**Dosyalar:** `app/detectors/posture.py` (yeni), gerekiyorsa `app/scoring.py`
**Bağımlılık:** T3

- `lean = shoulder_z_world - hip_z_world` (world landmarks). Mevcut `scoring.py`'daki baseline/delta mantığı korunur; sadece kaynağı image-z yerine world-z olur.
- Değeri smoothing için ham olarak `RawSignals.lean`'e yaz.

**Kabul kriteri:**
- Kişi baseline'a göre öne eğildiğinde `delta_lean < -0.03`; geri yaslandığında `> +0.03`.
- Değerler image-z'ye göre gözle görülür şekilde daha stabil (frame-to-frame titremesi düşük).

---

### T6 — Omurga / postür (dönme karışımını azalt)
**Dosyalar:** `app/detectors/posture.py`
**Bağımlılık:** T3

- Ham `shoulder_width/hip_width` oranı yerine: world landmarks'tan omuz-kalça dikey geometrisi + baş pitch birleşimiyle bir `spine_ratio`/`tilt` üret. Amaç: **yana dönme** ile **eğik/yorgun duruş**u ayırmak.
- `RawSignals.spine_ratio`, `RawSignals.shoulder_tilt` doldur.

**Kabul kriteri:**
- Kişi ~20° yana döndüğünde "eğik/yorgun" olarak **yanlış** işaretlenmiyor.
- Gerçekten öne çökmüş duruş düşük `spine_ratio` veriyor.

---

### T7 — Kollar kapalı + valid bayrağı
**Dosyalar:** `app/detectors/posture.py`
**Bağımlılık:** T3

- Bilek landmark'larının visibility'si eşiğin altındaysa `arms_crossed = None` (ölçülemedi). Aksi halde dokümandaki mesafe + gövde-önü (z) mantığıyla bool üret.
- `None` durumunda `scoring` tarafı zaten `valid:false` işaretliyor ve skora katmıyor — bu davranış korunmalı.

**Kabul kriteri:**
- Bilekler kadraj dışında → `arms_crossed=None`, JSON'da `valid:false`.
- Bilekler görünür ve gövde önünde birleşik → `true`.

---

### T8 — Duygu tanıma (EmotiEffLib ONNX, worker thread)
**Dosyalar:** `app/detectors/emotion.py` (yeni), `app/main.py` (worker entegrasyonu)
**Bağımlılık:** T1, T2 (yüz bbox / crop için)

- EmotiEffLib/HSEmotion ONNX modelini yükle. Girdi: yüz kırpımı (Face Landmarker bbox'ından). Çıktı: dominant duygu etiketi + skor sözlüğü.
- **Ayrı thread'de, ~1 Hz** çalışsın (her karede değil). Sonucu ilgili oturumun state'ine yazsın (örn. `SessionData.last_raw.emotion_*` veya ayrı bir alan). Ana işleme yolunu bloklamamalı.
- Model, mevcut DeepFace kurgusundaki gibi kuyruk/worker desenini kullanabilir ama JPEG değil, kırpılmış yüz array'i alsın.

**Kabul kriteri:**
- Duygu etiketi + skorlar üretiliyor; ana işleme frekansı (15fps hedefi) duygu modelinden etkilenmiyor.
- Yüz yokken son bilinen değeri koruyor, crash yok.

---

### T9 — `extract_signals()` entegrasyonu
**Dosyalar:** `app/processing.py`
**Bağımlılık:** T2–T8

- Stub'ı kaldır. Sırayla: Face Landmarker + Pose Landmarker çalıştır → birincil kişiyi seç (T10) → gaze (T4), lean (T5), spine (T6), arms (T7) hesapla → yüz kırpımını duygu worker'ına ver (T8) → `RawSignals`'ı doldur.
- Timestamp yönetimi: VIDEO mode için monoton artan ms timestamp üret (istemci saatine güvenme).
- Face ve Pose'un **aynı frame** üzerinde çalıştığından emin ol.

**Kabul kriteri:**
- `test_client.py` ile gerçek bir yüz/gövde videosu beslendiğinde `RawSignals`'ın tüm alanları anlamlı doluyor.
- Mevcut iskelet testleri (profil tek sefer, focus periyodik, çoklu-kiosk) hâlâ geçiyor.

---

### T10 — Birincil kişi seçimi
**Dosyalar:** `app/detectors/select.py` (yeni) veya `app/processing.py`
**Bağımlılık:** T2, T3

- Birden fazla yüz/gövde varsa **tek** kişi seç. Kural: en büyük yüz bbox (veya en merkezî / en yakın-z). Kuralı tek yerde, yapılandırılabilir tut.
- Face ve Pose sonuçlarını aynı kişiye eşle (konum/bbox örtüşmesine göre).
- Kimse yoksa `face_present=False`.

**Kabul kriteri:**
- Kadrajda iki kişi → tek, tutarlı profil üretiliyor (kare kare seçilen kişi zıplamıyor).
- Tek kişi senaryosunda davranış değişmiyor.

---

### T11 — Oturumdan ayrılma dayanıklılığı
**Dosyalar:** `app/main.py`, `app/session.py`
**Bağımlılık:** yok (bağımsız)

- Mevcut açık: stream tamamen kesilirse (WS disconnect) state güncellenmiyor, sadece 120sn GC devreye giriyor. `/stream` disconnect handler'ında oturumu uygun şekilde işaretle (örn. hızlı IDLE'a geçiş veya "kişi gitti" işareti).
- "Kişi ayrıldı"yı iki kaynaktan da yakala: (a) `face_present=False` + timeout, (b) stream disconnect.

**Kabul kriteri:**
- İstemci bağlantıyı düşürünce oturum makul sürede IDLE'a dönüyor; bir sonraki kişi için profil yeniden tetikleniyor.

---

### T12 — Konfigürasyon + testler
**Dosyalar:** `app/config.py` (yeni), `tests/` (yeni)
**Bağımlılık:** T9

- Eşik/pencere sabitlerini (`session.py` içindekiler dahil) tek bir config'e/env'e taşı: kalibrasyon süresi, focus eşiği/aralığı, göz-pozu kapı eşiği, birincil-kişi kuralı vb.
- `pytest` ile birim testleri: state machine geçişleri (IDLE→CALIBRATING→ACTIVE), focus streak sıfırlanması, profil-tek-sefer garantisi, çoklu-kiosk izolasyonu. Model çağrıları mock'lanabilir (`extract_signals`'ı fake ile).

**Kabul kriteri:**
- `pytest` yeşil.
- Eşikler kod değiştirmeden env ile ayarlanabiliyor.

---

## 6. Test & doğrulama protokolü

Orijinal teknik dokümanın 7. bölümündeki kalibrasyon senaryolarını referans al. En az:

- Frontal bakış → yüksek göz teması; ~40° yana dönme → düşük (T4 doğrulaması).
- Öne eğil / geri yaslan → delta lean işaret değişimi (T5).
- Yana dön → "eğik duruş" **yanlış pozitifi vermemeli** (T6).
- İki kişi → tek profil (T10).
- Kimse yok → profil üretilmemeli (state IDLE).
- 15fps sabit akışta CPU ve gecikme hedefte (canlı akış geri kalmıyor).

`test_client.py`'ı gerçek yüz içeren kayıtlı bir videoyu kare kare gönderecek şekilde genişletmek, deterministik doğrulama için faydalı olur.

---

## 7. Riskler / dikkat edilecekler

- **mediapipe sürüm/Python uyumu** hassastır; `requirements.txt`'te pinli tut, `python:3.11-slim` bazında test et.
- MediaPipe Tasks VIDEO modu **monoton artan timestamp** ister; azalan/tekrarlayan timestamp hata verir.
- EmotiEffLib worker'ı ana işleme frekansını düşürmemeli; kuyruğu tek elemanlı (drop-stale) tut.
- Duygu tanıma çıktısı **olasılıksal** — JSON'daki `confidence` alanları korunmalı, skor modeli düşük güveni indirgemeli.
- World landmarks image landmarks'tan farklı ölçekte; eski image-tabanlı eşikleri **birebir taşıma**, yeniden kalibre et.

---

## 8. Kapsam dışı (bu sprint'te YAPILMAYACAK)

- LLM / RAG entegrasyonu (başka ekip).
- JSON'dan doğal-dil `description` metin bloğu üretimi (gerekirse ayrı sprint; şu an ham JSON yeterli).
- Robotun fiziksel yüz-takibi (robot bunu lokal yapar; bu modül sadece profil + focus üretir).
- Kimlik tanıma / kişi tekilleştirme (kimliklendirme yok, yalnızca anlık birincil kişi).

---

## 9. Claude Code'a öneri sıra

`T1 → T2 → T3 → (T4, T5, T6, T7 paralel) → T8 → T10 → T9 → T11 → T12`

Her görevi ayrı commit + kısa açıklamayla ilerlet. T9 (entegrasyon) öncesi T2–T8'in tek tek çalıştığını doğrula.
