# İTÜ Bilgisayar Mühendisliği — AI Tanıtım Sistemi Knowledge Base

Bu dizin, İş Paketi B (Data & Knowledge Base) kapsamında toplanan ve yapılandırılan
verileri içerir. Amaç: AI tanıtım sisteminin **doğru, güncel, kaynaklı ve ikna edici**
cevaplar üretebilmesi için gerekli bilgi tabanını sağlamak.

## Dizin Yapısı (PDF §3.3'e uygun)

```
knowledge_base/
├── raw_data/          ← Ham toplanan veriler + kaynak/URL kayıtları (denetim izi)
│   ├── rankings/      ← SOURCES.md
│   ├── career/        ← SOURCES.md
│   ├── campus/        ← SOURCES.md (kulüpler & proje takımları dahil)
│   ├── faq/           ← SOURCES.md
│   ├── academic/      ← SOURCES.md + bbf_akademisyen_veri_seti-detayli.xlsx (kaynak dosya)
│   └── research/      ← SOURCES.md
├── processed/         ← Temizlenmiş, chunk'lara ayrılmış Türkçe Markdown (embed'e hazır)
│   ├── 01_ranking.md          07_akademik_kadro.md   (öğretim üyeleri)
│   ├── 02_akademik.md         08_laboratuvarlar.md   (16 laboratuvar)
│   ├── 03_arastirma.md        09_kulupler.md          (öğrenci kulüpleri)
│   ├── 04_kampus.md           10_proje_takimlari.md   (proje takımları)
│   ├── 05_kariyer.md          11_mezun_profilleri.md  (temsili kariyer profilleri)
│   └── 06_faq.md
├── vectors/           ← ChromaDB / FAISS dosyaları (Modül C tarafından üretilecek — şimdilik boş)
├── metadata.json      ← Tüm chunk'ların metadata indeksi (build_metadata.py üretir)
├── test_queries.json  ← Retrieval kalite testi için sorular + beklenen cevaplar
├── build_metadata.py  ← processed/*.md → metadata.json (Markdown = tek doğru kaynak)
└── README.md          ← Bu dosya
```

> **Not:** Markdown chunk'ları düzenledikten sonra `python3 build_metadata.py` çalıştırın;
> metadata.json processed/*.md'den otomatik yeniden üretilir (id benzersizliği ve tarih ≥2023 doğrulanır).

## Chunk Formatı

`processed/` altındaki her `.md` dosyası birden çok chunk içerir. Her chunk, üstünde
bir HTML-yorum metadata satırı taşır (PDF §3 Metadata Şeması):

```
<!-- id: ranking-001 | category: ranking | source: <URL> | verified: true | date: 2026-03 | priority: high | audience_type: akademik -->
## Başlık
İçerik...
```

- **category**: `kariyer | akademik | kampus | ranking | arastirma | faq`
- **source**: Kaynak URL veya doküman adı (her chunk kaynaklıdır — PDF Etik Kural)
- **verified**: `true` = birincil/resmi kaynaktan doğrulandı; `false` = ikincil/proxy, çift kontrol gerek
- **date**: Verinin ait olduğu / güncellendiği tarih (hepsi ≥ 2023)
- **priority**: `high | medium | low` (retrieval önceliği)
- **audience_type**: `kariyer_odakli | akademik | sosyal | genel`

`metadata.json` bu metadata'nın makine tarafından okunabilir indeksidir; chunk metni
Markdown dosyalarında kalır (Modül C embed sırasında `id` ile eşler).

## Ek Koleksiyonlar: `comparison/` ve `curriculum/` (AYRI — ana bot KB'sinden bağımsız)

Bu iki dizin, üniversiteler-arası karşılaştırma çalışması kapsamında eklendi ve **ana
`metadata.json`'a KARIŞMAZ** (public bot KB'si temiz kalır).

- **`comparison/`** — İTÜ mühendislik bölümlerinin Boğaziçi, ODTÜ, Koç, YTÜ karşılıklarıyla
  **nesnel, kaynaklı** karşılaştırması. **`audience_type: internal`** — PDF §4 etik kuralı gereği
  bot rakip kıyası/kötülemesi yapamaz; bu veri **dahilidir** (ekip mesaj stratejisi + "Neden İTÜ?"
  sorusuna İTÜ'nün kendi güçlü yanlarıyla cevap için). Kötüleme yok; İTÜ'nün önde OLMADIĞI metrikler
  (örn. QS genelde ODTÜ önde) dürüstçe belirtilir. 23 chunk, 3 eksen (C01-C05):
  C01 üniversite geneli · C02 İTÜ CS ↔ diğer üniv. · C03 İTÜ iç karşılaştırma · C04 bölüm matrisi ·
  C05 İTÜ yapısal avantajları. Build: `python3 comparison/build_metadata.py`. Denetim izi:
  `comparison/raw_data/SOURCES.md`.
- **`curriculum/`** — 5 üniversitede 46 mühendislik programının **tam ders listesi** (kod, ad, kredi,
  AKTS), `<uni>_<bolum>.md` biçiminde. Resmi kataloglardan `curriculum/extract.py` ile çekildi
  (İTÜ OBS, Boğaziçi bölüm siteleri, ODTÜ catalog, Koç [tarayıcı-UA], YTÜ [SSL workaround]). Dizin:
  `curriculum/index.md`. İTÜ programları olgusal/bot'a açılabilir; diğer üniversiteler karşılaştırma amaçlı.

## Veri Güvenilirliği ve Boşluk Politikası (ÖNEMLİ)

Kullanıcı onayı ile benimsenen politika: **proxy veri + boşluk işaretleme**.

- `verified: true` → İTÜ resmi / YÖK / ranking kuruluşu / TÜİK gibi **birincil kaynaktan** doğrulandı.
- `verified: false` → **proxy / ikincil** veri (örn. ulusal sektör ortalaması, topluluk maaş anketi).
  Metinde açıkça "sektör geneli / Türkiye geneli" olarak etiketlenmiştir; İTÜ-Bilgisayar'a
  özgü değildir.
- `<!-- TODO: ... -->` işaretleri → resmi bir İTÜ-Bilgisayar'a özgü rakamla değiştirilmesi
  gereken boşlukları gösterir (çoğu YÖK Atlas'ın JS ile render edilen tablolarında; tarayıcı
  ile manuel çekilmeli).

**Hiçbir sayı uydurulmamıştır.** Doğrulanamayan veya çelişkili bulunan iddialar
`raw_data/` altındaki kaynak günlüklerinde "FLAG" olarak not edilmiştir (örn. bazı
arama-özeti istihdam yüzdeleri TÜİK ile çeliştiği için kullanılmadı).

## Açık Boşluklar (resmi rakam gerektiren — team'in kapatması gereken)

1. **İTÜ-Bilgisayar'a özgü mezun istihdam oranı** — çoklu kaynakta doğrulandı ki **kamuya açık
   hiçbir yerde yayımlanmıyor** (YÖK Atlas SPA + tüm aynalar program-bazlı istihdam vermiyor).
   Yalnızca İTÜ Kariyer/mezun ofisi iç verisiyle doldurulabilir. Şu an TÜİK sektör proxy'si kullanılıyor.
2. **Başlangıç maaşı** İTÜ kırılımı yok (Türkiye geneli proxy).
3. **FAANG/büyük tech'te İTÜ mezunu sayısı** (LinkedIn okul analytics'i login gerektiriyor):
   https://www.linkedin.com/school/itu1773
4. **İsmi doğrulanmış ünlü Bilgisayar Mühendisliği mezunu** (doğrulanan marka isimler
   çoğunlukla elektrik/inşaat/endüstri alanlarından — bölümden teyit alınmalı).
5. **Gerçek (onaylı, isimsiz) mezun profilleri** — şu an 18 profil **temsili arketip** (verified:false).
   İTÜ Kariyer/mezun ofisi veya LinkedIn üzerinden onaylı isimsiz gerçek verilerle değiştirilmeli (PDF §2.2.2).
6. **Erasmus partner listesi**, **Kariyer Zirvesi firma listesi**, **proje takımı bazında yarışma dereceleri**
   (kaynak sayfalar JS-render / liste yayınlamıyor / 403).

> ✅ **Kapatılan boşluklar:** Öğretim üyesi kadrosu (53 öğretim üyesi + 16 laboratuvar, BBF veri setinden),
> öğrenci kulüpleri (~90) ve proje takımları (38) resmi kaynaklardan; **YÖK Atlas kontenjan + çoklu-yıl
> taban puan/başarı sıralaması** (2021–2025, ≥2 kaynakta doğrulanmış) eklendi.

> ⚠️ **Temsili mezun profilleri hakkında:** processed/11_mezun_profilleri.md içindeki 18 profil, gerçek
> bireyler değil **temsili kariyer arketipleridir** (PDF §1.2 doğrulanmamış kişisel hikâyeleri dışlar).
> Doğrulanmış verilere dayanır, açıkça "temsili" etiketlidir ve verified:false taşır.

## Kaynak Kalitesi Özeti

En güçlü, ikna edici, resmi kaynaklı veriler:
- QS 2026 Konu Sıralaması: Mühendislik & Teknoloji **91.**, Bilgisayar Bilimi & Bilgi Sistemleri **152.**
- YÖK Atlas seçicilik: Bilgisayar Müh. İTÜ'nün en yüksek sıralamalı bölümü (~1.400 başarı sırası)
- ABET akreditasyonu
- İTÜ Çekirdek: 325M$+ yatırım, 3.2 milyar$+ değerleme, dünyanın 1 numaralı üniversite kuluçkası (UBI Global)
- TÜİK 2024: bilişim sektörü %78,7 istihdam
- Teknofest 2023: İTÜ 11 ödül; ABD BLS: yazılım istihdamı 2023–2033 %17,9 büyüme
- **Akademik kadro:** 53 öğretim üyesi + 16 araştırma laboratuvarı (AI, veri bilimi, siber güvenlik, NLP, HPC, biyoinformatik…)
- **Öğrenci ekosistemi:** ~90 kulüp + 38 proje takımı (ACM, IEEE, AnthRo insansı robotik, YONGA çip tasarımı, Autobee dünya 1.'si)

## İçerik İstatistikleri (v0.3)

| Kategori | Chunk | | Kategori | Chunk |
|---|---|---|---|---|
| ranking | 12 | | kampus (+kulüpler) | 16 |
| akademik (+kadro) | 57 | | kariyer (+mezun profilleri) | 29 |
| arastirma (+lab+takım) | 35 | | faq | 15 |
| **Toplam** | **164** | | doğrulanmış / proxy | 133 / 31 |

PDF hedefi 200+ chunk. v0.3'te eklenenler: YÖK Atlas doğrulanmış çoklu-yıl taban/sıra/kontenjan (ranking-009, ranking-012)
ve 18 temsili mezun kariyer profili (processed/11_mezun_profilleri.md). Test sorusu: 78.
