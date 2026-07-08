# Ham Veri — Akademik & Bölüm (academic)

Denetim izi. FLAG = doğrulanamadı/dikkat.

## Bölüm & Yapı

- Bölüm **1980**'de program olarak kuruldu; **2010**'da **Bilgisayar ve Bilişim Fakültesi** çatısına taşındı.
  Kaynak (resmi): https://bm.itu.edu.tr/en · Fakülte: https://bbf.itu.edu.tr/en
- Lisans derecesi: **BSc Computer Engineering**, Bilgisayar ve Bilişim Fakültesi.
- **%100 İngilizce** program (2017–2018 alımından beri). https://www.tyyc.itu.edu.tr/ProgramHakkinda.php?Program=BLG_LS&Dili=EN · https://en.wikipedia.org/wiki/Istanbul_Technical_University

## Akreditasyon

- Lisans programı **ABET (Engineering Accreditation Commission / EAC)** akrediteli — Computer Engineering kriterleri.
  https://bm.itu.edu.tr/egitim/lisans/bilgisayar-muhendisligi/akreditasyon · Merkez: https://www.itu.edu.tr/akreditasyon
- FLAG: BM'ye özel **MÜDEK** durumu teyit edilemedi (BM ABET olarak sunuluyor; diğer birçok İTÜ müh. programı MÜDEK'li). https://www.mudek.org.tr/tr/akredit/akredite2025.shtm

## Programlar & Müfredat

- Lisans: Computer Engineering + **Information Systems Engineering (SUNY çift diploma)**.
  Lisansüstü: MSc/PhD Computer Engineering, **Game and Interaction Technologies** MSc, Information Technologies (tezsiz) MSc.
  https://bm.itu.edu.tr/en
- **Çift diploma / uluslararası ortak program:** öğrenciler öğreniminin yarısını partner **ABD üniversitelerinde**
  tamamlayıp çift diploma alabilir. https://www.studyinturkiye.gov.tr/UniversityTurkey/Detail?uId=115069
- Ders planı (kamuya açık): https://www.sis.itu.edu.tr/EN/student/undergraduate/course-plans/plans/BLG/201810.html
  (algoritmalar, veritabanları, mikroişlemciler vb. zorunlu dersler buradan alıntılanabilir).
- Öğrenci kapasitesi (ABET öz-değerlendirme 2016–2022, proxy): yeni öğrenci ~155–245/yıl; mezun ~103–180/yıl;
  toplam kayıtlı lisans ~978–1.058. https://bm.itu.edu.tr/egitim/lisans/bilgisayar-muhendisligi/akreditasyon

### FLAG
- Öğretim üyesi tam sayısı **çekilemedi** (personel sayfaları JS-render). https://bm.itu.edu.tr/personel <!-- TODO -->
- BM'ye özgü Erasmus partner sayısı/listesi ve ÇAP/yandal detayları yakalanamadı. Erasmus ofisi: https://erasmus.itu.edu.tr/ <!-- TODO -->

## BBF Akademik Kadro Veri Seti (yerel dosya — birincil)

- Kaynak dosya: `raw_data/academic/bbf_akademisyen_veri_seti-detayli.xlsx` (kullanıcı tarafından sağlandı; İTÜ BBF sayfası + akademi.itu.edu.tr profillerinden derlenmiş).
- İçerik: 135 kayıt — **53 öğretim üyesi** (≈19 Prof. Dr., 13 Doç. Dr., 6 Dr. Öğr. Üyesi), 53 araştırma görevlisi, 8 Turkcell destekli araştırmacı, 17 idari personel.
- Bölümler: Bilgisayar Mühendisliği, Yapay Zeka ve Veri Mühendisliği, Siber Güvenlik Mühendisliği (+ BİL Koordinatörlüğü, Yarı Zamanlı).
- Alanlar: Ad, Soyad, Ünvan, Bölüm, Araştırma Alanları, Çalışma Alanları, Eğitim Durumu, Laboratuvar Adı/Konumu, Ofis, Ofis Saatleri, E-posta, Telefon, Profil URL, Hakkında.
- **16 laboratuvar grubu** (Yapay Zeka Sistemleri, Veri Bilimi, Siber Güvenlik, Görsel Bilişim, Doğal Dil İşleme, Biyoinformatik, Sağlık Bilişimi, IoT, Robotik, Oyun Teknolojileri, Yüksek Performanslı Hesaplama, Ağ Sistemleri, Multimedya, Duygusal Hesaplama, Mobil Sistemler, Yazılım Mühendisliği).
- **GİZLİLİK NOTU:** Chunk'lara yalnızca ad-soyad, ünvan, bölüm, araştırma alanları, laboratuvar ve profil URL'si alındı. **Telefon, e-posta, ofis no ve ofis saatleri KB chunk'larına dahil EDİLMEDİ** (kişisel/operasyonel veri — PDF §4 etik kuralı). İsteğe bağlı olarak resmi profil URL'si kaynak olarak verilir.
- Chunk üretimi: `scratchpad/gen_faculty.py` → `processed/07_akademik_kadro.md` (48 chunk) ve `processed/08_laboratuvarlar.md` (16 chunk).
- Profil URL formatı: https://akademi.itu.edu.tr/<kullanıcı> (birincil, resmi).

## AI ekosistemi (bağlam)

- İTÜ'de ayrı bir **Yapay Zeka ve Veri Mühendisliği** lisans bölümü var — https://yapayzeka.itu.edu.tr/
  (Bilgisayar Müh.'nden farklı bölüm; AI ekosistemini güçlendiren bağlam olarak kullanılabilir, karıştırma).
