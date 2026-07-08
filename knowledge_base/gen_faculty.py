#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BBF akademisyen xlsx -> processed Markdown chunks (faculty + labs)."""
import openpyxl, re, unicodedata
from collections import Counter, defaultdict

SRC = "/home/aziz/life/ITU_AnthRo/itu_tanitim/knowledge_base/raw_data/academic/bbf_akademisyen_veri_seti-detayli.xlsx"
OUT_DIR = "/home/aziz/life/ITU_AnthRo/itu_tanitim/knowledge_base/processed"
DATE = "2025-07"
BBF_LABS = "https://bbf.itu.edu.tr/arastirma/ara%C5%9Ft%C4%B1rma-laboratuvarlar%C4%B1"

wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["BBF_Kadro"]
data = list(ws.iter_rows(values_only=True))
hdr = [str(h) for h in data[0]]
idx = {h: i for i, h in enumerate(hdr)}

def g(r, k):
    v = r[idx[k]]
    return "" if v in (None, "None") else str(v).strip()

def clean_list(s):
    """split on ; and , ; dedup case-insensitively preserving order."""
    parts = re.split(r"[;,]", s)
    seen, out = set(), []
    for p in parts:
        p = re.sub(r"\s*\([^)]*\)", "", p).strip()  # drop parenthetical glosses e.g. "(Artificial Intelligence)"
        if not p:
            continue
        key = unicodedata.normalize("NFKD", p.lower()).replace("ı","i")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def slug(name):
    m = {'ç':'c','ğ':'g','ı':'i','ö':'o','ş':'s','ü':'u','Ç':'c','Ğ':'g','İ':'i','Ö':'o','Ş':'s','Ü':'u'}
    s = "".join(m.get(ch, ch) for ch in name).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

rows = data[1:]
profs = [r for r in rows if g(r, "Kategori") == "Öğretim Üyesi"]

# ---- stats ----
unv = Counter(g(r, "Ünvan") for r in profs)
n_prof = sum(1 for r in profs if g(r,"Ünvan").startswith("Prof"))
n_doc = sum(1 for r in profs if g(r,"Ünvan").startswith("Doç"))
n_dr = sum(1 for r in profs if "Dr. Öğr" in g(r,"Ünvan"))
n_arasgor = sum(1 for r in rows if g(r,"Kategori")=="Araştırma Görevlisi")
n_turkcell = sum(1 for r in rows if g(r,"Kategori")=="Turkcell Destekli Araştırmacı")

# department buckets for professors
DEPT_KEYS = {
    "Bilgisayar Mühendisliği Bölümü": "Bilgisayar Mühendisliği",
    "Yapay Zeka ve Veri Mühendisliği Bölümü": "Yapay Zeka ve Veri Mühendisliği",
    "BIL Koordinatörlüğü Görevlileri": "Bilgisayar Mühendisliği (BİL Koordinatörlüğü)",
    "Yarı Zamanlı Öğretim Üyeleri": "Yarı Zamanlı Öğretim Üyeleri",
}
def dept_of(r):
    d = g(r,"Bölüm (BBF Sayfası)") or g(r,"Bölüm")
    return DEPT_KEYS.get(d, d)

chunks = []  # (id, category, source, verified, date, priority, audience_type, title, body)

# ===== 1) Faculty strength overview =====
overview = (
"İTÜ Bilgisayar ve Bilişim Fakültesi (BBF) güçlü ve geniş bir akademik kadroya sahiptir. "
f"Fakültede toplam **{len(profs)} öğretim üyesi** bulunur: yaklaşık {n_prof} Profesör, {n_doc} Doçent ve {n_dr} Dr. Öğretim Üyesi. "
f"Buna ek olarak yaklaşık {n_arasgor} araştırma görevlisi ve Turkcell destekli {n_turkcell} araştırmacı görev yapar. "
"Kadro; Bilgisayar Mühendisliği, Yapay Zeka ve Veri Mühendisliği ile Siber Güvenlik Mühendisliği bölümlerine yayılır ve "
"16'dan fazla araştırma laboratuvarında yapay zekâdan siber güvenliğe, biyoinformatikten yüksek performanslı hesaplamaya kadar geniş bir alanda çalışır. "
"Bu güçlü kadro, öğrencilere alanında uzman akademisyenlerle çalışma ve araştırma yapma imkânı sunar."
)
chunks.append(("kadro-overview","akademik",BBF_LABS,"true",DATE,"high","akademik",
               "BBF Akademik Kadro Gücü", overview))

# ===== 2) Department chunks =====
dept_group = defaultdict(list)
for r in profs:
    dept_group[dept_of(r)].append(r)

DEPT_INTRO = {
 "Bilgisayar Mühendisliği":
   "İTÜ Bilgisayar Mühendisliği Bölümü'nün öğretim üyeleri, algoritmalar ve teorik bilgisayar biliminden yapay zekâ, veri bilimi, bilgisayar ağları, gömülü sistemler ve yazılım mühendisliğine kadar geniş bir yelpazede araştırma yapar.",
 "Yapay Zeka ve Veri Mühendisliği":
   "İTÜ Yapay Zeka ve Veri Mühendisliği Bölümü'nün öğretim üyeleri; makine öğrenmesi, derin öğrenme, doğal dil işleme, bilgisayarlı görü ve büyük veri alanlarında çalışır. Bu bölüm, İTÜ'nün yapay zekâ odağını akademik olarak güçlendirir.",
}
di = 0
for dept, members in dept_group.items():
    if dept == "Yarı Zamanlı Öğretim Üyeleri":
        continue
    di += 1
    # aggregate top research areas across the dept
    areas = Counter()
    for r in members:
        for a in clean_list(g(r,"Araştırma Alanları")):
            areas[a] += 1
    top = [a for a,_ in areas.most_common(12)]
    intro = DEPT_INTRO.get(dept, f"İTÜ {dept} öğretim üyeleri geniş bir araştırma yelpazesinde çalışır.")
    named = [f"{g(r,'Ünvan')} {g(r,'Ad')} {g(r,'Soyad')}".strip() for r in members if g(r,"Ünvan").startswith(("Prof","Doç","Dr. Öğr"))]
    body = intro + f" Bölümde {len(members)} öğretim üyesi görev yapar."
    if top:
        body += " Öne çıkan araştırma alanları: " + ", ".join(top) + "."
    chunks.append((f"dept-{slug(dept)}","akademik",BBF_LABS,"true",DATE,"high","akademik",
                   f"{dept} Bölümü — Akademik Kadro ve Araştırma Alanları", body))

# ===== 3) Per-professor chunks (only those with research areas) =====
pc = 0
for r in profs:
    areas = clean_list(g(r,"Araştırma Alanları"))
    if not areas:
        continue
    pc += 1
    unvan = g(r,"Ünvan"); ad = g(r,"Ad"); soyad = g(r,"Soyad")
    full = f"{unvan} {ad} {soyad}".strip()
    dept = dept_of(r)
    lab = g(r,"Laboratuvar Adı")
    url = g(r,"Profil URL") or BBF_LABS
    hak = g(r,"Hakkında")
    body = f"{full}, İTÜ {dept} öğretim üyesidir. "
    body += "Araştırma alanları: " + ", ".join(areas[:8]) + ". "
    if lab:
        body += "İlgili laboratuvar(lar): " + lab.replace(";", ",") + ". "
    if "Lisans" in hak or "lisans" in hak or "doktora" in hak.lower():
        # first sentence of a real bio
        sent = re.split(r"(?<=[.\)])\s", hak)[0]
        if len(sent) > 30:
            body += sent.strip()
            if not body.endswith("."): body += "."
    chunks.append((f"kadro-{slug(ad+'-'+soyad)}","akademik",url,"true",DATE,"medium","akademik",
                   full + " — Araştırma Alanları", body.strip()))

# ===== 4) Lab chunks =====
# map each atomic lab -> set of research areas (from members whose lab list contains it) + count members
lab_areas = defaultdict(Counter)
lab_members = defaultdict(set)
for r in rows:
    labs = [x.strip() for x in g(r,"Laboratuvar Adı").split(";") if x.strip()]
    for lb in labs:
        lab_members[lb].add(f"{g(r,'Ad')} {g(r,'Soyad')}".strip())
        for a in clean_list(g(r,"Araştırma Alanları")):
            lab_areas[lb][a] += 1

lc = 0
for lb in sorted(lab_members):
    lc += 1
    top = [a for a,_ in lab_areas[lb].most_common(8)]
    n = len(lab_members[lb])
    body = f"İTÜ Bilgisayar ve Bilişim Fakültesi bünyesindeki {lb}, ilgili öğretim üyeleri ve araştırmacılarıyla aktif çalışmalar yürütür (yaklaşık {n} akademisyen ilişkili)."
    if top:
        body += " Başlıca çalışma konuları: " + ", ".join(top) + "."
    body += " Lisans ve lisansüstü öğrenciler bu laboratuvarda araştırma projelerinde yer alabilir."
    chunks.append((f"lab-{slug(lb)}","arastirma",BBF_LABS,"true",DATE,"medium","akademik",
                   lb, body))

# ---- write markdown files ----
def render(subset, title):
    out = [f"# {title}\n"]
    for (cid,cat,src,ver,dt,pri,aud,ttl,body) in subset:
        out.append(f"<!-- id: {cid} | category: {cat} | source: {src} | verified: {ver} | date: {dt} | priority: {pri} | audience_type: {aud} -->")
        out.append(f"## {ttl}")
        out.append(body + "\n")
    return "\n".join(out)

fac = [c for c in chunks if c[1]=="akademik"]
labs_c = [c for c in chunks if c[1]=="arastirma"]

with open(f"{OUT_DIR}/07_akademik_kadro.md","w",encoding="utf-8") as f:
    f.write(render(fac, "Akademik Kadro (akademik) — Kaynak: İTÜ BBF akademisyen veri seti"))
with open(f"{OUT_DIR}/08_laboratuvarlar.md","w",encoding="utf-8") as f:
    f.write(render(labs_c, "Araştırma Laboratuvarları (arastirma) — Kaynak: İTÜ BBF akademisyen veri seti"))

print(f"faculty chunks: {len(fac)} (overview+dept+{pc} profs) | lab chunks: {len(labs_c)}")
print("wrote 07_akademik_kadro.md and 08_laboratuvarlar.md")
