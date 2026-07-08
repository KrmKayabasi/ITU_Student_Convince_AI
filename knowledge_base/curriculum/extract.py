#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Üniversite müfredat (ders planı) çıkarıcı — İTÜ, Boğaziçi, ODTÜ, Koç, YTÜ.
Resmi katalog sayfalarından tam ders listelerini (kod, ad, kredi, AKTS) çeker ve
curriculum/<uni>_<bolum>.md dosyalarını üretir. Yeniden çalıştırılabilir.

Kaynak endpoint'leri: bkz. comparison/raw_data/SOURCES.md
"""
import urllib.request, ssl, re, os, sys
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
BUA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
NV = ssl.create_default_context(); NV.check_hostname=False; NV.verify_mode=ssl.CERT_NONE

def fetch(url, insecure=False, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": BUA, "Accept-Language": "tr,en"})
    with urllib.request.urlopen(req, timeout=timeout, context=(NV if insecure else None)) as r:
        return r.read().decode("utf-8", "ignore")

def cell_texts(row):
    return [re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip() for c in row.find_all(["td", "th"])]

def norm_credit(s):
    s = (s or "").replace(",", ".").strip()
    return s if re.match(r"^\d+(\.\d+)?$", s) else (s or "")

# ---------- per-site parsers: return list of (semester, code, name, credit, ects) ----------

CODEPAT = re.compile(r"^[A-ZİĞÜÇŞÖ]{2,4}\s?\d{3}")

def parse_itu(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []; sem = 0
    for t in soup.find_all("table"):
        if "Ders Kodu" not in t.get_text(" ", strip=True):
            continue  # not a course table
        sem += 1
        for r in t.find_all("tr"):
            c = cell_texts(r)
            if len(c) < 6 or not CODEPAT.match(c[0]):  # skips empty spacer + header + notes
                continue
            out.append((f"{sem}. Yarıyıl", c[0], c[1], norm_credit(c[4]), norm_credit(c[5])))
    return out

def parse_odtu(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []; sem = 0
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        head = cell_texts(rows[0]) if rows else []
        if "Course Code" not in " ".join(head):
            continue
        sem += 1
        for r in rows[1:]:
            c = cell_texts(r)
            if len(c) < 6 or not c[0] or c[0] == "Course Code":
                continue
            out.append((f"Semester {sem}", c[0], c[1], norm_credit(c[2]), norm_credit(c[5])))
    return out

def parse_koc(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []; sem = 0
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        head = cell_texts(rows[0]) if rows else []
        if "Course No" not in " ".join(head):
            continue
        sem += 1
        for r in rows[1:]:
            c = cell_texts(r)
            if len(c) < 4 or not c[0] or c[0] == "Course No":
                continue
            out.append((f"Semester {sem}", c[0], c[1], norm_credit(c[3]), ""))
    return out

def parse_ytu(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    # curriculum lives in the big table containing 'Yarıyıl' separators + 'Yerel Kredi' header
    target = None
    for t in soup.find_all("table"):
        txt = t.get_text(" ", strip=True)
        if "Yarıyıl" in txt and "Yerel Kredi" in txt:
            target = t; break
    if target is None:
        return out
    from collections import OrderedDict
    bysem = OrderedDict(); sem = "?"
    for r in target.find_all("tr"):
        c = cell_texts(r)
        if len(c) == 1 and "Yarıyıl" in c[0]:
            sem = c[0]; bysem.setdefault(sem, []); continue
        if not c or c[0] in ("Kodu", ""):
            continue
        # course row: Kodu | Önk. | Ders Adı | Ders | Uygulama | Laboratuar | Yerel Kredi | AKTS
        if len(c) >= 8 and re.match(r"^[A-ZİĞÜÇŞÖ]{2,4}\d{3,4}", c[0]):
            bysem.setdefault(sem, []).append((c[0], c[2], norm_credit(c[6]), norm_credit(c[7])))
    for sem, items in bysem.items():
        if len(items) > 20:  # YTÜ dumps its full elective pool into one term → collapse
            out.append((sem, f"({len(items)} seçmeli ders)", "Elektif havuzu — tam liste kaynakta", "", ""))
        else:
            for code, name, cr, ec in items:
                out.append((sem, code, name, cr, ec))
    return out

def parse_bogazici_cmpe(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []; sem = "?"
    for t in soup.find_all("table"):
        for r in t.find_all("tr"):
            c = cell_texts(r)
            if len(c) == 1 or (len(c) >= 2 and c[1] == "" and re.search(r"Semester|Year", c[0], re.I)):
                if re.search(r"Semester|Year", c[0], re.I): sem = c[0]
                continue
            if len(c) >= 4 and c[0] not in ("Code", "") and re.match(r"^[A-Z]{2,4}\s?\d", c[0]):
                ects = c[4] if len(c) >= 5 else ""
                out.append((sem, c[0], c[1], norm_credit(c[3]), norm_credit(ects)))
    return out

def parse_bogazici_generic(html):
    """Best-effort: any table row whose first cell looks like a BOUN course code."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    codepat = re.compile(r"^[A-Z]{2,4}\s?\d{3}")
    for t in soup.find_all("table"):
        head = " ".join(cell_texts(t.find("tr"))) if t.find("tr") else ""
        # try to locate credit/ects columns from header
        hcells = [h.lower() for h in cell_texts(t.find("tr"))] if t.find("tr") else []
        ci_credit = next((i for i,h in enumerate(hcells) if "credit" in h or "kredi" in h), None)
        ci_ects = next((i for i,h in enumerate(hcells) if "ects" in h or "akts" in h), None)
        for r in t.find_all("tr"):
            c = cell_texts(r)
            if len(c) >= 2 and codepat.match(c[0]):
                credit = norm_credit(c[ci_credit]) if ci_credit and ci_credit < len(c) else ""
                ects = norm_credit(c[ci_ects]) if ci_ects and ci_ects < len(c) else ""
                out.append(("", c[0], c[1], credit, ects))
    return out

# ---------- İTÜ planId resolver ----------

def itu_planid(program_kodu):
    html = fetch(f"https://obs.itu.edu.tr/public/DersPlan/DersPlanlariList?programKodu={program_kodu}&planTipiKodu=lisans")
    ids = [int(m) for m in re.findall(r"DersPlanDetay/(\d+)", html)]
    return max(ids) if ids else None

# ---------- targets ----------

ITU = [  # (bolum_slug, program_kodu, program_adi, dil)
    ("bilgisayar", "BLGE_LS", "Bilgisayar Mühendisliği", "%100 İngilizce"),
    ("yapay-zeka-veri", "YZVE_LS", "Yapay Zeka ve Veri Mühendisliği", "%100 İngilizce"),
    ("siber-guvenlik", "SECE_LS", "Siber Güvenlik Mühendisliği", "İngilizce"),
    ("elektrik", "ELKE_LS", "Elektrik Mühendisliği", "%100 İngilizce"),
    ("elektronik-haberlesme", "EHBE_LS", "Elektronik ve Haberleşme Mühendisliği", "%100 İngilizce"),
    ("kontrol-otomasyon", "KOME_LS", "Kontrol ve Otomasyon Mühendisliği", "%100 İngilizce"),
    ("robotik-otonom", "ROSE_LS", "Robotik ve Otonom Sistemleri Mühendisliği", "%100 İngilizce"),
    ("makina", "MAKE_LS", "Makina Mühendisliği", "%100 İngilizce"),
    ("imalat", "IMLE_LS", "İmalat Mühendisliği", "%100 İngilizce"),
    ("endustri", "ENDE_LS", "Endüstri Mühendisliği", "%100 İngilizce"),
    ("ucak", "UCK_LS", "Uçak Mühendisliği", "%30 İngilizce"),
    ("uzay", "UZBE_LS", "Uzay Mühendisliği", "%100 İngilizce"),
    ("iklim-meteoroloji", "IBM_LS", "İklim Bilimi ve Meteoroloji Mühendisliği", "%30 İngilizce"),
]
ODTU = [  # (slug, fac_prog, ad)
    ("bilgisayar", 571, "Computer Engineering"),
    ("elektrik-elektronik", 567, "Electrical and Electronics Engineering"),
    ("havacilik-uzay", 572, "Aerospace Engineering"),
    ("makina", 569, "Mechanical Engineering"),
    ("endustri", 568, "Industrial Engineering"),
    ("kimya", 563, "Chemical Engineering"),
    ("insaat", 562, "Civil Engineering"),
    ("cevre", 560, "Environmental Engineering"),
    ("gida", 573, "Food Engineering"),
    ("jeoloji", 564, "Geological Engineering"),
    ("metalurji-malzeme", 570, "Metallurgical and Materials Engineering"),
    ("maden", 565, "Mining Engineering"),
    ("petrol-dogalgaz", 566, "Petroleum and Natural Gas Engineering"),
]
KOC = [  # (slug, path_slug, ad)
    ("bilgisayar", "computer-engineering", "Computer Engineering"),
    ("elektrik-elektronik", "electrical-and-electronics-engineering", "Electrical and Electronics Engineering"),
    ("makina", "mechanical-engineering", "Mechanical Engineering"),
    ("endustri", "industrial-engineering", "Industrial Engineering"),
    ("kimya-biyoloji", "chemical-and-biological-engineering", "Chemical and Biological Engineering"),
]
YTU = [  # (slug, id, aid, ad, dil)
    ("bilgisayar", 550, 3, "Bilgisayar Mühendisliği", "%30 İngilizce"),
    ("yapay-zeka-veri", 557, 181, "Yapay Zeka ve Veri Mühendisliği", "%100 İngilizce"),
    ("elektrik", 7, 4, "Elektrik Mühendisliği", "%30 İngilizce"),
    ("elektronik-haberlesme", 6, 5, "Elektronik ve Haberleşme Mühendisliği", "%30 İngilizce"),
    ("kontrol-otomasyon", 403, 18, "Kontrol ve Otomasyon Mühendisliği", "%100 İngilizce"),
    ("makina", 391, 97, "Makine Mühendisliği", "%30 İngilizce"),
    ("endustri", 405, 32, "Endüstri Mühendisliği", "%100 İngilizce"),
    ("mekatronik", 404, 33, "Mekatronik Mühendisliği", "%100 İngilizce"),
    ("biyomedikal", 504, 151, "Biyomedikal Mühendisliği", "%100 İngilizce"),
]
BOGAZICI = [  # (slug, url, ad, parser)
    ("bilgisayar", "https://cmpe.bogazici.edu.tr/undergraduate/curriculum/", "Computer Engineering", "cmpe"),
    ("elektrik-elektronik", "https://ee.bogazici.edu.tr/tr/pages/ders-programi/1204", "Electrical and Electronics Engineering", "gen"),
    ("makina", "https://me.bogazici.edu.tr/tr/pages/ders-programi/2240", "Mechanical Engineering", "gen"),
    ("endustri", "https://ie.bogazici.edu.tr/content/academic-program", "Industrial Engineering", "gen"),
    ("insaat", "https://ce.bogazici.edu.tr/en/programs/undergraduate-program/1", "Civil Engineering", "gen"),
    ("kimya", "https://che.bogazici.edu.tr/tr/pages/ders-programi/2773", "Chemical Engineering", "gen"),
]

def write_md(uni, slug, dept_ad, dil, source, plan_note, courses, flag=""):
    path = os.path.join(HERE, f"{uni}_{slug}.md")
    lines = [f"# {UNI_ADI[uni]} — {dept_ad}", ""]
    meta = f"**Üniversite:** {UNI_ADI[uni]} · **Bölüm:** {dept_ad}"
    if dil: meta += f" · **Dil:** {dil}"
    lines += [meta, f"**Kaynak:** {source}", f"**Plan:** {plan_note} · **Ders sayısı:** {len(courses)}", ""]
    if flag:
        lines += [f"> ⚠️ {flag}", ""]
    lines += ["| Yarıyıl | Kod | Ders | Kredi | AKTS |", "|---|---|---|---|---|"]
    for sem, code, name, credit, ects in courses:
        name = name.replace("|", "/")
        lines.append(f"| {sem} | {code} | {name} | {credit} | {ects} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path

UNI_ADI = {"itu": "İTÜ (İstanbul Teknik Üniversitesi)", "bogazici": "Boğaziçi Üniversitesi",
           "odtu": "ODTÜ (Orta Doğu Teknik Üniversitesi)", "koc": "Koç Üniversitesi",
           "ytu": "YTÜ (Yıldız Teknik Üniversitesi)"}

def run():
    results = []
    only = sys.argv[1] if len(sys.argv) > 1 else "all"

    if only in ("all", "itu"):
        for slug, kod, ad, dil in ITU:
            try:
                pid = itu_planid(kod)
                if not pid:
                    results.append(("itu", slug, 0, "NO PLAN ID")); continue
                url = f"https://obs.itu.edu.tr/public/DersPlan/DersPlanDetay/{pid}"
                courses = parse_itu(fetch(url))
                flag = "" if len(courses) >= 15 else "Az ders çıktı — manuel kontrol."
                write_md("itu", slug, ad, dil, url, f"OBS planId {pid}", courses, flag)
                results.append(("itu", slug, len(courses), "OK"))
            except Exception as e:
                results.append(("itu", slug, 0, f"ERR {type(e).__name__}: {e}"))

    if only in ("all", "odtu"):
        for slug, fp, ad in ODTU:
            try:
                url = f"https://catalog.metu.edu.tr/program.php?fac_prog={fp}"
                courses = parse_odtu(fetch(url))
                flag = "" if len(courses) >= 15 else "Az ders çıktı — manuel kontrol."
                write_md("odtu", slug, ad, "%100 İngilizce", url, f"catalog fac_prog={fp}", courses, flag)
                results.append(("odtu", slug, len(courses), "OK"))
            except Exception as e:
                results.append(("odtu", slug, 0, f"ERR {type(e).__name__}: {e}"))

    if only in ("all", "koc"):
        for slug, ps, ad in KOC:
            try:
                url = f"https://eng.ku.edu.tr/en/{ps}/undergraduate/curriculum/"
                courses = parse_koc(fetch(url))
                flag = "" if len(courses) >= 15 else "Az ders çıktı — manuel kontrol."
                write_md("koc", slug, ad, "%100 İngilizce", url, "eng.ku.edu.tr curriculum", courses, flag)
                results.append(("koc", slug, len(courses), "OK"))
            except Exception as e:
                results.append(("koc", slug, 0, f"ERR {type(e).__name__}: {e}"))

    if only in ("all", "ytu"):
        for slug, pid, aid, ad, dil in YTU:
            try:
                url = f"https://bologna.yildiz.edu.tr/index.php?r=program/view&id={pid}&aid={aid}"
                courses = parse_ytu(fetch(url, insecure=True))
                flag = "" if len(courses) >= 15 else "Az ders çıktı — manuel kontrol."
                write_md("ytu", slug, ad, dil, url, f"bologna id={pid} aid={aid}", courses, flag)
                results.append(("ytu", slug, len(courses), "OK"))
            except Exception as e:
                results.append(("ytu", slug, 0, f"ERR {type(e).__name__}: {e}"))

    if only in ("all", "bogazici"):
        for slug, url, ad, ptype in BOGAZICI:
            try:
                html = fetch(url, insecure=True)
                courses = parse_bogazici_cmpe(html) if ptype == "cmpe" else parse_bogazici_generic(html)
                flag = "" if len(courses) >= 15 else "Az/temiz ders çıkmadı — bespoke CMS; manuel/agent tamamlaması gerekebilir."
                write_md("bogazici", slug, ad, "%100 İngilizce", url, "resmi bölüm sayfası", courses, flag)
                results.append(("bogazici", slug, len(courses), "OK" if len(courses) >= 15 else "FLAG"))
            except Exception as e:
                results.append(("bogazici", slug, 0, f"ERR {type(e).__name__}: {e}"))

    print(f"{'UNI':10} {'BOLUM':24} {'#DERS':>6}  STATUS")
    for uni, slug, n, st in results:
        print(f"{uni:10} {slug:24} {n:>6}  {st}")
    ok = sum(1 for *_, st in results if st == "OK")
    print(f"\nToplam: {len(results)} hedef | OK: {ok} | dosyalar: {HERE}")

if __name__ == "__main__":
    run()
