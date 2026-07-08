#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison/metadata.json'u comparison/processed/*.md'den yeniden üretir.
ANA KB'nin knowledge_base/metadata.json'ından AYRIDIR (bot KB'si temiz kalır; bu veri DAHİLİ).
Kullanım: python3 comparison/build_metadata.py
"""
import re, json, glob, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "processed")
CHUNK_RE = re.compile(r"<!--\s*(?P<meta>id:.*?)-->\s*\n\s*#{1,6}\s*(?P<title>.+)")

def parse_meta(s):
    d = {}
    for part in s.split("|"):
        if ":" in part:
            k, v = part.split(":", 1); d[k.strip()] = v.strip()
    return d

chunks = []
for md in sorted(glob.glob(os.path.join(PROCESSED, "*.md"))):
    rel = os.path.relpath(md, HERE)
    for m in CHUNK_RE.finditer(open(md, encoding="utf-8").read()):
        meta = parse_meta(m.group("meta"))
        if "id" not in meta:
            continue
        meta["verified"] = str(meta.get("verified", "")).lower() == "true"
        chunks.append({
            "id": meta.get("id"), "category": meta.get("category"), "source": meta.get("source"),
            "verified": meta["verified"], "date": meta.get("date"), "priority": meta.get("priority"),
            "audience_type": meta.get("audience_type"), "file": rel, "title": m.group("title").strip(),
        })

ids = [c["id"] for c in chunks]
dups = [i for i, n in Counter(ids).items() if n > 1]
if dups:
    raise SystemExit(f"HATA: yinelenen id: {dups}")
bad_date = [c["id"] for c in chunks if not c["date"] or int(c["date"][:4]) < 2023]
if bad_date:
    print("UYARI: 2023 öncesi/eksik tarih:", bad_date)
non_internal = [c["id"] for c in chunks if c["audience_type"] != "internal"]
if non_internal:
    print("UYARI: audience_type 'internal' değil:", non_internal)

doc = {
    "collection": "İTÜ Mühendislik — Üniversiteler-Arası Karşılaştırma (DAHİLİ)",
    "note": "DAHİLİ kullanım — nötr, kaynaklı; PDF §4 gereği kötüleme yok, public bot'a doğrudan beslenmez. "
            "Ana knowledge_base/metadata.json'dan AYRIDIR. build: python3 comparison/build_metadata.py",
    "version": "0.1", "generated": "2026-07-08", "language": "tr",
    "schema": ["id", "category", "source", "verified", "date", "priority", "audience_type", "file", "title"],
    "stats": {"total_chunks": len(chunks), "by_category": dict(Counter(c["category"] for c in chunks)),
              "verified_true": sum(1 for c in chunks if c["verified"])},
    "chunks": chunks,
}
with open(os.path.join(HERE, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
print(f"comparison/metadata.json: {len(chunks)} chunk | {dict(Counter(c['category'] for c in chunks))}")
