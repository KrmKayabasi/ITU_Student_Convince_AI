#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metadata.json'u processed/*.md içindeki chunk metadata yorumlarından yeniden üretir.
Markdown = tek doğru kaynak (source of truth). Ekip Markdown'ı düzenledikçe bu betiği
çalıştırıp metadata.json'u güncel tutar.

Kullanım:  python3 build_metadata.py
Chunk formatı:
  <!-- id: X | category: Y | source: URL | verified: true | date: 2025-07 | priority: high | audience_type: akademik -->
  ## Başlık
"""
import re, json, glob, os
from collections import Counter

KB = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(KB, "processed")

CHUNK_RE = re.compile(
    r"<!--\s*(?P<meta>id:.*?)-->\s*\n\s*#{1,6}\s*(?P<title>.+)")

def parse_meta(s):
    d = {}
    for part in s.split("|"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        d[k.strip()] = v.strip()
    return d

chunks = []
for md in sorted(glob.glob(os.path.join(PROCESSED, "*.md"))):
    rel = os.path.relpath(md, KB)
    txt = open(md, encoding="utf-8").read()
    for m in CHUNK_RE.finditer(txt):
        meta = parse_meta(m.group("meta"))
        title = m.group("title").strip()
        if "id" not in meta:
            continue
        meta["verified"] = str(meta.get("verified", "")).lower() == "true"
        chunks.append({
            "id": meta.get("id"),
            "category": meta.get("category"),
            "source": meta.get("source"),
            "verified": meta["verified"],
            "date": meta.get("date"),
            "priority": meta.get("priority"),
            "audience_type": meta.get("audience_type"),
            "file": rel,
            "title": title,
        })

ids = [c["id"] for c in chunks]
dups = [i for i, n in Counter(ids).items() if n > 1]
if dups:
    raise SystemExit(f"HATA: yinelenen id'ler: {dups}")

bad_date = [c["id"] for c in chunks if not c["date"] or int(c["date"][:4]) < 2023]
if bad_date:
    print("UYARI: 2023 öncesi/eksik tarihli chunk'lar:", bad_date)

by_cat = Counter(c["category"] for c in chunks)
vt = sum(1 for c in chunks if c["verified"])

doc = {
    "knowledge_base": "İTÜ Bilgisayar Mühendisliği — AI Tanıtım Sistemi",
    "work_package": "B — Data & Knowledge Base",
    "version": "0.2",
    "generated": "2026-07-08",
    "language": "tr",
    "schema": ["id", "category", "source", "verified", "date", "priority", "audience_type", "file", "title"],
    "notes": "metadata.json build_metadata.py tarafından processed/*.md'den otomatik üretilir. "
             "verified=false => proxy/ikincil veri (metinde etiketli). TODO işaretli chunk'lar resmi "
             "İTÜ-Bilgisayar rakamıyla güncellenmeli. Denetim izi raw_data/*/SOURCES.md içindedir.",
    "stats": {
        "total_chunks": len(chunks),
        "by_category": dict(by_cat),
        "verified_true": vt,
        "verified_false_proxy": len(chunks) - vt,
        "chunk_target_pdf": 200,
    },
    "chunks": chunks,
}

with open(os.path.join(KB, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)

print(f"metadata.json yazıldı: {len(chunks)} chunk | kategoriler: {dict(by_cat)} | doğrulanmış: {vt}")
