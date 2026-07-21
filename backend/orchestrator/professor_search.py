"""Official-only professor search backed by İTÜ Akademi."""

from __future__ import annotations

import asyncio
import html
import re
from typing import Any
from urllib.parse import quote_plus

import httpx


AKADEMI_BASE_URL = "https://akademi.itu.edu.tr"
_FACULTY_TITLES = ("prof", "doç", "doc", "dr. öğr", "dr. ogr")
_TAG_RE = re.compile(r"<[^>]+>")
_PERSON_RE = re.compile(
    r"<div\s+class=['\"]profil-image['\"][^>]*>\s*"
    r"<img\s+src=['\"](?P<image>[^'\"]+)['\"][^>]*>.*?"
    r"<span\s+class=['\"]title['\"]>(?P<title>.*?)</span>.*?"
    r"<span\s+class=['\"]name['\"]>(?P<name>.*?)</span>.*?"
    r"data-ajax-href=['\"]/summary/person/(?P<slug>[^/]+)/summary['\"]",
    re.IGNORECASE | re.DOTALL,
)


def _plain(fragment: str) -> str:
    text = re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", fragment)))
    return re.sub(r"\s+([,.;:])", r"\1", text).strip(" :,\n")


def parse_search_people(page: str) -> list[dict[str, str]]:
    """Extract person cards from the official Akademi search result HTML."""
    people: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _PERSON_RE.finditer(page):
        slug = match.group("slug").strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        image_url = match.group("image").strip()
        if image_url.startswith("//"):
            image_url = "https:" + image_url
        elif image_url.startswith("http://"):
            image_url = "https://" + image_url.removeprefix("http://")
        people.append(
            {
                "name": _plain(match.group("name")),
                "title": _plain(match.group("title")),
                "slug": slug,
                "image_url": image_url,
                "profile_url": f"{AKADEMI_BASE_URL}/tr/{slug}/",
            }
        )
    return people


def parse_profile(page: str) -> dict[str, str]:
    """Extract compact research metadata from an official profile page."""
    fields: dict[str, str] = {"department": "", "work_areas": "", "summary": ""}
    detail_patterns = {
        "work_areas": r"Çalışma Alanları</span>\s*<span[^>]*>(.*?)</span>",
        "department": r"Çalıştığı Birim</span>\s*<span[^>]*>(.*?)</span>",
    }
    for key, pattern in detail_patterns.items():
        match = re.search(pattern, page, re.IGNORECASE | re.DOTALL)
        if match:
            fields[key] = _plain(match.group(1))

    about = re.search(
        r"<h3[^>]*>\s*<span>HAKKINDA</span>\s*</h3>(.*?)(?:<div\s+class=['\"]social-list|</div>\s*</div>\s*</div>)",
        page,
        re.IGNORECASE | re.DOTALL,
    )
    if about:
        text = _plain(about.group(1))
        sentences = re.split(r"(?<=[.!?])\s+", text)
        fields["summary"] = " ".join(sentences[:2])[:500]
    return fields


class ItuProfessorSearch:
    """Queries and lightly enriches official İTÜ Akademi person results."""

    def __init__(self, *, timeout_s: float = 8.0, max_results: int = 5) -> None:
        self.timeout_s = timeout_s
        self.max_results = max_results

    async def search(self, topic: str, department: str = "") -> dict[str, Any]:
        topic = re.sub(r"\s+", " ", (topic or "").strip())[:120]
        department = re.sub(r"\s+", " ", (department or "").strip())[:120]
        if len(topic) < 2:
            raise ValueError("Arama konusu en az iki karakter olmalı")

        source_url = f"{AKADEMI_BASE_URL}/search-person?st={quote_plus(topic)}"
        headers = {"User-Agent": "ITU-Student-Convince-AI/1.0 (+official kiosk search)"}
        timeout = httpx.Timeout(self.timeout_s)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            response = await client.get(source_url)
            response.raise_for_status()
            candidates = parse_search_people(response.text)

            # Prefer faculty who can supervise students, while retaining the
            # official search ordering within each rank group.
            candidates = [
                person
                for person in candidates
                if any(
                    marker in person["title"].casefold()
                    for marker in _FACULTY_TITLES
                )
            ][: max(self.max_results * 2, 8)]
            profiles = await asyncio.gather(
                *(self._enrich(client, person) for person in candidates),
                return_exceptions=True,
            )

        enriched_candidates: list[dict[str, str]] = []
        department_folded = department.casefold()
        for person, profile in zip(candidates, profiles):
            enriched = dict(person)
            if isinstance(profile, dict):
                enriched.update(profile)
            else:
                enriched.update({"department": "", "work_areas": "", "summary": ""})
            if department_folded and department_folded not in (
                enriched.get("department", "").casefold()
            ):
                continue
            enriched.pop("slug", None)
            enriched_candidates.append(enriched)

        # Profiles with explicit official research areas are more useful than
        # sparse records while preserving Akademi's ordering otherwise.
        enriched_candidates.sort(key=lambda person: not bool(person["work_areas"]))
        results = enriched_candidates[: self.max_results]

        return {
            "query": topic,
            "department_filter": department,
            "results": results,
            "source_name": "İTÜ Akademi",
            "source_url": source_url,
        }

    async def _enrich(
        self, client: httpx.AsyncClient, person: dict[str, str]
    ) -> dict[str, str]:
        response = await client.get(person["profile_url"])
        response.raise_for_status()
        return parse_profile(response.text)
