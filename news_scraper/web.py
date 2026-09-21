"""Source-agnostic URL and HTML image discovery helpers."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag


IMAGE_ATTRIBUTES = ("data-src", "data-original", "data-lazy-src", "src")
SRCSET_ATTRIBUTES = ("data-srcset", "srcset")
TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonical_url(url: str) -> str:
    """Normalize a URL so tracking-only variants are treated as one resource."""
    parts = urlsplit(url)
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
         if key.lower() not in TRACKING_PARAMETERS and not key.lower().startswith("utm_")],
        doseq=True,
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def largest_srcset_candidate(srcset: str) -> str | None:
    """Return the widest (or densest) usable candidate from a srcset value."""
    candidates: list[tuple[float, str]] = []
    # Transformation URLs may contain commas. A comma followed by whitespace is
    # the reliable separator used by the source markup we support.
    for entry in re.split(r",(?=\s)", srcset):
        fields = entry.strip().split()
        if not fields:
            continue
        score = 0.0
        if len(fields) > 1:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)(w|x)", fields[-1])
            if match:
                score = float(match.group(1))
                if match.group(2) == "x":
                    score *= 10_000
        candidates.append((score, fields[0]))
    return max(candidates, default=(0.0, None))[1]


def image_urls_from_html(root: Tag, page_url: str) -> list[str]:
    """Visit a content root breadth-first and collect one URL per image node."""
    queue: deque[Tag] = deque([root])
    seen_nodes: set[int] = set()
    seen_urls: set[str] = set()
    urls: list[str] = []

    while queue:
        node = queue.popleft()
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        is_picture_fallback = node.name == "img" and isinstance(node.parent, Tag) and node.parent.name == "picture"
        candidates: list[str] = []
        if not is_picture_fallback and node.name in {"img", "source"}:
            for attribute in SRCSET_ATTRIBUTES:
                value = node.get(attribute)
                if isinstance(value, str) and (candidate := largest_srcset_candidate(value)):
                    candidates.append(candidate)
            for attribute in IMAGE_ATTRIBUTES:
                value = node.get(attribute)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
        for candidate in candidates:
            absolute = canonical_url(urljoin(page_url, candidate))
            if urlsplit(absolute).scheme in {"http", "https"} and absolute not in seen_urls:
                seen_urls.add(absolute)
                urls.append(absolute)
                break
        queue.extend(child for child in node.children if isinstance(child, Tag))
    return urls


def json_ld_articles(soup: BeautifulSoup) -> Iterable[dict]:
    """Yield Article-shaped JSON-LD objects, including objects in @graph arrays."""
    def walk(value: object) -> Iterable[dict]:
        if isinstance(value, dict):
            kind = value.get("@type")
            kinds = {kind} if isinstance(kind, str) else set(kind or []) if isinstance(kind, list) else set()
            if {"Article", "NewsArticle", "ReportageNewsArticle"} & kinds:
                yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for script in soup.select("script[type='application/ld+json']"):
        if script.string:
            try:
                yield from walk(json.loads(script.string))
            except json.JSONDecodeError:
                continue


def json_ld_image_urls(soup: BeautifulSoup, page_url: str) -> Iterable[str]:
    """Yield image URLs declared in JSON-LD metadata."""
    def walk(value: object) -> Iterable[str]:
        if isinstance(value, dict):
            image = value.get("image")
            if isinstance(image, str):
                yield image
            elif isinstance(image, list):
                yield from (item for item in image if isinstance(item, str))
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for script in soup.select("script[type='application/ld+json']"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        for image_url in walk(data):
            absolute = canonical_url(urljoin(page_url, image_url))
            if urlsplit(absolute).scheme in {"http", "https"}:
                yield absolute
