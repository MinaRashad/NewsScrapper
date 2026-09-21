"""Download the distinct images contained in one CNN article.

Usage:
    python cnn_image_scraper.py "https://www.cnn.com/..." [output_directory]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from PIL import Image, UnidentifiedImageError


USER_AGENT = "NewsImageScraper/0.1 (+https://github.com/your-org/news-scraper)"
CNN_HOSTS = {"cnn.com", "www.cnn.com", "edition.cnn.com"}
CNN_ARTICLE_PATH = re.compile(r"/\d{4}/\d{2}/\d{2}/")
IMAGE_ATTRIBUTES = ("data-src", "data-original", "data-lazy-src", "src")
SRCSET_ATTRIBUTES = ("data-srcset", "srcset")
TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
IMAGE_EXTENSIONS = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
ALLOWED_TOPICS = {
    "africa", "americas", "asia", "australia", "china", "europe", "india",
    "middle east", "united kingdom", "us politics", "trump", "facts first",
    "cnn polls", "elections 2026", "redistricting tracker", "epstein files", "world"
}
# Surnames are intentionally used: CNN headlines commonly omit the word "President".
US_PRESIDENT_NAMES = {
    "washington", "adams", "jefferson", "madison", "monroe", "jackson",
    "van buren", "harrison", "tyler", "polk", "taylor", "fillmore", "pierce",
    "buchanan", "lincoln", "johnson", "grant", "hayes", "garfield", "arthur",
    "cleveland", "mckinley", "roosevelt", "taft", "wilson", "harding", "coolidge",
    "hoover", "truman", "eisenhower", "kennedy", "nixon", "ford", "carter",
    "reagan", "bush", "clinton", "obama", "biden", "trump",
}
TOPIC_ALIASES = {
    "politics": "us politics",
    "u.s. politics": "us politics",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
}


def canonical_url(url: str) -> str:
    """Normalize a URL so tracking-only variants are treated as one image."""
    parts = urlsplit(url)
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
         if key.lower() not in TRACKING_PARAMETERS and not key.lower().startswith("utm_")],
        doseq=True,
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def cnn_page_kind(url: str) -> str:
    """Classify a link without visiting it: article, section/navigation, or external."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname not in CNN_HOSTS:
        return "external"
    return "article" if CNN_ARTICLE_PATH.search(parts.path) else "section"


def article_links(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Return only new CNN article links; section and off-site links are excluded."""
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        absolute = canonical_url(urljoin(page_url, href))
        if cnn_page_kind(absolute) == "article" and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def largest_srcset_candidate(srcset: str) -> str | None:
    """Return the widest (or densest) usable candidate from a srcset value."""
    candidates: list[tuple[float, str]] = []
    # CNN image transformation URLs contain commas (for example `c_fill,w_1280`).
    # A comma followed by whitespace starts a new candidate. CNN's transformation
    # commas have no following whitespace (for example `c_fill,w_1280`).
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


def article_root(soup: BeautifulSoup) -> Tag:
    """Choose CNN's article container, falling back to main then the document."""
    for selector in ("article", "[data-component-name='article-body']", ".article__content", "main"):
        found = soup.select_one(selector)
        if isinstance(found, Tag):
            return found
    return soup


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def normalized_topic(value: str) -> str:
    topic = normalized_text(value)
    return TOPIC_ALIASES.get(topic, topic)


def json_ld_articles(soup: BeautifulSoup) -> Iterable[dict]:
    """Yield article-shaped JSON-LD objects, including objects in @graph arrays."""
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
        if not script.string:
            continue
        try:
            yield from walk(json.loads(script.string))
        except json.JSONDecodeError:
            continue


def article_title_and_topics(soup: BeautifulSoup) -> tuple[str, set[str]]:
    """Read CNN's visible headline plus its JSON-LD and article-tag labels."""
    headline = soup.select_one("article h1, main h1, h1")
    title = headline.get_text(" ", strip=True) if isinstance(headline, Tag) else ""
    topics: set[str] = set()
    for article in json_ld_articles(soup):
        if not title and isinstance(article.get("headline"), str):
            title = article["headline"]
        section = article.get("articleSection")
        if isinstance(section, str):
            topics.add(normalized_topic(section))
        elif isinstance(section, list):
            topics.update(normalized_topic(item) for item in section if isinstance(item, str))

    root = article_root(soup)
    # CNN tags are links near the story metadata; only use links within the article,
    # avoiding unrelated navigation labels elsewhere on the page.
    for link in root.select("a[rel='tag'], a[class*='tag'], a[data-component-name*='tag']"):
        label = link.get_text(" ", strip=True)
        if label:
            topics.add(normalized_topic(label))
    for metadata in soup.select("meta[name='keywords'], meta[property='article:section']"):
        content = metadata.get("content")
        if isinstance(content, str):
            topics.update(normalized_topic(item) for item in content.split(",") if item.strip())
    if not title:
        metadata_title = soup.select_one("meta[property='og:title'], meta[name='twitter:title']")
        if isinstance(metadata_title, Tag) and isinstance(metadata_title.get("content"), str):
            title = metadata_title["content"]
    return title, topics


def mentions_us_president(title: str) -> bool:
    normalized_title = normalized_text(title)
    return any(re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", normalized_title) for name in US_PRESIDENT_NAMES)


def is_allowed_article(soup: BeautifulSoup) -> tuple[bool, str]:
    """Decide whether this article matches the user's topic or headline rules."""
    title, topics = article_title_and_topics(soup)
    matched_topics = sorted(ALLOWED_TOPICS & topics)
    if matched_topics:
        return True, f"matching tag: {matched_topics[0]}"
    if mentions_us_president(title):
        return True, "a U.S. president named in the headline"
    return False, "no allowed tag and no U.S. president named in the headline"


def bfs_image_urls(root: Tag, page_url: str) -> list[str]:
    """Visit article elements breadth-first and collect one best URL per image node."""
    queue: deque[Tag] = deque([root])
    seen_nodes: set[int] = set()
    seen_urls: set[str] = set()
    urls: list[str] = []

    while queue:
        node = queue.popleft()
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        # In a picture, its source child is the selected image.  Do not also save
        # the img fallback as a second article image.
        is_picture_fallback = node.name == "img" and isinstance(node.parent, Tag) and node.parent.name == "picture"
        candidates: list[str] = []
        # Do not treat scripts, iframes, and other embedded resources as images
        # merely because they also use a `src` attribute.
        if not is_picture_fallback and node.name in {"img", "source"}:
            for attribute in SRCSET_ATTRIBUTES:
                value = node.get(attribute)
                if isinstance(value, str):
                    candidate = largest_srcset_candidate(value)
                    if candidate:
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


def json_ld_image_urls(soup: BeautifulSoup, page_url: str) -> Iterable[str]:
    """CNN occasionally exposes lead images only in JSON-LD metadata."""
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


def extension_for(response: requests.Response, image_url: str) -> str:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    guessed = mimetypes.guess_extension(content_type) if content_type else None
    if guessed == ".jpe":
        return ".jpg"
    if guessed in IMAGE_EXTENSIONS:
        return guessed
    suffix = Path(urlsplit(image_url).path).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


@dataclass
class VisualImage:
    path: Path
    width: int
    height: int
    pixels: bytes

    @property
    def area(self) -> int:
        return self.width * self.height


def visual_image(path: Path, image_bytes: bytes) -> VisualImage | None:
    """Create a small, size-independent pixel representation for image comparison."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            # Comparing a common small image makes a thumbnail comparable to its
            # full-size source while avoiding expensive full-resolution comparisons.
            normalized = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
            return VisualImage(path, width, height, normalized.tobytes())
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def visually_same(first: VisualImage, second: VisualImage) -> bool:
    """Conservatively identify resize/re-encoding variants of the same picture."""
    first_ratio = first.width / first.height
    second_ratio = second.width / second.height
    if abs(first_ratio - second_ratio) > 0.02:
        return False
    mean_difference = sum(abs(left - right) for left, right in zip(first.pixels, second.pixels)) / len(first.pixels)
    return mean_difference <= 6


def remove_visual_duplicates(paths: Iterable[Path]) -> tuple[int, int]:
    """Keep the highest-resolution copy of each visually equivalent image set."""
    retained: list[VisualImage] = []
    removed = 0
    for path in paths:
        try:
            candidate = visual_image(path, path.read_bytes())
        except OSError:
            continue
        if candidate is None:
            continue
        equivalent = next((image for image in retained if visually_same(image, candidate)), None)
        if equivalent is None:
            retained.append(candidate)
            continue
        if candidate.area > equivalent.area:
            equivalent.path.unlink()
            retained[retained.index(equivalent)] = candidate
        else:
            path.unlink()
        removed += 1
    return len(retained), removed


def existing_image_state(output_dir: Path) -> tuple[set[str], list[VisualImage]]:
    """Load prior downloads so repeated crawls do not re-save their variants."""
    content_hashes: set[str] = set()
    visual_images: list[VisualImage] = []
    if not output_dir.is_dir():
        return content_hashes, visual_images
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        try:
            body = path.read_bytes()
        except OSError:
            continue
        content_hashes.add(hashlib.sha256(body).hexdigest())
        image = visual_image(path, body)
        if image is not None:
            visual_images.append(image)
    return content_hashes, visual_images


def download_image_urls(
    image_urls: Iterable[str], output_dir: Path, session: requests.Session, content_hashes: set[str], visual_images: list[VisualImage]
) -> list[Path]:
    """Save image URLs once globally, retaining the largest visual variant.

    Pages often expose a thumbnail before the matching full-size image (for
    example, through JSON-LD).  A larger later download replaces every smaller
    visually matching file, including files that were present before this run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for position, image_url in enumerate(image_urls, start=1):
        try:
            image_response = session.get(image_url, timeout=30)
            image_response.raise_for_status()
        except requests.RequestException as error:
            print(f"Skipping {image_url}: {error}")
            continue
        if not image_response.headers.get("Content-Type", "").lower().startswith("image/"):
            print(f"Skipping non-image response: {image_url}")
            continue
        body = image_response.content
        digest = hashlib.sha256(body).hexdigest()
        if digest in content_hashes:
            continue
        destination = output_dir / f"{len(content_hashes):05d}-{digest[:12]}{extension_for(image_response, image_url)}"
        candidate = visual_image(destination, body)
        equivalents = [image for image in visual_images if candidate and visually_same(image, candidate)]
        # Never replace a matching image with an equal- or lower-resolution copy.
        if equivalents and candidate is not None and candidate.area <= max(image.area for image in equivalents):
            continue
        try:
            # Write first: a failed download/write must not destroy the existing
            # thumbnail or full-size file it would otherwise replace.
            destination.write_bytes(body)
        except OSError as error:
            print(f"Skipping {image_url}: could not save image ({error})")
            continue
        content_hashes.add(digest)
        saved.append(destination)
        if candidate is not None:
            for equivalent in equivalents:
                try:
                    old_digest = hashlib.sha256(equivalent.path.read_bytes()).hexdigest()
                    equivalent.path.unlink()
                except OSError as error:
                    print(f"Could not remove lower-resolution duplicate {equivalent.path.name}: {error}")
                    continue
                content_hashes.discard(old_digest)
                visual_images.remove(equivalent)
                if equivalent.path in saved:
                    saved.remove(equivalent.path)
            visual_images.append(candidate)
        print(f"Saved {destination.name}")
    return saved


def image_urls_from_article(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls = bfs_image_urls(article_root(soup), page_url)
    for image_url in json_ld_image_urls(soup, page_url):
        if image_url not in urls:
            urls.append(image_url)
    return urls


def download_cnn_article_images(
    start_url: str, output_dir: Path = Path("downloaded_images")
) -> list[Path]:
    """Fetch one CNN article and save its distinct images if it passes the filter."""
    if cnn_page_kind(start_url) != "article":
        raise ValueError("Expected a CNN article URL with a /YYYY/MM/DD/ date path.")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    response = session.get(start_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    allowed, reason = is_allowed_article(soup)
    if not allowed:
        print(f"Skipped article: {reason}.")
        return []
    print(f"Article accepted ({reason}).")
    content_hashes, visual_images = existing_image_state(output_dir)
    return download_image_urls(image_urls_from_article(soup, response.url), output_dir, session, content_hashes, visual_images)


def crawl_cnn_images(start_url: str, output_dir: Path = Path("downloaded_images"), max_articles: int = 100) -> list[Path]:
    """Breadth-first crawl from a CNN start page, visiting only article links.

    A home/section page is used solely to discover article links. It is never treated
    as an article and its images are never downloaded.
    """
    if cnn_page_kind(start_url) == "external":
        raise ValueError("The start URL must be on cnn.com, www.cnn.com, or edition.cnn.com.")
    if max_articles < 1:
        raise ValueError("max_articles must be at least 1.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    queue: deque[str] = deque([canonical_url(start_url)])
    visited_pages: set[str] = set()
    visited_articles = 0
    content_hashes, visual_images = existing_image_state(output_dir)
    saved: list[Path] = []

    while queue and visited_articles < max_articles:
        page_url = queue.popleft()
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)
        try:
            response = session.get(page_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Skipping page {page_url}: {error}")
            continue
        resolved_url = canonical_url(response.url)
        if cnn_page_kind(resolved_url) == "external":
            print(f"Skipping redirected off-site page: {resolved_url}")
            continue
        soup = BeautifulSoup(response.text, "html.parser")

        # Add only article links, in document order, to preserve BFS traversal.
        for link in article_links(soup, resolved_url):
            if link not in visited_pages:
                queue.append(link)
        if cnn_page_kind(resolved_url) != "article":
            print(f"Skipping CNN section/navigation page: {resolved_url}")
            continue

        visited_articles += 1
        allowed, reason = is_allowed_article(soup)
        if not allowed:
            print(f"Skipping article ({reason}): {resolved_url}")
            continue
        print(f"Article accepted ({reason}): {resolved_url}")
        saved.extend(download_image_urls(image_urls_from_article(soup, resolved_url), output_dir, session, content_hashes, visual_images))

    print(f"Visited {visited_articles} article(s); downloaded {len(saved)} unique image(s).")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Breadth-first download of unique images from relevant CNN articles.")
    parser.add_argument("start_url", help="CNN home, section, or article URL")
    parser.add_argument("output_dir", nargs="?", default="downloaded_images", type=Path)
    parser.add_argument("--max-articles", type=int, default=100, help="Maximum CNN article pages to visit (default: 100)")
    parser.add_argument("--deduplicate-only", action="store_true", help="Remove visual size/encoding duplicates already in output_dir")
    args = parser.parse_args()
    if args.deduplicate_only:
        retained, removed = remove_visual_duplicates(path for path in args.output_dir.iterdir() if path.is_file())
        print(f"Retained {retained} visually distinct image(s); removed {removed} lower-resolution duplicate(s).")
        return
    downloaded = crawl_cnn_images(args.start_url, args.output_dir, args.max_articles)
    print(f"Downloaded {len(downloaded)} unique image(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
