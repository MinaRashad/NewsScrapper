"""BBC article-image crawler and command-line entry point.

The crawler visits BBC pages breadth-first, but only queues URLs whose path has
the BBC article form ``/articles/<id>``.  This works for English URLs such as
``/news/articles/c98r62j4jl5mo`` as well as language sections such as
``/mundo/articles/...``.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag

from news_scraper.images import (
    download_image_urls,
    existing_image_state,
    remove_images_without_minimum_faces,
    remove_visual_duplicates,
)
from news_scraper.web import canonical_url, image_urls_from_html, json_ld_image_urls


USER_AGENT = "NewsImageScraper/0.1 (+https://github.com/your-org/news-scraper)"
BBC_HOSTS = {"bbc.com", "www.bbc.com", "bbc.co.uk", "www.bbc.co.uk"}


def bbc_page_kind(url: str) -> str:
    """Classify a link as a BBC article, BBC navigation page, or external URL."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname not in BBC_HOSTS:
        return "external"
    path_parts = [part.casefold() for part in parts.path.split("/") if part]
    try:
        article_index = path_parts.index("articles")
    except ValueError:
        return "section"
    return "article" if article_index + 1 < len(path_parts) and path_parts[article_index + 1] else "section"


def article_links(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Return distinct BBC article links in document order."""
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        absolute = canonical_url(urljoin(page_url, href))
        if bbc_page_kind(absolute) == "article" and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def article_root(soup: BeautifulSoup) -> Tag:
    """Choose BBC's article container, falling back to main then the document."""
    for selector in ("article", "[data-component='text-block']", "main"):
        found = soup.select_one(selector)
        if isinstance(found, Tag):
            return found
    return soup


# Compatibility name shared with the CNN adapter and its callers.
bfs_image_urls = image_urls_from_html


def image_urls_from_article(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Collect article-body and JSON-LD image URLs without duplicates."""
    urls = bfs_image_urls(article_root(soup), page_url)
    for image_url in json_ld_image_urls(soup, page_url):
        if image_url not in urls:
            urls.append(image_url)
    return urls


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return session


def download_bbc_article_images(start_url: str, output_dir: Path = Path("downloaded_images")) -> list[Path]:
    """Download distinct qualifying images from one BBC article."""
    if bbc_page_kind(start_url) != "article":
        raise ValueError("Expected a BBC article URL containing /articles/<id>.")
    session = _session()
    response = session.get(start_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    hashes, visuals = existing_image_state(output_dir)
    return download_image_urls(image_urls_from_article(soup, response.url), output_dir, session, hashes, visuals)


def crawl_bbc_images(
    start_url: str, output_dir: Path = Path("downloaded_images"), max_articles: int = 100,
    min_images: int | None = None,
) -> list[Path]:
    """Crawl BBC articles breadth-first until a limit, target, or queue exhaustion."""
    if bbc_page_kind(start_url) == "external":
        raise ValueError("The start URL must be on bbc.com or bbc.co.uk.")
    if max_articles < 1:
        raise ValueError("max_articles must be at least 1.")
    if min_images is not None and min_images < 1:
        raise ValueError("min_images must be at least 1.")

    session = _session()
    queue: deque[str] = deque([canonical_url(start_url)])
    visited_pages: set[str] = set()
    visited_articles = 0
    hashes, visuals = existing_image_state(output_dir)
    saved: list[Path] = []

    while queue and (len(saved) < min_images if min_images is not None else visited_articles < max_articles):
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
        if bbc_page_kind(resolved_url) == "external":
            print(f"Skipping redirected off-site page: {resolved_url}")
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for link in article_links(soup, resolved_url):
            if link not in visited_pages:
                queue.append(link)
        if bbc_page_kind(resolved_url) != "article":
            continue

        visited_articles += 1
        saved.extend(download_image_urls(image_urls_from_article(soup, resolved_url), output_dir, session, hashes, visuals))

    print(f"Visited {visited_articles} BBC article(s); downloaded {len(saved)} unique image(s).")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Breadth-first download of unique BBC article images.")
    parser.add_argument("start_url", help="BBC home, section, or article URL")
    parser.add_argument("output_dir", nargs="?", default="downloaded_images", type=Path)
    parser.add_argument("--max-articles", type=int, default=100, help="Maximum BBC article pages to visit (default: 100)")
    parser.add_argument("--min-images", type=int, help="Continue until this many images are downloaded; overrides --max-articles.")
    parser.add_argument("--deduplicate-only", action="store_true", help="Remove visual duplicates and images with fewer than two faces")
    args = parser.parse_args()
    if args.deduplicate_only:
        files = [path for path in args.output_dir.iterdir() if path.is_file()]
        retained_faces, removed_faces = remove_images_without_minimum_faces(files)
        retained, removed = remove_visual_duplicates(path for path in args.output_dir.iterdir() if path.is_file())
        print(f"Retained {retained_faces} image(s) with at least two faces; removed {removed_faces}. Retained {retained} visually distinct image(s); removed {removed} duplicate(s).")
        return
    downloaded = crawl_bbc_images(args.start_url, args.output_dir, args.max_articles, args.min_images)
    print(f"Downloaded {len(downloaded)} unique image(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
