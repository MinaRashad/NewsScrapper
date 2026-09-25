"""CNN source adapter and command-line entry point.

Reusable URL/HTML parsing and image storage live in :mod:`news_scraper.web`
and :mod:`news_scraper.images`, ready for adapters for other news sources.
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag

from news_scraper.images import (
    download_image_urls, existing_image_state, remove_images_without_minimum_faces, remove_visual_duplicates,
    visual_image, visually_same,
)
from news_scraper.web import (
    canonical_url, image_urls_from_html, json_ld_articles, json_ld_image_urls,
    largest_srcset_candidate,
)


USER_AGENT = "NewsImageScraper/0.1 (+https://github.com/your-org/news-scraper)"
CNN_HOSTS = {"cnn.com", "www.cnn.com", "edition.cnn.com"}
CNN_ARTICLE_PATH = re.compile(r"/\d{4}/\d{2}/\d{2}/")
ALLOWED_TOPICS = {
    "africa", "americas", "asia", "australia", "china", "europe", "india",
    "middle east", "united kingdom", "us politics", "trump", "facts first",
    "cnn polls", "elections 2026", "redistricting tracker", "epstein files", "world",
}
US_PRESIDENT_NAMES = {
    "washington", "adams", "jefferson", "madison", "monroe", "jackson", "van buren", "harrison", "tyler", "polk",
    "taylor", "fillmore", "pierce", "buchanan", "lincoln", "johnson", "grant", "hayes", "garfield", "arthur", "cleveland",
    "mckinley", "roosevelt", "taft", "wilson", "harding", "coolidge", "hoover", "truman", "eisenhower", "kennedy", "nixon",
    "ford", "carter", "reagan", "bush", "clinton", "obama", "biden", "trump",
}
TOPIC_ALIASES = {"politics": "us politics", "u.s. politics": "us politics", "uk": "united kingdom", "u.k.": "united kingdom"}


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
    return TOPIC_ALIASES.get(normalized_text(value), normalized_text(value))


def article_title_and_topics(soup: BeautifulSoup) -> tuple[str, set[str]]:
    """Read CNN's visible headline plus JSON-LD and article-tag labels."""
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
    for link in article_root(soup).select("a[rel='tag'], a[class*='tag'], a[data-component-name*='tag']"):
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
    """Allow every CNN article; topic/section filtering is currently disabled."""
    # To restore the former topic/section filter, uncomment this block:
    # title, topics = article_title_and_topics(soup)
    # matched_topics = sorted(ALLOWED_TOPICS & topics)
    # if matched_topics:
    #     return True, f"matching tag: {matched_topics[0]}"
    # if mentions_us_president(title):
    #     return True, "a U.S. president named in the headline"
    # return False, "no allowed tag and no U.S. president named in the headline"
    return True, "topic/section filtering disabled"


# Compatibility name for existing callers; the implementation is source-agnostic.
bfs_image_urls = image_urls_from_html


def image_urls_from_article(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls = bfs_image_urls(article_root(soup), page_url)
    for image_url in json_ld_image_urls(soup, page_url):
        if image_url not in urls:
            urls.append(image_url)
    return urls


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return session


def download_cnn_article_images(start_url: str, output_dir: Path = Path("downloaded_images")) -> list[Path]:
    """Fetch one accepted CNN article and download its distinct images."""
    if cnn_page_kind(start_url) != "article":
        raise ValueError("Expected a CNN article URL with a /YYYY/MM/DD/ date path.")
    print(f"Starting article image download: {start_url}")
    session = _session()
    response = session.get(start_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    allowed, reason = is_allowed_article(soup)
    if not allowed:
        print(f"Skipped article: {reason}.")
        return []
    print(f"Article accepted ({reason}).")
    if output_dir.is_dir():
        existing_files = [path for path in output_dir.iterdir() if path.is_file()]
        print(f"Checking {len(existing_files)} existing file(s) in {output_dir} for the face requirement.")
        remove_images_without_minimum_faces(existing_files)
    hashes, visuals = existing_image_state(output_dir)
    return download_image_urls(image_urls_from_article(soup, response.url), output_dir, session, hashes, visuals)


def crawl_cnn_images(
    start_url: str, output_dir: Path = Path("downloaded_images"), max_articles: int = 100,
    min_images: int | None = None,
) -> list[Path]:
    """Crawl CNN articles until the limit, target image count, or queue exhaustion."""
    if cnn_page_kind(start_url) == "external":
        raise ValueError("The start URL must be on cnn.com, www.cnn.com, or edition.cnn.com.")
    if max_articles < 1:
        raise ValueError("max_articles must be at least 1.")
    if min_images is not None and min_images < 1:
        raise ValueError("min_images must be at least 1.")
    target_description = f"{min_images} image(s)" if min_images is not None else f"up to {max_articles} article(s)"
    print(f"Starting CNN crawl from {start_url}; target: {target_description}.")
    session = _session()
    queue: deque[str] = deque([canonical_url(start_url)])
    visited_pages: set[str] = set()
    visited_articles = 0
    if output_dir.is_dir():
        existing_files = [path for path in output_dir.iterdir() if path.is_file()]
        print(f"Checking {len(existing_files)} existing file(s) in {output_dir} for the face requirement.")
        remove_images_without_minimum_faces(existing_files)
    hashes, visuals = existing_image_state(output_dir)
    saved: list[Path] = []
    # A requested image target takes precedence over the normal article cap. This
    # lets the crawl continue until it has enough retained images or runs out of
    # newly discovered CNN articles.
    while queue and (
        len(saved) < min_images if min_images is not None else visited_articles < max_articles
    ):
        page_url = queue.popleft()
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)
        print(f"Fetching page ({visited_articles} article(s) visited, {len(saved)} image(s) downloaded, {len(queue)} queued): {page_url}")
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
        discovered_links = article_links(soup, resolved_url)
        for link in discovered_links:
            if link not in visited_pages:
                queue.append(link)
        if discovered_links:
            print(f"Discovered {len(discovered_links)} CNN article link(s); {len(queue)} page(s) now queued.")
        if cnn_page_kind(resolved_url) != "article":
            print(f"Skipping CNN section/navigation page: {resolved_url}")
            continue
        visited_articles += 1
        allowed, reason = is_allowed_article(soup)
        if not allowed:
            print(f"Skipping article ({reason}): {resolved_url}")
            continue
        print(f"Article accepted ({reason}): {resolved_url}")
        saved.extend(download_image_urls(image_urls_from_article(soup, resolved_url), output_dir, session, hashes, visuals))
        print(f"Progress: {visited_articles} article(s) visited; {len(saved)} unique image(s) retained.")
    print(f"Visited {visited_articles} article(s); downloaded {len(saved)} unique image(s).")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Breadth-first download of unique CNN article images.")
    parser.add_argument("start_url", help="CNN home, section, or article URL")
    parser.add_argument("output_dir", nargs="?", default="downloaded_images", type=Path)
    parser.add_argument("--max-articles", type=int, default=100, help="Maximum CNN article pages to visit (default: 100)")
    parser.add_argument(
        "--min-images", type=int,
        help="Continue until this many images are downloaded or no more articles remain; overrides --max-articles.",
    )
    parser.add_argument("--deduplicate-only", action="store_true", help="Remove visual size/encoding duplicates already in output_dir")
    args = parser.parse_args()
    if args.deduplicate_only:
        retained_faces, removed_faces = remove_images_without_minimum_faces(
            path for path in args.output_dir.iterdir() if path.is_file()
        )
        retained, removed = remove_visual_duplicates(path for path in args.output_dir.iterdir() if path.is_file())
        print(
            f"Retained {retained_faces} image(s) with at least two faces; removed {removed_faces}. "
            f"Retained {retained} visually distinct image(s); removed {removed} lower-resolution duplicate(s)."
        )
        return
    downloaded = crawl_cnn_images(args.start_url, args.output_dir, args.max_articles, args.min_images)
    print(f"Downloaded {len(downloaded)} unique image(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
