import unittest
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from bbc_image_scraper import article_links, bbc_page_kind, crawl_bbc_images


class FakePageResponse:
    def __init__(self, url: str, text: str):
        self.url = url
        self.text = text

    def raise_for_status(self):
        pass


class FakePageSession:
    def __init__(self, responses):
        self.responses = responses
        self.requested_urls = []

    def get(self, url, timeout):
        self.requested_urls.append(url)
        return self.responses[url]


class BBCImageScraperTests(unittest.TestCase):
    def test_articles_path_is_recognized_in_english_and_language_sections(self):
        self.assertEqual(bbc_page_kind("https://www.bbc.com/news/articles/c98r62j4jl5mo"), "article")
        self.assertEqual(bbc_page_kind("https://www.bbc.com/mundo/articles/c1mn6n6x0dyo"), "article")
        self.assertEqual(bbc_page_kind("https://www.bbc.com/news"), "section")
        self.assertEqual(bbc_page_kind("https://example.com/articles/c98r62j4jl5mo"), "external")

    def test_article_links_skip_navigation_and_external_urls(self):
        soup = BeautifulSoup(
            '<a href="/news">News</a><a href="/news/articles/c98r62j4jl5mo">Story</a>'
            '<a href="/mundo/articles/c1mn6n6x0dyo">Mundo</a><a href="https://example.com/articles/x">Elsewhere</a>',
            "html.parser",
        )
        self.assertEqual(
            article_links(soup, "https://www.bbc.com"),
            ["https://www.bbc.com/news/articles/c98r62j4jl5mo", "https://www.bbc.com/mundo/articles/c1mn6n6x0dyo"],
        )

    def test_crawl_is_breadth_first(self):
        home = "https://www.bbc.com/news"
        first = "https://www.bbc.com/news/articles/first"
        second = "https://www.bbc.com/news/articles/second"
        session = FakePageSession({
            home: FakePageResponse(home, f'<a href="{first}">First</a><a href="{second}">Second</a>'),
            first: FakePageResponse(first, "<article></article>"),
            second: FakePageResponse(second, "<article></article>"),
        })
        with patch("bbc_image_scraper._session", return_value=session), patch(
            "bbc_image_scraper.download_image_urls", side_effect=[[Path("first.png")], [Path("second.png")]]
        ):
            saved = crawl_bbc_images(home, max_articles=2)
        self.assertEqual(saved, [Path("first.png"), Path("second.png")])
        self.assertEqual(session.requested_urls, [home, first, second])


if __name__ == "__main__":
    unittest.main()
