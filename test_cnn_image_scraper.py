import unittest
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

from cnn_image_scraper import (
    article_links,
    bfs_image_urls,
    canonical_url,
    cnn_page_kind,
    existing_image_state,
    download_image_urls,
    is_allowed_article,
    largest_srcset_candidate,
    visual_image,
    visually_same,
)


class FakeImageResponse:
    def __init__(self, body: bytes):
        self.content = body
        self.headers = {"Content-Type": "image/png"}

    def raise_for_status(self):
        pass


class FakeImageSession:
    def __init__(self, responses: dict[str, bytes]):
        self.responses = responses

    def get(self, url: str, timeout: int):
        return FakeImageResponse(self.responses[url])


class CNNImageScraperTests(unittest.TestCase):
    test_output_dir = Path(".test_image_replacement_output")

    def setUp(self):
        self.test_output_dir.mkdir(exist_ok=True)
        for path in self.test_output_dir.iterdir():
            if path.is_file():
                path.unlink()

    def tearDown(self):
        for path in self.test_output_dir.iterdir():
            if path.is_file():
                path.unlink()
        self.test_output_dir.rmdir()

    @staticmethod
    def png_bytes(size: tuple[int, int]) -> bytes:
        image = BytesIO()
        Image.new("RGB", size, (35, 110, 180)).save(image, "PNG")
        return image.getvalue()

    def test_visual_comparison_matches_resized_variant(self):
        small = BytesIO()
        large = BytesIO()
        Image.new("RGB", (100, 50), (35, 110, 180)).save(small, "PNG")
        Image.new("RGB", (1000, 500), (35, 110, 180)).save(large, "PNG")
        small_image = visual_image(Path("small.png"), small.getvalue())
        large_image = visual_image(Path("large.png"), large.getvalue())
        self.assertIsNotNone(small_image)
        self.assertIsNotNone(large_image)
        self.assertTrue(visually_same(small_image, large_image))

    def test_larger_later_variant_replaces_thumbnail(self):
        thumbnail = self.png_bytes((100, 50))
        full_size = self.png_bytes((1000, 500))
        saved = download_image_urls(
            ["https://example.test/thumb.png", "https://example.test/full.png"],
            self.test_output_dir,
            FakeImageSession({"https://example.test/thumb.png": thumbnail, "https://example.test/full.png": full_size}),
            set(),
            [],
        )
        files = list(self.test_output_dir.iterdir())
        self.assertEqual(len(files), 1)
        with Image.open(files[0]) as image:
            self.assertEqual(image.size, (1000, 500))
        self.assertEqual(saved, files)

    def test_larger_variant_replaces_existing_thumbnail(self):
        thumbnail = self.png_bytes((100, 50))
        full_size = self.png_bytes((1000, 500))
        old_path = self.test_output_dir / "thumbnail.png"
        old_path.write_bytes(thumbnail)
        hashes, images = existing_image_state(self.test_output_dir)
        download_image_urls(
            ["https://example.test/full.png"], self.test_output_dir,
            FakeImageSession({"https://example.test/full.png": full_size}), hashes, images,
        )
        files = list(self.test_output_dir.iterdir())
        self.assertEqual(len(files), 1)
        self.assertNotEqual(files[0], old_path)
        with Image.open(files[0]) as image:
            self.assertEqual(image.size, (1000, 500))

    def test_missing_output_folder_has_empty_existing_state(self):
        hashes, images = existing_image_state(Path("folder-that-does-not-exist"))
        self.assertEqual(hashes, set())
        self.assertEqual(images, [])

    def test_only_date_pattern_cnn_urls_are_articles(self):
        self.assertEqual(cnn_page_kind("https://edition.cnn.com/2026/09/18/world/story"), "article")
        self.assertEqual(cnn_page_kind("https://edition.cnn.com/world"), "section")
        self.assertEqual(cnn_page_kind("https://www.bbc.com/news"), "external")

    def test_article_links_skip_section_and_off_site_links(self):
        soup = BeautifulSoup(
            '<a href="/world">World</a><a href="/2026/09/18/world/story">Story</a><a href="https://www.bbc.com/news">BBC</a>',
            "html.parser",
        )
        self.assertEqual(article_links(soup, "https://edition.cnn.com"), ["https://edition.cnn.com/2026/09/18/world/story"])

    def test_srcset_prefers_largest_image(self):
        self.assertEqual(largest_srcset_candidate("small.jpg 320w, large.jpg 1200w"), "large.jpg")

    def test_srcset_keeps_commas_inside_cnn_image_url(self):
        self.assertEqual(
            largest_srcset_candidate(
                "https://cdn.cnn.com/image/upload/c_fill,w_320/photo.jpg 320w, https://cdn.cnn.com/image/upload/c_fill,w_1280/photo.jpg 1280w"
            ),
            "https://cdn.cnn.com/image/upload/c_fill,w_1280/photo.jpg",
        )

    def test_bfs_collects_article_images_once_in_bfs_order(self):
        soup = BeautifulSoup(
            "<article><img src='/first.jpg'><div><picture><source srcset='/wide.jpg 1200w, /narrow.jpg 320w'><img src='/fallback.jpg'></picture></div><img src='/first.jpg'></article>",
            "html.parser",
        )
        self.assertEqual(
            bfs_image_urls(soup.article, "https://www.cnn.com/story"),
            ["https://www.cnn.com/first.jpg", "https://www.cnn.com/wide.jpg"],
        )

    def test_bfs_ignores_script_and_iframe_sources(self):
        soup = BeautifulSoup('<article><script src="tracker.js"></script><iframe src="embed.html"></iframe><img src="photo.jpg"></article>', "html.parser")
        self.assertEqual(bfs_image_urls(soup.article, "https://www.cnn.com/story"), ["https://www.cnn.com/photo.jpg"])

    def test_canonical_url_removes_tracking(self):
        self.assertEqual(canonical_url("https://CDN.CNN.com/a.jpg?utm_source=x&size=large#caption"), "https://cdn.cnn.com/a.jpg?size=large")

    def test_allowed_json_ld_section_passes(self):
        soup = BeautifulSoup(
            '<script type="application/ld+json">{"@type":"NewsArticle", "headline":"A story", "articleSection":"Africa"}</script>',
            "html.parser",
        )
        self.assertTrue(is_allowed_article(soup)[0])

    def test_keyword_alias_for_us_politics_passes(self):
        soup = BeautifulSoup('<meta name="keywords" content="News, Politics, Election">', "html.parser")
        self.assertTrue(is_allowed_article(soup)[0])

    def test_president_named_in_headline_passes_even_with_health_tag(self):
        soup = BeautifulSoup(
            '<h1>Trump split up the nation’s immunization program</h1><script type="application/ld+json">{"@type":"NewsArticle", "articleSection":"Health"}</script>',
            "html.parser",
        )
        self.assertTrue(is_allowed_article(soup)[0])

    def test_unrelated_section_and_headline_are_rejected(self):
        soup = BeautifulSoup(
            '<h1>How to sleep better tonight</h1><script type="application/ld+json">{"@type":"NewsArticle", "articleSection":"Health"}</script>',
            "html.parser",
        )
        self.assertFalse(is_allowed_article(soup)[0])


if __name__ == "__main__":
    unittest.main()
