# CNN article image scraper

The script crawls CNN breadth-first from a CNN home, section, or article URL. It adds
only CNN article URLs to its queue, skipping CNN section/navigation pages and all
off-site links. It uses the largest `srcset` candidate for each image and deduplicates
first by normalized URL and then by downloaded bytes.
It also compares image pixels after scaling them to a common size, so a visually
equivalent thumbnail or re-encoded variant is skipped in favor of the highest-
resolution version.
Images are retained only when OpenCV detects at least two frontal human faces.
During a crawl (and `--deduplicate-only`), existing output images with fewer than
two detected faces are deleted as well.

## Project structure

`cnn_image_scraper.py` is the CNN adapter and command-line entry point. The
source-independent pieces are available for future adapters in:

- `news_scraper/web.py` — URL cleanup plus HTML, `srcset`, and JSON-LD image discovery.
- `news_scraper/images.py` — downloading, pixel-based deduplication, and replacement of smaller image variants.

To add another publication, keep its host validation, article link rules,
article-container selectors, and filtering policy in a new source adapter; use
the shared modules for image discovery and storage.

All CNN articles are accepted. The prior topic/section filtering code remains
commented in `is_allowed_article` for easy restoration.

```powershell
python -m pip install -r requirements.txt
python cnn_image_scraper.py "https://edition.cnn.com" images --max-articles 100
```

To keep crawling until a target number of images is downloaded, use
`--min-images`. It ignores `--max-articles` and stops early only when the target is
met; otherwise it finishes after all discovered CNN articles have been visited.

```powershell
python cnn_image_scraper.py "https://edition.cnn.com" images --min-images 500
```

Clean an existing output folder without crawling again:

```powershell
python cnn_image_scraper.py "https://edition.cnn.com" downloaded_images --deduplicate-only
```

Only the start URL is required. The output folder is optional and defaults to
`downloaded_images`; the default crawl limit is 100 article pages. Supplying
`--min-images` overrides that cap.

```powershell
python -m unittest -v
```
