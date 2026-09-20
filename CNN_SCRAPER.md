# CNN article image scraper

The script crawls CNN breadth-first from a CNN home, section, or article URL. It adds
only CNN article URLs to its queue, skipping CNN section/navigation pages and all
off-site links. It uses the largest `srcset` candidate for each image and deduplicates
first by normalized URL and then by downloaded bytes.
It also compares image pixels after scaling them to a common size, so a visually
equivalent thumbnail or re-encoded variant is skipped in favor of the highest-
resolution version.

Before downloading, it accepts only articles tagged as Africa, Americas, Asia,
Australia, China, Europe, India, Middle East, United Kingdom, US Politics, Trump,
Facts First, CNN Polls, Elections 2026, Redistricting Tracker, or Epstein Files.
It also accepts an article whose headline names a U.S. president, even when its tag is
outside that list (for example, a Health article with "Trump" in its headline).

```powershell
python -m pip install -r requirements.txt
python cnn_image_scraper.py "https://edition.cnn.com" images --max-articles 100
```

Clean an existing output folder without crawling again:

```powershell
python cnn_image_scraper.py "https://edition.cnn.com" downloaded_images --deduplicate-only
```

Only the start URL is required. The output folder is optional and defaults to
`downloaded_images`; the default crawl limit is 100 article pages.

```powershell
python -m unittest -v
```
