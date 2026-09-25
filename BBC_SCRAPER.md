# BBC article image scraper

`bbc_image_scraper.py` crawls BBC pages breadth-first and downloads qualifying
images from article pages. It recognizes the current BBC article URL shape by
the `/articles/<id>` path segment, including language sections such as
`/mundo/articles/<id>`.

```powershell
python -m pip install -r requirements.txt
python bbc_image_scraper.py "https://www.bbc.com/news" BBC_images --max-articles 100
```

Start from a specific article, for example:

```powershell
python bbc_image_scraper.py "https://www.bbc.com/news/articles/c98r62j4jl5mo" BBC_images
```

Use `--min-images 500` to continue until that many qualifying images have been
saved, or until no newly discovered BBC articles remain.
