"""Reusable image download, comparison, and deduplication helpers."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

import cv2
import numpy
import requests
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
MINIMUM_FACE_COUNT = 2
_FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


def face_count(image_bytes: bytes) -> int:
    """Return the number of frontal human faces detected in image bytes."""
    encoded = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or _FACE_CASCADE.empty():
        return 0
    faces = _FACE_CASCADE.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces)


def remove_images_without_minimum_faces(
    paths: Iterable[Path], minimum_faces: int = MINIMUM_FACE_COUNT,
    face_counter: Callable[[bytes], int] | None = None,
) -> tuple[int, int]:
    """Delete unreadable images and images with fewer than ``minimum_faces`` faces."""
    counter = face_counter or face_count
    retained = removed = 0
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            has_enough_faces = counter(path.read_bytes()) >= minimum_faces
        except (OSError, ValueError, cv2.error):
            has_enough_faces = False
        if has_enough_faces:
            retained += 1
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as error:
            print(f"Could not remove face-filtered image {path.name}: {error}")
    return retained, removed


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
    """Create a small, size-independent pixel representation for comparison."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            normalized = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
            return VisualImage(path, width, height, normalized.tobytes())
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def visually_same(first: VisualImage, second: VisualImage) -> bool:
    """Conservatively identify resize/re-encoding variants of one picture."""
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
        if image := visual_image(path, body):
            visual_images.append(image)
    return content_hashes, visual_images


def download_image_urls(
    image_urls: Iterable[str], output_dir: Path, session: requests.Session, content_hashes: set[str], visual_images: list[VisualImage],
    minimum_faces: int = MINIMUM_FACE_COUNT, face_counter: Callable[[bytes], int] | None = None,
) -> list[Path]:
    """Download images with enough faces, retaining the largest visual variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = face_counter or face_count
    saved: list[Path] = []
    for image_url in image_urls:
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
        try:
            detected_faces = counter(body)
        except (ValueError, cv2.error):
            detected_faces = 0
        if detected_faces < minimum_faces:
            print(f"Skipping {image_url}: found {detected_faces} face(s); need at least {minimum_faces}.")
            continue
        destination = output_dir / f"{len(content_hashes):05d}-{digest[:12]}{extension_for(image_response, image_url)}"
        candidate = visual_image(destination, body)
        equivalents = [image for image in visual_images if candidate and visually_same(image, candidate)]
        if equivalents and candidate is not None and candidate.area <= max(image.area for image in equivalents):
            continue
        try:
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
