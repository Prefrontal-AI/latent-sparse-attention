"""arXiv's HTML of a paper at a pinned version, fetched politely once and kept.

One request at a time, a pause after each, and a back-off that doubles on every failed attempt. A page is kept
gzipped as `<id>v<version>.html.gz`, so a rebuild run again reads its cache and asks arXiv for nothing.
"""

import gzip
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "PaperTrail-build-dataset/1.0 (benchmark rebuild; polite single-threaded fetch)"
ATTEMPTS = 6
TIMEOUT_SECONDS = 120


def page_path(cache: Path, arxiv_id: str, version: int) -> Path:
    """Return where one paper's page at one version is kept."""
    return cache / f"{arxiv_id}v{version}.html.gz"


def fetch_page(arxiv_id: str, version: int, cache: Path, delay: float) -> str:
    """Return one paper's HTML at one version, from the cache or from arXiv.

    Args:
        arxiv_id: The paper.
        version: Its pinned version.
        cache: Where pages are kept.
        delay: Seconds to wait after a request, doubled after each failed attempt.

    Returns:
        The page.

    Raises:
        FileNotFoundError: arXiv has no HTML of that version - the paper was withdrawn or its HTML removed.
        ConnectionError: Every attempt failed.
    """
    path = page_path(cache, arxiv_id, version)
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf8", "ignore")
    url = f"https://arxiv.org/html/{arxiv_id}v{version}"
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                raw = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            # A stopped job leaves no half-written page behind
            scratch = path.with_name(f".{path.name}.partial")
            scratch.write_bytes(gzip.compress(raw))
            scratch.replace(path)
            time.sleep(delay)
            return raw.decode("utf8", "ignore")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(
                    f"{url} not found - the paper was withdrawn or its HTML removed"
                ) from error
        except (urllib.error.URLError, TimeoutError):
            pass
        wait = delay * 2**attempt
        print(f"  retry {url} in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise ConnectionError(f"could not fetch {url}")
