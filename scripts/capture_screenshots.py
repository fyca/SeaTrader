from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("dashboard", "http://127.0.0.1:8008/"),
    ("builder", "http://127.0.0.1:8008/builder"),
]


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        for name, url in PAGES:
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)

        browser.close()


if __name__ == "__main__":
    main()
