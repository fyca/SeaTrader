from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)

THEMES = ["classic", "fun", "dark"]
DENSITIES = ["comfortable", "compact"]

PAGES = [
    ("dashboard", "http://127.0.0.1:8008/"),
    ("builder", "http://127.0.0.1:8008/builder"),
]

HUB_PAGES = [
    ("hub", "http://127.0.0.1:8099/", 90000),
]


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        for theme in THEMES:
            for dens in DENSITIES:
                for name, url in PAGES:
                    page.goto(url, wait_until="domcontentloaded")
                    # set theme/density after we are on same-origin
                    page.evaluate(
                        """([theme, dens]) => {
                          try { localStorage.setItem('ui_theme', theme); } catch(e) {}
                          try { localStorage.setItem('ui_density', dens); } catch(e) {}
                        }""",
                        [theme, dens],
                    )
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_timeout(1400)
                    page.screenshot(
                        path=str(OUT / f"{name}-{theme}-{dens}.png"),
                        full_page=True,
                    )

        # Hub screenshots (single style)
        for name, url, wait_ms in HUB_PAGES:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)
            page.screenshot(
                path=str(OUT / f"{name}.png"),
                full_page=True,
            )

        browser.close()


if __name__ == "__main__":
    main()
