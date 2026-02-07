from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def get_sp500_symbols() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia.

    Free, but not mission-critical. If it ever breaks, replace with a static snapshot.
    """
    # Prefer a simple CSV snapshot (more reliable than scraping Wikipedia).
    csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        df = pd.read_csv(csv_url)
        syms = df["Symbol"].astype(str).tolist()
        return sorted(set(syms))
    except Exception:
        # Fallback: Wikipedia scrape
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        syms = df["Symbol"].astype(str).tolist()
        return sorted(set(syms))
