from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_excel_cached(path: Path, sheet_name: str, usecols: Iterable[str]) -> pd.DataFrame:
    resolved = Path(path).resolve()
    columns = tuple(str(value) for value in usecols)
    return _read_excel(str(resolved), int(resolved.stat().st_mtime_ns), resolved.stat().st_size, str(sheet_name), columns).copy(deep=True)


@lru_cache(maxsize=24)
def _read_excel(path: str, mtime_ns: int, size: int, sheet_name: str, usecols: tuple[str, ...]) -> pd.DataFrame:
    del mtime_ns, size
    return pd.read_excel(path, sheet_name=sheet_name, usecols=list(usecols))
