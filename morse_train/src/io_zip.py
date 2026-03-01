import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class SignalFrame:
    frame: pd.DataFrame
    source_format: str
    zip_path: str


def _find_first_entry(entries: List[zipfile.ZipInfo], suffix: str) -> Optional[zipfile.ZipInfo]:
    matches = [e for e in entries if not e.is_dir() and e.filename.lower().endswith(suffix)]
    if not matches:
        return None
    matches.sort(key=lambda x: x.filename)
    return matches[0]


def _read_csv_blob(blob: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = blob.decode(encoding)
            return pd.read_csv(io.StringIO(text), header=None, na_values=["NA", ""])
        except UnicodeDecodeError:
            continue
    raise ValueError("Failed to decode CSV bytes using utf-8-sig/gb18030/utf-16")


def _read_xlsx_blob(blob: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(blob), header=None)


def _norm_header(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _parse_dual_header(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.shape[0] < 3:
        raise ValueError("Raw table has too few rows")

    header_row_1 = raw_df.iloc[0]
    header_row_2 = raw_df.iloc[1]
    data_df = raw_df.iloc[2:].copy()

    time_col = None
    delta_cols = {}

    def infer_channel(idx: int) -> str:
        direct = _norm_header(header_row_1.iloc[idx])
        if direct.startswith("ch"):
            return direct.upper()
        if idx > 0:
            left = _norm_header(header_row_1.iloc[idx - 1])
            if left.startswith("ch"):
                return left.upper()
        return ""

    for idx in range(raw_df.shape[1]):
        h2 = _norm_header(header_row_2.iloc[idx])
        if "time" in h2 and "(s)" in h2:
            time_col = idx
            continue

        if "r/r0" in h2 and "%" in h2:
            ch = infer_channel(idx)
            if ch in ("CH1", "CH2"):
                delta_cols[ch] = idx

    if time_col is None:
        raise ValueError("Failed to locate Time (s) column")
    if "CH1" not in delta_cols or "CH2" not in delta_cols:
        raise ValueError("Failed to locate CH1/CH2 delta columns")

    out = pd.DataFrame(
        {
            "time_s": pd.to_numeric(data_df.iloc[:, time_col], errors="coerce"),
            "ch1": pd.to_numeric(data_df.iloc[:, delta_cols["CH1"]], errors="coerce"),
            "ch2": pd.to_numeric(data_df.iloc[:, delta_cols["CH2"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if out.empty:
        raise ValueError("Parsed frame is empty")
    return out


def load_signal_from_zip(zip_path) -> SignalFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = list(zf.infolist())
        csv_entry = _find_first_entry(entries, ".csv")
        xlsx_entry = _find_first_entry(entries, ".xlsx")

        if csv_entry is not None:
            blob = zf.read(csv_entry)
            raw = _read_csv_blob(blob)
            parsed = _parse_dual_header(raw)
            return SignalFrame(frame=parsed, source_format="csv", zip_path=str(zip_path))

        if xlsx_entry is not None:
            blob = zf.read(xlsx_entry)
            raw = _read_xlsx_blob(blob)
            parsed = _parse_dual_header(raw)
            return SignalFrame(frame=parsed, source_format="xlsx", zip_path=str(zip_path))

    raise ValueError(f"No CSV/XLSX found in zip: {zip_path}")

