from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def eur(value: float) -> str:
    return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def kwh(value: float) -> str:
    return f"{value:,.1f} kWh".replace(",", "X").replace(".", ",").replace("X", ".")


def kw(value: float) -> str:
    return f"{value:,.1f} kW".replace(",", "X").replace(".", ",").replace("X", ".")


def co2(value_kg: float) -> str:
    return f"{value_kg:,.1f} kg CO2".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def summary_box_text(assumptions: Dict[str, float], missing: Iterable[str]) -> str:
    lines = ["Wichtige Annahmen:"]
    for key, value in assumptions.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4g}")
        else:
            lines.append(f"- {key}: {value}")

    missing = list(missing)
    if missing:
        lines.append("\nFehlende/optionale Annahmen:")
        for m in missing:
            lines.append(f"- {m}")
    return "\n".join(lines)


def format_dataframe_money(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 2))
    return out
