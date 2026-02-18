from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from model import InputParameters, compute_cashflows


def _clone_with(params: InputParameters, **kwargs) -> InputParameters:
    return replace(params, **kwargs)


def run_scenarios(params: InputParameters) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run Base/Best/Worst scenario set."""
    scenario_map = {
        "Base": params,
        "Best": _clone_with(
            params,
            diesel_price_eur_per_l=params.diesel_price_eur_per_l * 1.15,
            electricity_spot_mean_eur_per_mwh=params.electricity_spot_mean_eur_per_mwh * 0.85,
            electricity_spot_stdev_eur_per_mwh=params.electricity_spot_stdev_eur_per_mwh * 0.8,
            operating_days_per_year=min(365, int(params.operating_days_per_year * 1.05)),
            diesel_min_load_factor=max(0.2, params.diesel_min_load_factor * 0.9),
            battery_capacity_fade_per_year=max(0.0, params.battery_capacity_fade_per_year * 0.8),
        ),
        "Worst": _clone_with(
            params,
            diesel_price_eur_per_l=params.diesel_price_eur_per_l * 0.9,
            electricity_spot_mean_eur_per_mwh=params.electricity_spot_mean_eur_per_mwh * 1.25,
            electricity_spot_stdev_eur_per_mwh=params.electricity_spot_stdev_eur_per_mwh * 1.25,
            operating_days_per_year=max(0, int(params.operating_days_per_year * 0.9)),
            diesel_min_load_factor=min(0.6, params.diesel_min_load_factor * 1.1),
            battery_capacity_fade_per_year=min(0.2, params.battery_capacity_fade_per_year * 1.2),
        ),
    }

    rows = []
    details: Dict[str, object] = {}
    for name, s_params in scenario_map.items():
        result = compute_cashflows(s_params)
        details[name] = result
        rows.append(
            {
                "scenario": name,
                "daily_savings_eur": result.daily.savings_day,
                "npv_eur": result.npv_eur,
                "irr": result.irr,
                "payback_years": result.payback_years,
                "payback_days": result.payback_days,
                "battery_units": result.daily.battery_units_used,
            }
        )
    return pd.DataFrame(rows), details


def _set_param_value(params: InputParameters, parameter: str, value: float) -> InputParameters:
    if not hasattr(params, parameter):
        raise ValueError(f"Unbekannter Parameter: {parameter}")
    return replace(params, **{parameter: value})


def sensitivity_1d(
    params: InputParameters,
    parameter: str,
    value_min: float,
    value_max: float,
    n_steps: int = 15,
) -> pd.DataFrame:
    """1D sensitivity impact on NPV and Payback."""
    values = np.linspace(value_min, value_max, n_steps)
    rows: List[Dict[str, float]] = []
    for value in values:
        p = _set_param_value(params, parameter, float(value))
        r = compute_cashflows(p)
        rows.append(
            {
                "parameter": parameter,
                "value": float(value),
                "npv_eur": r.npv_eur,
                "payback_years": np.nan if r.payback_years is None else r.payback_years,
            }
        )
    return pd.DataFrame(rows)


def tornado(
    params: InputParameters,
    parameters: List[str],
    variation_pct: float = 0.2,
    top_n: int = 5,
) -> pd.DataFrame:
    """Tornado input-output impact analysis (+/- variation on NPV)."""
    base_npv = compute_cashflows(params).npv_eur
    rows: List[Dict[str, float]] = []

    for p_name in parameters:
        if not hasattr(params, p_name):
            continue
        base_value = getattr(params, p_name)
        if not isinstance(base_value, (int, float)):
            continue

        low_val = base_value * (1.0 - variation_pct)
        high_val = base_value * (1.0 + variation_pct)
        low_params = _set_param_value(params, p_name, float(low_val))
        high_params = _set_param_value(params, p_name, float(high_val))

        low_npv = compute_cashflows(low_params).npv_eur
        high_npv = compute_cashflows(high_params).npv_eur

        rows.append(
            {
                "parameter": p_name,
                "low_npv": low_npv,
                "high_npv": high_npv,
                "base_npv": base_npv,
                "impact_abs": max(abs(low_npv - base_npv), abs(high_npv - base_npv)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("impact_abs", ascending=False).head(top_n)


def monte_carlo_npv(
    params: InputParameters,
    runs: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Optional Monte-Carlo simulation for NPV using normal spot/diesel uncertainty."""
    rng = np.random.default_rng(seed)
    spot_samples = rng.normal(
        loc=params.electricity_spot_mean_eur_per_mwh,
        scale=max(1e-9, params.electricity_spot_stdev_eur_per_mwh),
        size=runs,
    )
    spot_samples = np.clip(spot_samples, 0.0, None)

    diesel_sigma = max(0.01, params.diesel_price_eur_per_l * 0.12)
    diesel_samples = np.clip(rng.normal(params.diesel_price_eur_per_l, diesel_sigma, size=runs), 0.0, None)

    out = np.empty(runs)
    for i in range(runs):
        sim_params = replace(
            params,
            electricity_spot_mean_eur_per_mwh=float(spot_samples[i]),
            diesel_price_eur_per_l=float(diesel_samples[i]),
        )
        out[i] = compute_cashflows(sim_params).npv_eur

    return pd.DataFrame({"run": np.arange(1, runs + 1), "npv_eur": out})
