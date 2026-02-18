from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd


DAYS_PER_MONTH = 30.0


@dataclass
class InputParameters:
    """Input container for TCO model assumptions."""

    profile_mode: str = "constant"  # constant, peak, csv
    constant_load_kw: float = 10.0
    operating_hours_per_day: float = 10.0
    base_load_kw: float = 6.0
    peak_load_kw: float = 18.0
    peak_hours: float = 4.0
    csv_profile_values: Optional[List[float]] = None
    direct_daily_energy_kwh: Optional[float] = 100.0

    operating_days_per_year: int = 220
    horizon_years: int = 5
    discount_rate: float = 0.08

    auto_size_units: bool = True
    comparison_basis: str = "incremental"  # incremental, replacement

    # Diesel
    diesel_price_eur_per_l: float = 1.75
    diesel_rated_power_kw: float = 40.0
    diesel_units: int = 1
    diesel_kwh_per_l_rated: float = 3.0
    diesel_min_load_factor: float = 0.3
    diesel_idle_fuel_l_per_h: float = 1.2
    diesel_low_load_penalty_l_per_h: float = 0.3
    diesel_maintenance_eur_per_day: float = 4.5
    diesel_maintenance_eur_per_hour: float = 0.0
    diesel_logistics_eur_per_l: float = 0.05
    diesel_logistics_eur_per_day: float = 0.0
    diesel_theft_shrinkage_rate: float = 0.02
    diesel_downtime_cost_eur_per_day: float = 0.0
    diesel_capex_eur: float = 50000.0
    diesel_residual_value_pct: float = 0.0

    # Carbon
    carbon_enabled: bool = False
    carbon_price_eur_per_tco2: float = 0.0
    diesel_emission_factor_kgco2_per_l: float = 2.68

    # Battery
    battery_units: int = 1
    battery_nominal_capacity_kwh: float = 100.0
    battery_usable_dod: float = 0.8
    battery_roundtrip_efficiency: float = 0.88
    battery_max_discharge_power_kw: float = 30.0
    battery_max_charge_power_kw: float = 25.0
    battery_charge_hours_per_day: float = 12.0
    battery_capacity_fade_per_year: float = 0.03
    battery_degradation_per_cycle: float = 0.0
    battery_cycle_life_to_eol: Optional[float] = 3000.0
    battery_replacement_year: Optional[int] = None
    battery_eol_capacity_fraction: float = 0.8

    battery_capex_unit_eur: float = 12500.0
    charger_capex_eur: float = 2000.0
    connection_capex_eur: float = 1500.0
    battery_insurance_theft_eur_per_day: float = 0.3
    battery_maintenance_eur_per_day: float = 0.5
    battery_end_of_life_cost_eur: float = 0.0
    battery_residual_value_pct: float = 0.0

    # Electricity pricing
    electricity_price_mode: str = "spot"  # fixed, spot
    electricity_fixed_eur_per_kwh: float = 0.15
    electricity_spot_mean_eur_per_mwh: float = 99.88
    electricity_spot_stdev_eur_per_mwh: float = 21.09
    spot_strategy: str = "optimized"  # optimized, average, worst
    spot_alpha: float = 1.0
    energy_fee_eur_per_kwh: float = 0.02
    fixed_fee_eur_per_day: float = 0.0
    fixed_fee_eur_per_month: float = 30.0
    demand_charge_eur_per_kw_month: float = 0.0


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]


@dataclass
class DailyResult:
    daily_energy_kwh: float
    peak_kw: float
    avg_kw: float

    diesel_fuel_l: float
    diesel_cost_breakdown: Dict[str, float]
    battery_cost_breakdown: Dict[str, float]

    diesel_opex_day: float
    battery_opex_day: float
    savings_day: float

    diesel_co2_kg_day: float
    battery_co2_kg_day: float
    co2_savings_kg_day: float

    battery_units_used: int
    diesel_units_used: int
    battery_feasible: bool
    battery_feasibility_notes: List[str]


@dataclass
class CashflowResult:
    daily: DailyResult
    yearly_table: pd.DataFrame
    cashflows: List[float]
    cumulative_cashflows: List[float]
    payback_years: Optional[float]
    payback_days: Optional[float]
    npv_eur: float
    irr: Optional[float]
    lcoe_like_eur_per_kwh: Optional[float]


def _interp_piecewise(x: float, points: List[Tuple[float, float]]) -> float:
    pts = sorted(points, key=lambda p: p[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def kwh_per_l_at_load(load_factor: float, kwh_per_l_rated: float) -> float:
    """Approximate genset efficiency curve (kWh per liter) as function of load factor."""
    curve = [
        (0.1, 0.42 * kwh_per_l_rated),
        (0.25, 0.60 * kwh_per_l_rated),
        (0.50, 0.80 * kwh_per_l_rated),
        (0.75, 0.92 * kwh_per_l_rated),
        (1.00, 1.00 * kwh_per_l_rated),
    ]
    return max(0.01, _interp_piecewise(load_factor, curve))


def _build_hourly_profile(params: InputParameters) -> np.ndarray:
    profile = np.zeros(24, dtype=float)
    if params.profile_mode == "constant":
        hours = max(0.0, min(24.0, params.operating_hours_per_day))
        active_hours = int(math.floor(hours))
        frac = hours - active_hours
        if active_hours > 0:
            profile[:active_hours] = params.constant_load_kw
        if frac > 0 and active_hours < 24:
            profile[active_hours] = params.constant_load_kw * frac
    elif params.profile_mode == "peak":
        hours = max(0.0, min(24.0, params.operating_hours_per_day))
        peak_h = max(0.0, min(hours, params.peak_hours))
        base_h = hours - peak_h
        if base_h > 0:
            profile[: int(base_h)] = params.base_load_kw
            rem = base_h - int(base_h)
            if rem > 0 and int(base_h) < 24:
                profile[int(base_h)] = params.base_load_kw * rem
        start_peak = int(math.ceil(base_h))
        if peak_h > 0:
            full_peak = int(peak_h)
            end = min(24, start_peak + full_peak)
            profile[start_peak:end] = params.peak_load_kw
            rem_peak = peak_h - full_peak
            if rem_peak > 0 and end < 24:
                profile[end] = params.peak_load_kw * rem_peak
    elif params.profile_mode == "csv" and params.csv_profile_values:
        values = np.array(params.csv_profile_values[:24], dtype=float)
        if len(values) < 24:
            values = np.concatenate([values, np.zeros(24 - len(values))])
        profile = np.clip(values, a_min=0.0, a_max=None)
    else:
        profile[: int(params.operating_hours_per_day)] = params.constant_load_kw

    if params.direct_daily_energy_kwh is not None and params.direct_daily_energy_kwh > 0:
        energy = float(profile.sum())
        if energy > 0:
            profile *= params.direct_daily_energy_kwh / energy

    return np.clip(profile, a_min=0.0, a_max=None)


def validate_inputs(params: InputParameters) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    efficiencies = [params.battery_roundtrip_efficiency, params.battery_usable_dod]
    if any(e <= 0 or e > 1 for e in efficiencies):
        errors.append("Batterieeffizienzen und DoD müssen im Bereich (0,1] liegen.")

    if not 0 <= params.operating_days_per_year <= 365:
        errors.append("Betriebstage/Jahr müssen zwischen 0 und 365 liegen.")

    positive_fields = {
        "Generatorleistung": params.diesel_rated_power_kw,
        "Generatoreffizienz": params.diesel_kwh_per_l_rated,
        "Batteriekapazität": params.battery_nominal_capacity_kwh,
        "Batterie-Entladeleistung": params.battery_max_discharge_power_kw,
        "Batterie-Ladeleistung": params.battery_max_charge_power_kw,
        "Ladezeit": params.battery_charge_hours_per_day,
        "Horizont": params.horizon_years,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            errors.append(f"{name} muss > 0 sein.")

    non_negative = {
        "Dieselpreis": params.diesel_price_eur_per_l,
        "Strompreis mean": params.electricity_spot_mean_eur_per_mwh,
        "Strompreis stdev": params.electricity_spot_stdev_eur_per_mwh,
        "WACC": params.discount_rate,
        "Wartung Diesel": params.diesel_maintenance_eur_per_day,
        "Wartung Batterie": params.battery_maintenance_eur_per_day,
    }
    for name, value in non_negative.items():
        if value < 0:
            errors.append(f"{name} darf nicht negativ sein.")

    if params.discount_rate > 1:
        warnings.append("Diskontsatz > 100% ist ungewöhnlich.")

    return ValidationResult(errors=errors, warnings=warnings)


def _spot_component(params: InputParameters) -> float:
    if params.electricity_price_mode == "fixed":
        return max(0.0, params.electricity_fixed_eur_per_kwh)

    mean = params.electricity_spot_mean_eur_per_mwh
    stdev = params.electricity_spot_stdev_eur_per_mwh
    alpha = params.spot_alpha

    if params.spot_strategy == "optimized":
        value_mwh = mean - alpha * stdev
    elif params.spot_strategy == "worst":
        value_mwh = mean + alpha * stdev
    else:
        value_mwh = mean

    return max(0.0, value_mwh / 1000.0)


def _electricity_total_cost_per_day(charge_energy_kwh: float, charge_power_kw: float, params: InputParameters) -> Tuple[float, Dict[str, float]]:
    spot_eur_per_kwh = _spot_component(params)
    variable_eur_per_kwh = spot_eur_per_kwh + params.energy_fee_eur_per_kwh
    variable_cost = charge_energy_kwh * variable_eur_per_kwh

    fixed_daily = params.fixed_fee_eur_per_day + (params.fixed_fee_eur_per_month / DAYS_PER_MONTH)
    demand_daily = params.demand_charge_eur_per_kw_month * charge_power_kw / DAYS_PER_MONTH

    total = variable_cost + fixed_daily + demand_daily
    breakdown = {
        "grid_energy": variable_cost,
        "grid_fixed": fixed_daily,
        "grid_demand": demand_daily,
    }
    return total, breakdown


def _autosize_diesel_units(peak_kw: float, params: InputParameters) -> int:
    return max(1, int(math.ceil(peak_kw / max(1e-9, params.diesel_rated_power_kw))))


def _autosize_battery_units(peak_kw: float, daily_energy_kwh: float, year_capacity_factor: float, params: InputParameters) -> int:
    usable_capacity_unit = params.battery_nominal_capacity_kwh * params.battery_usable_dod * year_capacity_factor
    power_units = math.ceil(peak_kw / max(1e-9, params.battery_max_discharge_power_kw))
    energy_units = math.ceil(daily_energy_kwh / max(1e-9, usable_capacity_unit))
    charge_energy_kwh = daily_energy_kwh / max(1e-9, params.battery_roundtrip_efficiency)
    charge_power_needed = charge_energy_kwh / max(1e-9, params.battery_charge_hours_per_day)
    charge_units = math.ceil(charge_power_needed / max(1e-9, params.battery_max_charge_power_kw))
    return max(1, int(max(power_units, energy_units, charge_units)))


def _battery_capacity_factor_for_year(params: InputParameters, year: int) -> float:
    cycles_per_year = params.operating_days_per_year
    cycle_fade = params.battery_degradation_per_cycle * cycles_per_year * (year - 1)
    yearly_fade = params.battery_capacity_fade_per_year * (year - 1)
    total_fade = cycle_fade + yearly_fade
    return max(params.battery_eol_capacity_fraction, 1.0 - total_fade)


def _degradation_cost_per_kwh(params: InputParameters, units: int) -> float:
    if params.battery_cycle_life_to_eol and params.battery_cycle_life_to_eol > 0:
        throughput = (
            params.battery_nominal_capacity_kwh
            * params.battery_usable_dod
            * params.battery_cycle_life_to_eol
            * units
        )
        if throughput > 0:
            return (params.battery_capex_unit_eur * units) / throughput
    return 0.0


def compute_daily(profile: np.ndarray, params: InputParameters, year: int = 1) -> DailyResult:
    """Compute daily costs and feasibility for diesel vs battery."""
    loads = np.array(profile, dtype=float)
    daily_energy_kwh = float(loads.sum())
    peak_kw = float(loads.max()) if len(loads) else 0.0
    avg_kw = daily_energy_kwh / 24.0

    diesel_units = params.diesel_units
    if params.auto_size_units:
        diesel_units = max(diesel_units, _autosize_diesel_units(peak_kw, params))

    total_diesel_rated_kw = diesel_units * params.diesel_rated_power_kw

    fuel_l = 0.0
    running_hours = 0.0
    for load_kw in loads:
        if load_kw <= 0:
            continue
        running_hours += 1.0
        load_factor = load_kw / max(1e-9, total_diesel_rated_kw)
        min_load_kw = params.diesel_min_load_factor * total_diesel_rated_kw
        if load_kw < min_load_kw:
            penalty_factor = 1.0 - (load_kw / max(1e-9, min_load_kw))
            fuel_l += params.diesel_idle_fuel_l_per_h + params.diesel_low_load_penalty_l_per_h * penalty_factor
        else:
            eff = kwh_per_l_at_load(load_factor, params.diesel_kwh_per_l_rated)
            fuel_l += load_kw / max(1e-9, eff)

    theft_l = fuel_l * params.diesel_theft_shrinkage_rate
    billed_fuel_l = fuel_l + theft_l
    fuel_cost = billed_fuel_l * params.diesel_price_eur_per_l
    maintenance = params.diesel_maintenance_eur_per_day + running_hours * params.diesel_maintenance_eur_per_hour
    logistics = billed_fuel_l * params.diesel_logistics_eur_per_l + params.diesel_logistics_eur_per_day
    carbon_cost = 0.0
    diesel_co2_kg = billed_fuel_l * params.diesel_emission_factor_kgco2_per_l
    if params.carbon_enabled:
        carbon_cost = (diesel_co2_kg / 1000.0) * params.carbon_price_eur_per_tco2

    diesel_opex_day = fuel_cost + maintenance + logistics + params.diesel_downtime_cost_eur_per_day + carbon_cost

    # Battery side
    year_capacity_factor = _battery_capacity_factor_for_year(params, year)
    battery_units = params.battery_units
    if params.auto_size_units:
        battery_units = max(
            battery_units,
            _autosize_battery_units(peak_kw, daily_energy_kwh, year_capacity_factor, params),
        )

    feasible = True
    notes: List[str] = []

    max_discharge_kw_total = battery_units * params.battery_max_discharge_power_kw
    if peak_kw > max_discharge_kw_total + 1e-9:
        feasible = False
        notes.append("Peak-Leistung kann nicht vollständig durch Batterie gedeckt werden.")

    usable_capacity_total = (
        battery_units
        * params.battery_nominal_capacity_kwh
        * params.battery_usable_dod
        * year_capacity_factor
    )
    if daily_energy_kwh > usable_capacity_total + 1e-9:
        feasible = False
        notes.append("Tagesenergie übersteigt nutzbare Batteriekapazität.")

    charge_energy_kwh = daily_energy_kwh / max(1e-9, params.battery_roundtrip_efficiency)
    max_charge_kwh = battery_units * params.battery_max_charge_power_kw * params.battery_charge_hours_per_day
    charge_power_needed_kw = charge_energy_kwh / max(1e-9, params.battery_charge_hours_per_day)
    if charge_energy_kwh > max_charge_kwh + 1e-9:
        feasible = False
        notes.append("Ladefenster zu klein: charge_power * hours < required_charge_energy.")

    electricity_total, electricity_breakdown = _electricity_total_cost_per_day(charge_energy_kwh, charge_power_needed_kw, params)
    degr_cost_per_kwh = _degradation_cost_per_kwh(params, battery_units)
    degradation_cost = degr_cost_per_kwh * daily_energy_kwh

    battery_maintenance = params.battery_maintenance_eur_per_day
    insurance = params.battery_insurance_theft_eur_per_day
    battery_opex_day = (
        electricity_total
        + battery_maintenance
        + insurance
        + degradation_cost
        + params.battery_end_of_life_cost_eur / max(1, params.operating_days_per_year)
    )

    battery_co2_kg = 0.0
    co2_savings = diesel_co2_kg - battery_co2_kg

    diesel_breakdown = {
        "fuel": fuel_cost,
        "maintenance": maintenance,
        "logistics": logistics,
        "downtime": params.diesel_downtime_cost_eur_per_day,
        "carbon": carbon_cost,
        "theft_loss": theft_l * params.diesel_price_eur_per_l,
    }
    battery_breakdown = {
        **electricity_breakdown,
        "maintenance": battery_maintenance,
        "insurance_theft": insurance,
        "degradation": degradation_cost,
    }

    return DailyResult(
        daily_energy_kwh=daily_energy_kwh,
        peak_kw=peak_kw,
        avg_kw=avg_kw,
        diesel_fuel_l=billed_fuel_l,
        diesel_cost_breakdown=diesel_breakdown,
        battery_cost_breakdown=battery_breakdown,
        diesel_opex_day=diesel_opex_day,
        battery_opex_day=battery_opex_day,
        savings_day=diesel_opex_day - battery_opex_day,
        diesel_co2_kg_day=diesel_co2_kg,
        battery_co2_kg_day=battery_co2_kg,
        co2_savings_kg_day=co2_savings,
        battery_units_used=battery_units,
        diesel_units_used=diesel_units,
        battery_feasible=feasible,
        battery_feasibility_notes=notes,
    )


def compute_yearly(params: InputParameters, horizon_years: Optional[int] = None) -> pd.DataFrame:
    """Compute yearly OPEX projection with degradation and optional unit re-sizing."""
    if horizon_years is None:
        horizon_years = params.horizon_years

    profile = _build_hourly_profile(params)
    rows: List[Dict[str, float]] = []
    for year in range(1, horizon_years + 1):
        daily = compute_daily(profile, params, year=year)
        days = params.operating_days_per_year
        rows.append(
            {
                "year": year,
                "daily_energy_kwh": daily.daily_energy_kwh,
                "battery_units": daily.battery_units_used,
                "diesel_units": daily.diesel_units_used,
                "diesel_opex_year": daily.diesel_opex_day * days,
                "battery_opex_year": daily.battery_opex_day * days,
                "savings_year": daily.savings_day * days,
                "diesel_co2_t_year": daily.diesel_co2_kg_day * days / 1000.0,
                "battery_co2_t_year": daily.battery_co2_kg_day * days / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _battery_total_capex(params: InputParameters, units: int) -> float:
    return units * params.battery_capex_unit_eur + params.charger_capex_eur + params.connection_capex_eur


def compute_cashflows(params: InputParameters) -> CashflowResult:
    """Build full cashflow table and key finance KPIs."""
    profile = _build_hourly_profile(params)
    daily_year1 = compute_daily(profile, params, year=1)
    yearly = compute_yearly(params, horizon_years=params.horizon_years)

    battery_capex = _battery_total_capex(params, daily_year1.battery_units_used)
    diesel_capex = params.diesel_capex_eur * daily_year1.diesel_units_used

    if params.comparison_basis == "replacement":
        initial_invest = battery_capex - diesel_capex
    else:
        initial_invest = battery_capex

    cashflows = [-initial_invest]

    replacement_year = params.battery_replacement_year
    if replacement_year is None and params.battery_cycle_life_to_eol and params.battery_cycle_life_to_eol > 0:
        cycles_per_year = max(1, params.operating_days_per_year)
        est_life_years = params.battery_cycle_life_to_eol / cycles_per_year
        if est_life_years < params.horizon_years:
            replacement_year = max(1, int(math.ceil(est_life_years)))

    for year in range(1, params.horizon_years + 1):
        savings = float(yearly.loc[yearly["year"] == year, "savings_year"].iloc[0])
        cf = savings
        if replacement_year is not None and year == replacement_year:
            cf -= daily_year1.battery_units_used * params.battery_capex_unit_eur
        if year == params.horizon_years:
            batt_residual = battery_capex * params.battery_residual_value_pct
            diesel_residual = diesel_capex * params.diesel_residual_value_pct
            if params.comparison_basis == "replacement":
                cf += batt_residual - diesel_residual
            else:
                cf += batt_residual
        cashflows.append(cf)

    npv = compute_npv(cashflows, params.discount_rate)
    irr = compute_irr(cashflows)
    payback_years, payback_days = compute_payback(cashflows, params.operating_days_per_year)

    discounted_energy = 0.0
    discounted_cost = 0.0
    for year in range(1, params.horizon_years + 1):
        disc = 1 / ((1 + params.discount_rate) ** year)
        discounted_energy += daily_year1.daily_energy_kwh * params.operating_days_per_year * disc
        discounted_cost += float(yearly.loc[yearly["year"] == year, "battery_opex_year"].iloc[0]) * disc
    discounted_cost += max(0.0, initial_invest)
    lcoe_like = discounted_cost / discounted_energy if discounted_energy > 0 else None

    cumulative = []
    run = 0.0
    for cf in cashflows:
        run += cf
        cumulative.append(run)

    cashflow_table = pd.DataFrame(
        {
            "year": list(range(0, params.horizon_years + 1)),
            "cashflow_eur": cashflows,
            "cumulative_cashflow_eur": cumulative,
        }
    )

    return CashflowResult(
        daily=daily_year1,
        yearly_table=yearly,
        cashflows=cashflows,
        cumulative_cashflows=cumulative,
        payback_years=payback_years,
        payback_days=payback_days,
        npv_eur=npv,
        irr=irr,
        lcoe_like_eur_per_kwh=lcoe_like,
    )


def compute_payback(cashflows: List[float], operating_days_per_year: int) -> Tuple[Optional[float], Optional[float]]:
    """Simple payback with interpolation in-year."""
    cumulative = 0.0
    prev_cumulative = 0.0
    for year, cf in enumerate(cashflows):
        cumulative += cf
        if year == 0:
            prev_cumulative = cumulative
            continue
        if cumulative >= 0:
            if cf == 0:
                pb_years = float(year)
            else:
                fraction = (-prev_cumulative) / cf
                pb_years = (year - 1) + max(0.0, min(1.0, fraction))
            pb_days = pb_years * operating_days_per_year
            return pb_years, pb_days
        prev_cumulative = cumulative
    return None, None


def compute_npv(cashflows: List[float], discount: float) -> float:
    """Net present value for annual cashflows."""
    return float(sum(cf / ((1 + discount) ** t) for t, cf in enumerate(cashflows)))


def compute_irr(cashflows: List[float], tol: float = 1e-7, max_iter: int = 200) -> Optional[float]:
    """Robust IRR via bisection, returns None if no sign change."""

    def npv_at(rate: float) -> float:
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

    low, high = -0.999, 10.0
    f_low, f_high = npv_at(low), npv_at(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv_at(mid)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def assumptions_summary(params: InputParameters) -> Dict[str, float]:
    """Compact assumption summary for UI/export."""
    return {
        "Dieselpreis_€/l": params.diesel_price_eur_per_l,
        "Strom_Mean_€/MWh": params.electricity_spot_mean_eur_per_mwh,
        "Strom_Stdev_€/MWh": params.electricity_spot_stdev_eur_per_mwh,
        "Betriebstage/Jahr": params.operating_days_per_year,
        "Horizont_Jahre": params.horizon_years,
        "Diskontsatz": params.discount_rate,
        "Batterie_RTE": params.battery_roundtrip_efficiency,
        "Batterie_DoD": params.battery_usable_dod,
    }


def to_input_dataframe(params: InputParameters) -> pd.DataFrame:
    data = asdict(params)
    return pd.DataFrame({"parameter": list(data.keys()), "value": list(data.values())})
