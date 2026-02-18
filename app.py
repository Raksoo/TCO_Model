from __future__ import annotations

from dataclasses import asdict
from io import StringIO
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from export import export_all
from model import (
    InputParameters,
    assumptions_summary,
    compute_cashflows,
    to_input_dataframe,
    validate_inputs,
)
from scenarios import run_scenarios, sensitivity_1d, tornado
from utils import co2, eur, format_dataframe_money, kwh, kw, pct, summary_box_text


st.set_page_config(page_title="TCO Baustellen-Energie", layout="wide")
st.title("TCO: Dieselgenerator vs Mobile Sodium-Ion Batterie")


@st.cache_data
def cached_cashflow(params_dict: dict):
    return compute_cashflows(InputParameters(**params_dict))


@st.cache_data
def cached_scenarios(params_dict: dict):
    return run_scenarios(InputParameters(**params_dict))


@st.cache_data
def cached_sensitivity(params_dict: dict, parameter: str, pct_range: float, steps: int):
    p = InputParameters(**params_dict)
    base = getattr(p, parameter)
    vmin = float(base) * (1.0 - pct_range)
    vmax = float(base) * (1.0 + pct_range)
    return sensitivity_1d(p, parameter, vmin, vmax, steps)


@st.cache_data
def cached_tornado(params_dict: dict, parameters: List[str], variation: float, top_n: int):
    return tornado(InputParameters(**params_dict), parameters, variation, top_n)


def _parse_csv_profile(upload) -> List[float]:
    if upload is None:
        return []
    content = upload.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("CSV enthält keine numerischen Werte")
    return numeric.iloc[:, 0].astype(float).tolist()[:24]


with st.sidebar:
    st.header("Kerneingaben")

    profile_mode = st.selectbox("Lastprofil", ["constant", "peak", "csv"], index=0)
    operating_hours = st.slider("Betriebsstunden/Tag", 0.0, 24.0, 10.0, 0.5)
    daily_energy = st.number_input("Tagesenergie (kWh/Tag)", min_value=0.0, value=100.0, step=1.0)

    col1, col2 = st.columns(2)
    with col1:
        constant_load = st.number_input("Konstant (kW)", min_value=0.0, value=10.0, step=0.5)
        base_load = st.number_input("Base (kW)", min_value=0.0, value=6.0, step=0.5)
    with col2:
        peak_load = st.number_input("Peak (kW)", min_value=0.0, value=18.0, step=0.5)
        peak_hours = st.slider("Peak Hours", 0.0, 24.0, 4.0, 0.5)

    csv_upload = st.file_uploader("CSV-Profil (24 Werte)", type=["csv"])

    st.subheader("Wirtschaft")
    operating_days = st.slider("Betriebstage/Jahr", 0, 365, 220)
    horizon = st.slider("Horizont (Jahre)", 1, 15, 5)
    discount = st.slider("Diskontsatz", 0.0, 0.30, 0.08, 0.005)
    comparison = st.selectbox("Vergleichsbasis", ["incremental", "replacement"], index=0)
    auto_size = st.checkbox("Auto-size Units", value=True)

    st.subheader("Diesel & Batterie")
    diesel_price = st.number_input("Dieselpreis (€/l)", min_value=0.0, value=1.75, step=0.01)
    diesel_power = st.number_input("Generatorleistung (kW)", min_value=1.0, value=40.0, step=1.0)
    diesel_eff = st.number_input("Effizienz bei Nennlast (kWh/l)", min_value=0.1, value=3.0, step=0.1)
    batt_cap = st.number_input("Batteriekapazität je Unit (kWh)", min_value=1.0, value=100.0, step=1.0)
    batt_units = st.number_input("Battery Units", min_value=1, value=1, step=1)
    batt_rte = st.slider("Roundtrip Efficiency", 0.1, 1.0, 0.88, 0.01)

    st.subheader("Strompreis")
    price_mode = st.selectbox("Strompreismodell", ["spot", "fixed"], index=0)
    spot_mean = st.number_input("Spot Mean (€/MWh)", min_value=0.0, value=99.88, step=0.1)
    spot_stdev = st.number_input("Spot Stdev (€/MWh)", min_value=0.0, value=21.09, step=0.1)
    spot_strategy = st.selectbox("Strategie", ["optimized", "average", "worst"], index=0)
    fixed_price = st.number_input("Fixpreis (€/kWh)", min_value=0.0, value=0.15, step=0.01)

    with st.expander("Erweiterte Annahmen", expanded=False):
        diesel_units = st.number_input("Diesel Units", min_value=1, value=1, step=1)
        diesel_min_load = st.slider("Diesel Min Load Factor", 0.1, 0.8, 0.3, 0.01)
        diesel_idle = st.number_input("Diesel Idle (l/h)", min_value=0.0, value=1.2, step=0.1)
        diesel_penalty = st.number_input("Low-load penalty (l/h)", min_value=0.0, value=0.3, step=0.05)
        diesel_maint = st.number_input("Diesel Wartung (€/Tag)", min_value=0.0, value=4.5, step=0.1)

        batt_dod = st.slider("Battery DoD", 0.1, 1.0, 0.8, 0.01)
        batt_discharge = st.number_input("Max Discharge je Unit (kW)", min_value=0.1, value=30.0, step=1.0)
        batt_charge = st.number_input("Max Charge je Unit (kW)", min_value=0.1, value=25.0, step=1.0)
        batt_charge_h = st.slider("Charge Hours", 1.0, 24.0, 12.0, 0.5)
        batt_fade = st.slider("Capacity fade/year", 0.0, 0.2, 0.03, 0.005)

        batt_capex = st.number_input("Battery CAPEX je Unit (€)", min_value=0.0, value=12500.0, step=500.0)
        charger_capex = st.number_input("Charger CAPEX (€)", min_value=0.0, value=2000.0, step=100.0)
        connection_capex = st.number_input("Connection CAPEX (€)", min_value=0.0, value=1500.0, step=100.0)
        diesel_capex = st.number_input("Diesel CAPEX (€)", min_value=0.0, value=50000.0, step=500.0)

        energy_fee = st.number_input("Energy Fee (€/kWh)", min_value=0.0, value=0.02, step=0.005)
        fixed_day = st.number_input("Fixed Fee (€/Tag)", min_value=0.0, value=0.0, step=0.1)
        fixed_month = st.number_input("Fixed Fee (€/Monat)", min_value=0.0, value=30.0, step=1.0)
        demand_fee = st.number_input("Demand Fee (€/kW/Monat)", min_value=0.0, value=0.0, step=0.1)

        carbon_enabled = st.checkbox("Carbon aktiv", value=False)
        carbon_price = st.number_input("CO2-Preis (€/t)", min_value=0.0, value=0.0, step=5.0)
        emission_factor = st.number_input("Diesel EF (kgCO2/l)", min_value=0.0, value=2.68, step=0.01)

# Defaults for collapsed advanced block if untouched
locals_defaults = locals()
diesel_units = int(locals_defaults.get("diesel_units", 1))
diesel_min_load = float(locals_defaults.get("diesel_min_load", 0.3))
diesel_idle = float(locals_defaults.get("diesel_idle", 1.2))
diesel_penalty = float(locals_defaults.get("diesel_penalty", 0.3))
diesel_maint = float(locals_defaults.get("diesel_maint", 4.5))

batt_dod = float(locals_defaults.get("batt_dod", 0.8))
batt_discharge = float(locals_defaults.get("batt_discharge", 30.0))
batt_charge = float(locals_defaults.get("batt_charge", 25.0))
batt_charge_h = float(locals_defaults.get("batt_charge_h", 12.0))
batt_fade = float(locals_defaults.get("batt_fade", 0.03))

batt_capex = float(locals_defaults.get("batt_capex", 12500.0))
charger_capex = float(locals_defaults.get("charger_capex", 2000.0))
connection_capex = float(locals_defaults.get("connection_capex", 1500.0))
diesel_capex = float(locals_defaults.get("diesel_capex", 50000.0))

energy_fee = float(locals_defaults.get("energy_fee", 0.02))
fixed_day = float(locals_defaults.get("fixed_day", 0.0))
fixed_month = float(locals_defaults.get("fixed_month", 30.0))
demand_fee = float(locals_defaults.get("demand_fee", 0.0))

carbon_enabled = bool(locals_defaults.get("carbon_enabled", False))
carbon_price = float(locals_defaults.get("carbon_price", 0.0))
emission_factor = float(locals_defaults.get("emission_factor", 2.68))

csv_values: List[float] = []
if profile_mode == "csv" and csv_upload is not None:
    try:
        csv_values = _parse_csv_profile(csv_upload)
    except Exception as exc:
        st.error(f"CSV-Fehler: {exc}")

params = InputParameters(
    profile_mode=profile_mode,
    constant_load_kw=constant_load,
    operating_hours_per_day=operating_hours,
    base_load_kw=base_load,
    peak_load_kw=peak_load,
    peak_hours=peak_hours,
    csv_profile_values=csv_values if csv_values else None,
    direct_daily_energy_kwh=None if daily_energy <= 0 else daily_energy,
    operating_days_per_year=operating_days,
    horizon_years=horizon,
    discount_rate=discount,
    auto_size_units=auto_size,
    comparison_basis=comparison,
    diesel_price_eur_per_l=diesel_price,
    diesel_rated_power_kw=diesel_power,
    diesel_units=diesel_units,
    diesel_kwh_per_l_rated=diesel_eff,
    diesel_min_load_factor=diesel_min_load,
    diesel_idle_fuel_l_per_h=diesel_idle,
    diesel_low_load_penalty_l_per_h=diesel_penalty,
    diesel_maintenance_eur_per_day=diesel_maint,
    diesel_capex_eur=diesel_capex,
    battery_units=int(batt_units),
    battery_nominal_capacity_kwh=batt_cap,
    battery_usable_dod=batt_dod,
    battery_roundtrip_efficiency=batt_rte,
    battery_max_discharge_power_kw=batt_discharge,
    battery_max_charge_power_kw=batt_charge,
    battery_charge_hours_per_day=batt_charge_h,
    battery_capacity_fade_per_year=batt_fade,
    battery_capex_unit_eur=batt_capex,
    charger_capex_eur=charger_capex,
    connection_capex_eur=connection_capex,
    electricity_price_mode=price_mode,
    electricity_fixed_eur_per_kwh=fixed_price,
    electricity_spot_mean_eur_per_mwh=spot_mean,
    electricity_spot_stdev_eur_per_mwh=spot_stdev,
    spot_strategy=spot_strategy,
    energy_fee_eur_per_kwh=energy_fee,
    fixed_fee_eur_per_day=fixed_day,
    fixed_fee_eur_per_month=fixed_month,
    demand_charge_eur_per_kw_month=demand_fee,
    carbon_enabled=carbon_enabled,
    carbon_price_eur_per_tco2=carbon_price,
    diesel_emission_factor_kgco2_per_l=emission_factor,
)

validation = validate_inputs(params)
for w in validation.warnings:
    st.warning(w)
if validation.errors:
    for e in validation.errors:
        st.error(e)
    st.stop()

result = cached_cashflow(asdict(params))
scenarios_df, _ = cached_scenarios(asdict(params))

if not result.daily.battery_feasible:
    st.error("Infeasible Batterie-Auslegung: " + " | ".join(result.daily.battery_feasibility_notes))

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Diesel OPEX/Tag", eur(result.daily.diesel_opex_day))
k2.metric("Batterie OPEX/Tag", eur(result.daily.battery_opex_day))
k3.metric("Einsparung/Tag", eur(result.daily.savings_day))
k4.metric("Payback", "n/a" if result.payback_years is None else f"{result.payback_years:.2f} Jahre")
k5.metric("NPV", eur(result.npv_eur))
k6.metric("IRR", "n/a" if result.irr is None else pct(result.irr))

k7, k8, k9 = st.columns(3)
k7.metric("Diesel €/kWh", eur(result.daily.diesel_opex_day / max(1e-9, result.daily.daily_energy_kwh)))
k8.metric("Batterie €/kWh", eur(result.daily.battery_opex_day / max(1e-9, result.daily.daily_energy_kwh)))
k9.metric("CO2-Einsparung/Tag", co2(result.daily.co2_savings_kg_day))


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inputs",
    "Daily",
    "Cashflow",
    "Szenarien",
    "Export",
])

with tab1:
    st.info(summary_box_text(assumptions_summary(params), []))
    st.write(f"Auto-size Ergebnis: Batterie Units = **{result.daily.battery_units_used}**, Diesel Units = **{result.daily.diesel_units_used}**")
    st.dataframe(to_input_dataframe(params), use_container_width=True)
    st.caption("Model limitations: vereinfachte Lastauflösung (stündlich), keine intraday-Optimierung, keine Steuermodellierung.")

with tab2:
    daily_df = pd.DataFrame([
        {"alternative": "Diesel", **result.daily.diesel_cost_breakdown, "total": result.daily.diesel_opex_day},
        {"alternative": "Batterie", **result.daily.battery_cost_breakdown, "total": result.daily.battery_opex_day},
    ]).fillna(0.0)
    st.dataframe(format_dataframe_money(daily_df, [c for c in daily_df.columns if c != "alternative"]), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Diesel", "Batterie"], [result.daily.diesel_opex_day, result.daily.battery_opex_day])
    ax.set_ylabel("€/Tag")
    ax.set_title("Daily OPEX")
    st.pyplot(fig)

with tab3:
    cashflow_df = pd.DataFrame({
        "year": list(range(0, params.horizon_years + 1)),
        "cashflow_eur": result.cashflows,
        "cumulative_cashflow_eur": result.cumulative_cashflows,
    })
    st.dataframe(format_dataframe_money(result.yearly_table, ["diesel_opex_year", "battery_opex_year", "savings_year"]), use_container_width=True)
    st.dataframe(format_dataframe_money(cashflow_df, ["cashflow_eur", "cumulative_cashflow_eur"]), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cashflow_df["year"], cashflow_df["cumulative_cashflow_eur"], marker="o")
    ax.axhline(0, linestyle="--")
    if result.payback_years is not None:
        ax.axvline(result.payback_years, linestyle=":")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Kumulierter Cashflow (€)")
    st.pyplot(fig)

with tab4:
    st.dataframe(scenarios_df, use_container_width=True)

    sens_param = st.selectbox(
        "Sensitivity Parameter",
        [
            "diesel_price_eur_per_l",
            "electricity_spot_mean_eur_per_mwh",
            "operating_days_per_year",
            "battery_roundtrip_efficiency",
        ],
    )
    sens_range = st.slider("Range (+/-)", 0.05, 0.50, 0.20, 0.05)
    sens_steps = st.slider("Schritte", 5, 30, 15)
    sens_df = cached_sensitivity(asdict(params), sens_param, sens_range, sens_steps)
    st.dataframe(sens_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sens_df["value"], sens_df["npv_eur"])
    ax.set_xlabel(sens_param)
    ax.set_ylabel("NPV (€)")
    ax.set_title("1D Sensitivity")
    st.pyplot(fig)

    tor_df = cached_tornado(
        asdict(params),
        [
            "diesel_price_eur_per_l",
            "electricity_spot_mean_eur_per_mwh",
            "electricity_spot_stdev_eur_per_mwh",
            "operating_days_per_year",
            "battery_capacity_fade_per_year",
        ],
        0.2,
        5,
    )
    st.dataframe(tor_df, use_container_width=True)

    if not tor_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        y = range(len(tor_df))
        ax.barh(y, tor_df["high_npv"] - tor_df["base_npv"], alpha=0.8, label="+20%")
        ax.barh(y, tor_df["low_npv"] - tor_df["base_npv"], alpha=0.8, label="-20%")
        ax.set_yticks(list(y))
        ax.set_yticklabels(tor_df["parameter"])
        ax.set_xlabel("NPV Delta (€)")
        ax.legend()
        st.pyplot(fig)

with tab5:
    daily_export = pd.DataFrame([
        {
            "daily_energy_kwh": result.daily.daily_energy_kwh,
            "peak_kw": result.daily.peak_kw,
            "avg_kw": result.daily.avg_kw,
            "diesel_opex_day": result.daily.diesel_opex_day,
            "battery_opex_day": result.daily.battery_opex_day,
            "savings_day": result.daily.savings_day,
            "diesel_fuel_l": result.daily.diesel_fuel_l,
            "diesel_co2_kg_day": result.daily.diesel_co2_kg_day,
            "co2_savings_kg_day": result.daily.co2_savings_kg_day,
            "battery_units": result.daily.battery_units_used,
            "diesel_units": result.daily.diesel_units_used,
        }
    ])
    cashflow_export = pd.DataFrame({
        "year": list(range(0, params.horizon_years + 1)),
        "cashflow_eur": result.cashflows,
        "cumulative_cashflow_eur": result.cumulative_cashflows,
    })

    sens_export = cached_sensitivity(asdict(params), "diesel_price_eur_per_l", 0.2, 15)
    payload = export_all(
        inputs_df=to_input_dataframe(params),
        daily_df=daily_export,
        cashflow_df=cashflow_export,
        scenarios_df=scenarios_df,
        sensitivity_df=sens_export,
    )

    st.download_button("Daily CSV", payload["daily_csv"], file_name="daily_comparison.csv", mime="text/csv")
    st.download_button("Cashflow CSV", payload["cashflow_csv"], file_name="cashflows.csv", mime="text/csv")
    st.download_button(
        "Excel (.xlsx)",
        payload["excel"],
        file_name="tco_model_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption(
    f"Lastprofil: {params.profile_mode} | Energie: {kwh(result.daily.daily_energy_kwh)} | "
    f"Peak: {kw(result.daily.peak_kw)} | Avg: {kw(result.daily.avg_kw)}"
)
