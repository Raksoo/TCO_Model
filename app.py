import numpy as np
import pandas as pd
import streamlit as st
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Matplotlib ist nicht installiert. Bitte: `pip install -r requirements.txt`")
    st.stop()


def compute_daily(
    daily_demand_kwh,
    diesel_price,
    generator_eff_kwh_per_l,
    part_load_factor,
    diesel_maintenance_per_day,
    diesel_logistics_per_day,
    electricity_price,
    charge_efficiency,
    battery_maintenance_per_day,
    include_co2_costs=False,
    co2_price_per_t=0.0,
    diesel_emission_factor_kg_per_l=2.68,
    diesel_consumption_override_l_per_day=None,
):
    """Berechnet tägliche OPEX für Diesel und Batterie."""
    diesel_base_l_per_day = daily_demand_kwh / generator_eff_kwh_per_l
    if diesel_consumption_override_l_per_day is None:
        diesel_l_per_day = diesel_base_l_per_day * part_load_factor
    else:
        diesel_l_per_day = diesel_consumption_override_l_per_day
        diesel_base_l_per_day = diesel_l_per_day / part_load_factor if part_load_factor > 0 else diesel_l_per_day
    diesel_energy_cost = diesel_l_per_day * diesel_price

    diesel_co2_kg_per_day = diesel_l_per_day * diesel_emission_factor_kg_per_l
    if include_co2_costs:
        diesel_co2_t_per_day = diesel_co2_kg_per_day / 1000.0
        diesel_co2_cost = diesel_co2_t_per_day * co2_price_per_t
    else:
        diesel_co2_cost = 0.0

    diesel_opex_per_day = (
        diesel_energy_cost + diesel_maintenance_per_day + diesel_logistics_per_day + diesel_co2_cost
    )

    battery_charge_kwh = daily_demand_kwh / charge_efficiency
    battery_energy_cost = battery_charge_kwh * electricity_price
    battery_opex_per_day = battery_energy_cost + battery_maintenance_per_day

    return {
        "diesel_base_l_per_day": diesel_base_l_per_day,
        "diesel_l_per_day": diesel_l_per_day,
        "diesel_energy_cost": diesel_energy_cost,
        "diesel_logistics_cost": diesel_logistics_per_day,
        "diesel_co2_kg_per_day": diesel_co2_kg_per_day,
        "diesel_co2_cost": diesel_co2_cost,
        "diesel_opex_per_day": diesel_opex_per_day,
        "battery_charge_kwh": battery_charge_kwh,
        "battery_energy_cost": battery_energy_cost,
        "battery_opex_per_day": battery_opex_per_day,
    }


def compute_cashflows(
    savings_per_day,
    operating_days_per_year,
    horizon_years,
    battery_capex,
    degradation_rate=0.0,
    total_battery_investment=None,
    annual_inflow_year_1=None,
):
    """Erstellt Cashflow-Reihe: Jahr 0 = -CAPEX, Jahr 1..n = jährliche Einsparung."""
    investment = battery_capex if total_battery_investment is None else total_battery_investment
    if annual_inflow_year_1 is None:
        annual_savings_year_1 = savings_per_day * operating_days_per_year
    else:
        annual_savings_year_1 = annual_inflow_year_1
    annual_savings_series = np.array(
        [annual_savings_year_1 * ((1.0 - degradation_rate) ** (year - 1)) for year in range(1, horizon_years + 1)],
        dtype=float,
    )
    cashflows = np.concatenate((np.array([-investment], dtype=float), annual_savings_series))
    years = np.arange(0, horizon_years + 1)
    cumulative_cashflow = np.cumsum(cashflows)

    return years, cashflows, cumulative_cashflow, annual_savings_year_1


def compute_npv(cashflows, discount_rate):
    """Berechnet NPV auf Basis der Cashflows und des Diskontsatzes."""
    r = discount_rate
    t = np.arange(len(cashflows))
    return float(np.sum(cashflows / np.power(1.0 + r, t)))


def compute_irr(cashflows, tol=1e-7, max_iter=1000):
    """Berechnet IRR via Bisektionsverfahren (robust ohne externe Pakete)."""

    def npv_at_rate(rate):
        t = np.arange(len(cashflows))
        return float(np.sum(cashflows / np.power(1.0 + rate, t)))

    low, high = -0.9999, 10.0
    f_low, f_high = npv_at_rate(low), npv_at_rate(high)

    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv_at_rate(mid)

        if abs(f_mid) < tol:
            return float(mid)

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return float((low + high) / 2.0)


def compute_payback(cumulative_cashflow):
    """Berechnet Payback in Jahren mit linearer Interpolation zwischen zwei Jahren."""
    for i in range(1, len(cumulative_cashflow)):
        prev_val = cumulative_cashflow[i - 1]
        curr_val = cumulative_cashflow[i]

        if curr_val >= 0:
            if prev_val == curr_val:
                return float(i)
            fraction = (0 - prev_val) / (curr_val - prev_val)
            return float((i - 1) + fraction)

    return np.nan


def apply_scenario(diesel_price, electricity_price, scenario):
    """Passt Energiepreise gemäß Szenario an."""
    if scenario == "Best Case":
        diesel_price *= 1.2
        electricity_price *= 0.8
    elif scenario == "Worst Case":
        diesel_price *= 0.8
        electricity_price *= 1.2
    return diesel_price, electricity_price


def format_payback(payback_years):
    if np.isnan(payback_years):
        return "Kein Payback im Horizont"
    return f"{payback_years:.2f} Jahre"


def format_irr(irr):
    if np.isnan(irr):
        return "n/a"
    return f"{irr * 100:.2f} %"


def compute_simple_amortization(total_investment, annual_revenue):
    """Einfache statische Amortisation: CAPEX / Einnahmen pro Jahr."""
    if annual_revenue <= 0:
        return np.nan
    return float(total_investment / annual_revenue)


def compute_profit_after_break_even(cumulative_cashflow, payback_years):
    """Gewinn nach Break-even bis zum Ende der Batterielebensdauer."""
    if np.isnan(payback_years):
        return np.nan
    return float(max(0.0, cumulative_cashflow[-1]))


def compute_case_npv(
    diesel_price_case,
    electricity_price_case,
    daily_demand,
    generator_eff,
    part_load_factor,
    diesel_maintenance,
    diesel_logistics,
    charge_efficiency,
    battery_maintenance,
    include_co2_costs,
    co2_price,
    diesel_emission_factor,
    operating_days,
    horizon_years,
    total_investment,
    degradation_rate,
    discount_rate,
    diesel_consumption_override_l_per_day=None,
    revenue_adjustment_factor=1.0,
):
    daily_case = compute_daily(
        daily_demand_kwh=daily_demand,
        diesel_price=diesel_price_case,
        generator_eff_kwh_per_l=generator_eff,
        part_load_factor=part_load_factor,
        diesel_maintenance_per_day=diesel_maintenance,
        diesel_logistics_per_day=diesel_logistics,
        electricity_price=electricity_price_case,
        charge_efficiency=charge_efficiency,
        battery_maintenance_per_day=battery_maintenance,
        include_co2_costs=include_co2_costs,
        co2_price_per_t=co2_price,
        diesel_emission_factor_kg_per_l=diesel_emission_factor,
        diesel_consumption_override_l_per_day=diesel_consumption_override_l_per_day,
    )
    savings_per_day_case = daily_case["diesel_opex_per_day"] - daily_case["battery_opex_per_day"]
    max_revenue_case = max(0.0, savings_per_day_case * operating_days)
    annual_revenue_case = max(0.0, max_revenue_case * revenue_adjustment_factor)
    _, cashflows_case, _, _ = compute_cashflows(
        savings_per_day=savings_per_day_case,
        operating_days_per_year=int(operating_days),
        horizon_years=int(horizon_years),
        battery_capex=total_investment,
        degradation_rate=degradation_rate,
        total_battery_investment=total_investment,
        annual_inflow_year_1=annual_revenue_case,
    )
    return compute_npv(cashflows_case, discount_rate)


st.set_page_config(page_title="TCO Baustellenenergie", layout="wide")
st.title("TCO-Modell: Dieselgenerator vs. Mobile Sodium-Ion Batterie")

with st.sidebar:
    st.header("Szenario")
    scenario = st.selectbox("Szenario auswählen", ["Base", "Best Case", "Worst Case"])

    st.header("Diesel")
    diesel_price = st.number_input("Dieselpreis (€/l)", min_value=0.0, value=1.75, step=0.05)
    generator_eff = st.number_input("Wirkungsgrad Generator (kWh/l)", min_value=0.1, value=3.0, step=0.1)
    part_load_factor = st.number_input("Teillast-Faktor", min_value=0.5, value=1.15, step=0.01, format="%.2f")
    use_manual_diesel_consumption = st.checkbox("Dieselverbrauch pro Tag manuell eingeben", value=False)
    diesel_consumption_per_day = st.number_input(
        "Dieselverbrauch pro Tag (l/Tag)",
        min_value=0.0,
        value=38.33,
        step=0.1,
        format="%.2f",
        disabled=not use_manual_diesel_consumption,
    )
    diesel_maintenance = st.number_input("Wartung Diesel (€/Tag)", min_value=0.0, value=15.0, step=1.0)
    diesel_logistics = st.number_input("Logistikkosten pro Tag (€)", min_value=0.0, value=0.0, step=5.0)

    st.header("Batterie")
    electricity_price = st.number_input("Strompreis (€/kWh)", min_value=0.0, value=0.10, step=0.01, format="%.2f")
    charge_efficiency = st.number_input("Ladeeffizienz", min_value=0.1, max_value=1.0, value=0.88, step=0.01, format="%.2f")
    battery_maintenance = st.number_input("Wartung Batterie (€/Tag)", min_value=0.0, value=7.0, step=1.0)
    battery_capex = st.number_input("Batterie CAPEX (€)", min_value=0.0, value=50000.0, step=500.0)
    charging_infra_capex = st.number_input("Ladeinfrastruktur CAPEX (€)", min_value=0.0, value=0.0, step=500.0)
    grid_connection_capex = st.number_input("Netzanschluss CAPEX (€)", min_value=0.0, value=0.0, step=500.0)
    battery_lifetime_years = st.number_input("Lebensdauer Batterie (Jahre)", min_value=1, max_value=30, value=5, step=1)
    degradation_pct = st.number_input("Degradation (% pro Jahr)", min_value=0.0, max_value=100.0, value=3.0, step=0.5)

    st.header("Allgemein")
    daily_demand = st.number_input("Tagesbedarf (kWh/Tag)", min_value=0.1, value=350.0, step=5.0)
    operating_days = st.number_input("Betriebstage pro Jahr", min_value=1, max_value=366, value=220, step=1)
    horizon_years = st.number_input("Betrachtungszeitraum (Jahre)", min_value=1, max_value=30, value=5, step=1)
    discount_rate_pct = st.number_input("Diskontsatz (%)", min_value=0.0, value=8.0, step=0.5)
    revenue_adjustment_pct = st.slider("Abschlag/Aufschlag Einnahmen (%)", min_value=-50, max_value=50, value=0, step=5)

    st.header("Optional CO2")
    include_co2_costs = st.checkbox("CO2-Kosten berücksichtigen", value=False)
    co2_price = st.number_input("CO2-Preis (€ pro t)", min_value=0.0, value=0.0, step=5.0, disabled=not include_co2_costs)
    diesel_emission_factor = st.number_input(
        "Emissionsfaktor Diesel (kg CO2/l)",
        min_value=0.0,
        value=2.68,
        step=0.01,
        format="%.2f",
        disabled=not include_co2_costs,
    )

adj_diesel_price, adj_electricity_price = apply_scenario(diesel_price, electricity_price, scenario)
discount_rate = discount_rate_pct / 100.0
degradation_rate = degradation_pct / 100.0
infrastructure_invest = charging_infra_capex + grid_connection_capex
total_battery_investment = battery_capex + infrastructure_invest
diesel_consumption_override = diesel_consumption_per_day if use_manual_diesel_consumption else None
effective_horizon_years = int(min(horizon_years, battery_lifetime_years))

# Kernberechnungen
daily = compute_daily(
    daily_demand_kwh=daily_demand,
    diesel_price=adj_diesel_price,
    generator_eff_kwh_per_l=generator_eff,
    part_load_factor=part_load_factor,
    diesel_maintenance_per_day=diesel_maintenance,
    diesel_logistics_per_day=diesel_logistics,
    electricity_price=adj_electricity_price,
    charge_efficiency=charge_efficiency,
    battery_maintenance_per_day=battery_maintenance,
    include_co2_costs=include_co2_costs,
    co2_price_per_t=co2_price,
    diesel_emission_factor_kg_per_l=diesel_emission_factor,
    diesel_consumption_override_l_per_day=diesel_consumption_override,
)

savings_per_day = daily["diesel_opex_per_day"] - daily["battery_opex_per_day"]
max_annual_revenue = max(0.0, savings_per_day * operating_days)
revenue_adjustment_factor = 1.0 + (revenue_adjustment_pct / 100.0)
annual_revenue = max(0.0, max_annual_revenue * revenue_adjustment_factor)
simple_amortization_years = compute_simple_amortization(total_battery_investment, annual_revenue)
years, cashflows, cumulative_cashflow, annual_savings = compute_cashflows(
    savings_per_day=savings_per_day,
    operating_days_per_year=int(operating_days),
    horizon_years=effective_horizon_years,
    battery_capex=battery_capex,
    degradation_rate=degradation_rate,
    total_battery_investment=total_battery_investment,
    annual_inflow_year_1=annual_revenue,
)
npv = compute_npv(cashflows, discount_rate)
irr = compute_irr(cashflows)
payback_years = compute_payback(cumulative_cashflow)
profit_after_break_even = compute_profit_after_break_even(cumulative_cashflow, payback_years)
co2_savings_per_year_t = (daily["diesel_co2_kg_per_day"] * operating_days) / 1000.0

with st.sidebar:
    st.markdown("---")
    st.subheader("Amortisation (statisch)")
    st.metric("Max. Einnahmen pro Jahr (€)", f"{max_annual_revenue:,.2f}")
    st.metric("Angesetzte Einnahmen pro Jahr (€)", f"{annual_revenue:,.2f}")
    st.metric("Amortisation (Jahre)", format_payback(simple_amortization_years))

st.caption(
    f"Aktives Szenario: **{scenario}** | Effektiver Dieselpreis: **{adj_diesel_price:.2f} €/l** | "
    f"Effektiver Strompreis: **{adj_electricity_price:.2f} €/kWh**"
)
if int(horizon_years) > int(battery_lifetime_years):
    st.info(
        f"Cashflow/Amortisation werden auf die Batterielebensdauer begrenzt: "
        f"{int(battery_lifetime_years)} Jahre (statt {int(horizon_years)} Jahre Betrachtungszeitraum)."
    )

tab1, tab2, tab3, tab4 = st.tabs(["Daily Vergleich", "Amortisation", "KPIs", "Sensitivität"])

with tab1:
    st.subheader("Täglicher OPEX-Vergleich")

    daily_df = pd.DataFrame(
        {
            "Kennzahl": [
                "Basisverbrauch Diesel",
                "Effektiver Verbrauch Diesel",
                "Energieeinsatz Batterie",
                "Energiekosten (€/Tag)",
                "Logistikkosten (€/Tag)",
                "CO2-Ausstoß Diesel (kg/Tag)",
                "CO2-Kosten (€/Tag)",
                "Wartung (€/Tag)",
                "Gesamtkosten OPEX (€/Tag)",
            ],
            "Dieselgenerator": [
                f"{daily['diesel_base_l_per_day']:.2f} l/Tag",
                f"{daily['diesel_l_per_day']:.2f} l/Tag",
                "-",
                f"{daily['diesel_energy_cost']:.2f}",
                f"{daily['diesel_logistics_cost']:.2f}",
                f"{daily['diesel_co2_kg_per_day']:.2f}",
                f"{daily['diesel_co2_cost']:.2f}",
                f"{diesel_maintenance:.2f}",
                f"{daily['diesel_opex_per_day']:.2f}",
            ],
            "Sodium-Ion Batterie": [
                "-",
                "-",
                f"{daily['battery_charge_kwh']:.2f} kWh/Tag",
                f"{daily['battery_energy_cost']:.2f}",
                "0.00",
                "0.00",
                f"0.00",
                f"{battery_maintenance:.2f}",
                f"{daily['battery_opex_per_day']:.2f}",
            ],
        }
    )
    st.dataframe(daily_df, use_container_width=True, hide_index=True)

    fig_opex, ax_opex = plt.subplots()
    ax_opex.bar(["Diesel", "Batterie"], [daily["diesel_opex_per_day"], daily["battery_opex_per_day"]])
    ax_opex.set_ylabel("€/Tag")
    ax_opex.set_title("OPEX pro Tag")
    st.pyplot(fig_opex)

with tab2:
    st.subheader("Cashflow und Amortisation")
    annual_savings_col = np.array(["-"] + [f"{v:,.2f}" for v in cashflows[1:]], dtype=object)

    cashflow_df = pd.DataFrame(
        {
            "Jahr": years,
            "Cashflow (€)": np.round(cashflows, 2),
            "Kumuliert (€)": np.round(cumulative_cashflow, 2),
            "Einsparung im Jahr (€)": annual_savings_col,
        }
    )
    st.dataframe(cashflow_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Statische Amortisation: CAPEX / Einnahmen pro Jahr = {total_battery_investment:,.2f} € / "
        f"{annual_revenue:,.2f} € = {format_payback(simple_amortization_years)}"
    )
    st.caption(
        "Cashflow-Linie endet mit dem Ersatzzeitpunkt (Batterie-Lebensdauer)."
    )
    if not np.isnan(profit_after_break_even):
        st.caption(f"Gewinn nach Break-even bis Ersatz: {profit_after_break_even:,.2f} €")

    fig_cf, ax_cf = plt.subplots()
    ax_cf.plot(years, cumulative_cashflow, marker="o")
    ax_cf.axhline(0, linestyle="--")

    if not np.isnan(payback_years):
        payback_value = np.interp(payback_years, years, cumulative_cashflow)
        ax_cf.scatter([payback_years], [payback_value], zorder=3)
        ax_cf.annotate(f"Payback: {payback_years:.2f} J", (payback_years, payback_value), textcoords="offset points", xytext=(8, 8))
    else:
        st.info("Im gewählten Horizont wird kein Payback erreicht.")

    ax_cf.set_xlabel("Jahr")
    ax_cf.set_ylabel("Kumulierter Cashflow (€)")
    ax_cf.set_title("Kumulierte Cashflows")
    st.pyplot(fig_cf)

with tab3:
    st.subheader("KPIs")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9, col10 = st.columns(4)
    col11, col12 = st.columns(2)

    col1.metric("OPEX Diesel (€/Tag)", f"{daily['diesel_opex_per_day']:.2f}")
    col2.metric("OPEX Batterie (€/Tag)", f"{daily['battery_opex_per_day']:.2f}")
    col3.metric("Einsparung (€/Tag)", f"{savings_per_day:.2f}")

    col4.metric("Payback", format_payback(payback_years))
    col5.metric("NPV (€)", f"{npv:,.2f}")
    col6.metric("IRR", format_irr(irr))
    col7.metric("Effektiver Dieselverbrauch (l/Tag)", f"{daily['diesel_l_per_day']:.2f}")
    col8.metric("Infrastruktur-Invest Batterie (€)", f"{infrastructure_invest:,.2f}")
    col9.metric("CO2-Ausstoß Diesel (kg/Tag)", f"{daily['diesel_co2_kg_per_day']:.2f}")
    col10.metric("CO2-Ersparnis pro Jahr (t)", f"{co2_savings_per_year_t:.2f}")
    col11.metric("Amortisation (CAPEX/Einnahmen)", format_payback(simple_amortization_years))
    col12.metric(
        "Gewinn nach Break-even bis Ersatz (€)",
        "n/a" if np.isnan(profit_after_break_even) else f"{profit_after_break_even:,.2f}",
    )

    st.markdown(
        f"**Jährliche Einsparung (Jahr 1):** {annual_savings:,.2f} €  \\\n"
        f"**Gesamtinvestition Batterie + Infrastruktur:** {total_battery_investment:,.2f} €  \\\n"
        f"**Annahmen:** {int(operating_days)} Betriebstage/Jahr, "
        f"Horizont {effective_horizon_years} Jahre (Lebensdauer {int(battery_lifetime_years)}), "
        f"Diskontsatz {discount_rate_pct:.2f}%, Degradation {degradation_pct:.2f}%, "
        f"Einnahmen-Anpassung {revenue_adjustment_pct:+d}%"
    )

with tab4:
    st.subheader("1D Sensitivität auf NPV")
    variation_pct = st.slider("Variation um den Basiswert (%)", min_value=0, max_value=30, value=30, step=5)
    variation = variation_pct / 100.0
    factors = [1.0 - variation, 1.0, 1.0 + variation]
    x_labels = [f"-{variation_pct}%", "Base", f"+{variation_pct}%"]

    diesel_npvs = [
        compute_case_npv(
            diesel_price_case=adj_diesel_price * factor,
            electricity_price_case=adj_electricity_price,
            daily_demand=daily_demand,
            generator_eff=generator_eff,
            part_load_factor=part_load_factor,
            diesel_maintenance=diesel_maintenance,
            diesel_logistics=diesel_logistics,
            charge_efficiency=charge_efficiency,
            battery_maintenance=battery_maintenance,
            include_co2_costs=include_co2_costs,
            co2_price=co2_price,
            diesel_emission_factor=diesel_emission_factor,
            operating_days=operating_days,
            horizon_years=effective_horizon_years,
            total_investment=total_battery_investment,
            degradation_rate=degradation_rate,
            discount_rate=discount_rate,
            diesel_consumption_override_l_per_day=diesel_consumption_override,
            revenue_adjustment_factor=revenue_adjustment_factor,
        )
        for factor in factors
    ]

    electricity_npvs = [
        compute_case_npv(
            diesel_price_case=adj_diesel_price,
            electricity_price_case=adj_electricity_price * factor,
            daily_demand=daily_demand,
            generator_eff=generator_eff,
            part_load_factor=part_load_factor,
            diesel_maintenance=diesel_maintenance,
            diesel_logistics=diesel_logistics,
            charge_efficiency=charge_efficiency,
            battery_maintenance=battery_maintenance,
            include_co2_costs=include_co2_costs,
            co2_price=co2_price,
            diesel_emission_factor=diesel_emission_factor,
            operating_days=operating_days,
            horizon_years=effective_horizon_years,
            total_investment=total_battery_investment,
            degradation_rate=degradation_rate,
            discount_rate=discount_rate,
            diesel_consumption_override_l_per_day=diesel_consumption_override,
            revenue_adjustment_factor=revenue_adjustment_factor,
        )
        for factor in factors
    ]

    sensitivity_df = pd.DataFrame(
        {
            "Variation": x_labels,
            "NPV Dieselpreis (€)": np.round(diesel_npvs, 2),
            "NPV Strompreis (€)": np.round(electricity_npvs, 2),
        }
    )
    st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    fig_sens, ax_sens = plt.subplots()
    ax_sens.plot(x_labels, diesel_npvs, marker="o", label="Dieselpreis")
    ax_sens.plot(x_labels, electricity_npvs, marker="o", label="Strompreis")
    ax_sens.set_ylabel("NPV (€)")
    ax_sens.set_title("Sensitivität NPV")
    ax_sens.legend()
    st.pyplot(fig_sens)
