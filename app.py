import streamlit as st
import pandas as pd
import numpy as np

# ---------- HomeRise core logic ----------

def homerise_simulator(home_value, stake_eur, tenure, market_cagr, min_irr):
    """
    Eenvoudig model:
    - HomeRise investeert stake_eur op t=0
    - Woning heeft startwaarde home_value en groeit met market_cagr per jaar
    - HomeRise ontvangt max van:
        1) minimum IRR floor
        2) percentage van de eindwaarde
    """

    if home_value is None or home_value <= 0:
        raise ValueError("home value must be > 0")
    if stake_eur is None or stake_eur < 0:
        raise ValueError("stake must be ≥ 0")
    if tenure <= 0:
        raise ValueError("tenure must be > 0")

    V0 = float(home_value)
    stake_eur = float(stake_eur)
    g = market_cagr / 100.0
    r_min = min_irr / 100.0
    T = int(tenure)

    # stake als % van property value
    stake_pct = min(stake_eur / V0, 1.0)

    years = list(range(0, T + 1))
    values = []
    for t in years:
        Vt = V0 * (1 + g) ** t
        values.append(Vt)

    # Eindwaarde woning
    V_T = values[-1]

    # Scenario 1: minimum IRR floor
    floor_payoff = stake_eur * (1 + r_min) ** T

    # Scenario 2: percentage van de eindwaarde
    share_payoff = stake_pct * V_T

    # Werkelijke payoff voor HomeRise
    homerise_payoff = max(floor_payoff, share_payoff)

    # IRR berekenen
    cashflows = [-stake_eur] + [0.0] * (T - 1) + [homerise_payoff]
    try:
        irr = np.irr(cashflows)
    except Exception:
        irr = None

    # Equity voor huiseigenaar
    owner_exit_equity = V_T - homerise_payoff

    df_values = pd.DataFrame({
        "year": years,
        "property_value": values
    })

    summary = {
        "initial_property_value": V0,
        "final_property_value": V_T,
        "stake_pct": stake_pct,
        "stake_eur": stake_eur,
        "floor_payoff": floor_payoff,
        "share_payoff": share_payoff,
        "homerise_payoff": homerise_payoff,
        "owner_exit_equity": owner_exit_equity,
        "irr": irr
    }

    return df_values, summary


# ---------- Streamlit UI ----------

st.set_page_config(page_title="HomeRise Simulator", layout="wide")

st.title("🏡 HomeRise Simulator")
st.write("Speel met de parameters en zie de payoff + IRR van HomeRise.")

# simple access gate
access_code = st.sidebar.text_input("access code", type="password")

if access_code != "HR2025": 
    st.warning("voer de juiste access code in om de simulator te gebruiken.")
    st.stop()

st.sidebar.header("Invoerparameters")

home_value = st.sidebar.number_input(
    "Woningwaarde vandaag (€)",
    min_value=50000.0, max_value=5_000_000.0,
    value=500_000.0, step=10_000.0, format="%.0f"
)

stake_eur = st.sidebar.number_input(
    "HomeRise investering (€)",
    min_value=0.0, max_value=2_000_000.0,
    value=100_000.0, step=10_000.0, format="%.0f"
)

tenure = st.sidebar.slider("Looptijd (jaar)", 1, 30, 10)

market_cagr = st.sidebar.slider(
    "Woninggroei (% per jaar)",
    min_value=-5.0, max_value=10.0,
    value=3.0, step=0.5
)

min_irr = st.sidebar.slider(
    "Minimum IRR HomeRise (% per jaar)",
    min_value=0.0, max_value=20.0,
    value=7.0, step=0.5
)

# Berekening
try:
    df_values, summary = homerise_simulator(
        home_value, stake_eur, tenure, market_cagr, min_irr
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kernresultaten")

        st.metric("Eindwaarde woning", f"€ {summary['final_property_value']:,.0f}")
        st.metric("Payoff HomeRise", f"€ {summary['homerise_payoff']:,.0f}")
        st.metric("Equity huiseigenaar bij exit", f"€ {summary['owner_exit_equity']:,.0f}")

        if summary["irr"] is not None:
            st.metric("IRR HomeRise", f"{summary['irr'] * 100:,.2f}%")
        else:
            st.write("IRR kon niet worden berekend.")

        st.write(f"Stake percentage: **{summary['stake_pct'] * 100:,.2f}%**")

    with col2:
        st.subheader("Waardeontwikkeling woning")
        chart_df = df_values.rename(columns={"year": "jaar", "property_value": "woningwaarde"})
        st.line_chart(chart_df.set_index("jaar"))

    st.subheader("Waarde per jaar")
    st.dataframe(df_values)

except Exception as e:
    st.error(f"Fout: {e}")

