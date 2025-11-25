import streamlit as st
import pandas as pd
import numpy as np

# ---------- helpers ----------

def euro(x):
    """Format number as euro without decimals, with dot as thousands separator."""
    try:
        return "€ {:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return x


def xirr(cashflows):
    """Compute IRR using Newton's method — replacement for np.irr."""
    try:
        guess = 0.1
        for _ in range(100):
            npv = sum(cf / ((1 + guess) ** t) for t, cf in enumerate(cashflows))
            d_npv = sum(
                -t * cf / ((1 + guess) ** (t + 1)) for t, cf in enumerate(cashflows)
            )
            if abs(npv) < 1e-8:
                return guess
            guess = guess - npv / d_npv
        return guess
    except Exception:
        return None


def mortgage_profiles(loan_amount, annual_rate, years, owner_cost_case3):
    """
    Bereken jaarlijkse 'kosten' als verloren geld:

    - HomeRise: kosten alleen in het laatste jaar en alleen bij Case 3 (market CAGR < min IRR)
      => owner_cost_case3
    - Aflossingsvrij: alleen rente (geen aflossing)
    - Lineair: alleen rente per jaar (aflossing telt niet als kosten)
    - Annuïtair: alleen rentecomponent per jaar (aflossing telt niet als kosten)

    Alles op jaarbasis (kosten in jaar t, niet cumulatief).
    """
    r = annual_rate
    n = years
    years_list = list(range(1, n + 1))

    # HomeRise: alleen 'kosten' in laatste jaar, en alleen als owner_cost_case3 > 0
    hr_costs = []
    for t in years_list:
        if t == n and owner_cost_case3 > 0:
            hr_costs.append(owner_cost_case3)
        else:
            hr_costs.append(0.0)

    # Aflossingsvrij: alleen rente, schuld blijft gelijk
    io_costs = [loan_amount * r for _ in years_list]

    # Lineair: alleen rente per jaar, aflossing is vermogensopbouw
    linear_costs = []
    yearly_principal = loan_amount / n if n > 0 else 0.0
    remaining = loan_amount
    for _ in years_list:
        interest = remaining * r
        linear_costs.append(interest)   # alleen rente als 'kosten'
        remaining -= yearly_principal
        remaining = max(remaining, 0.0)

    # Annuïtair: jaarlijkse betaling = rente + aflossing,
    # maar we tellen alleen de rentecomponent als 'kosten'
    annuity_costs = []
    if r == 0 or n == 0:
        annual_payment = loan_amount / n if n > 0 else 0.0
    else:
        annual_payment = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    remaining = loan_amount
    for _ in years_list:
        interest = remaining * r
        principal = annual_payment - interest
        remaining -= principal
        remaining = max(remaining, 0.0)
        annuity_costs.append(interest)  # alleen rente als 'kosten'

    df = pd.DataFrame(
        {
            "jaar": years_list,
            "HomeRise (alleen Case 3)": hr_costs,
            "Aflossingsvrij": io_costs,
            "Lineair": linear_costs,
            "Annuïtair": annuity_costs,
        }
    )

    return df

# ---------- HomeRise core logic ----------

def homerise_simulator(home_value, stake_eur, tenure, market_cagr, min_irr):
    """
    Simpel HomeRise-model:

    - HomeRise investeert stake_eur op t=0
    - Woning heeft startwaarde home_value en groeit met market_cagr per jaar
    - HomeRise ontvangt de max van:
        1) minimum IRR floor
        2) percentage van de eindwaarde (stake_pct * V_T)

    'Kosten' voor de eigenaar definiëren we als:
    - Alleen in Case 3: market CAGR < min IRR
    - Kosten = floor_payoff - share_payoff (extra waarde die naar HomeRise gaat
      boven het pro-rata aandeel in de woning)
    - In Case 1 & 2: kosten = 0 (HomeRise verdient mee met groei die zij mede mogelijk maakt)
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

    # stake als % van property value (max 100%)
    stake_pct = min(stake_eur / V0, 1.0)

    years = list(range(0, T + 1))
    values = []
    for t in years:
        Vt = V0 * (1 + g) ** t
        values.append(Vt)

    # Eindwaarde woning
    V_T = values[-1]

    # Pro-rata payoff (equity share) - wat 'eerlijk' zou zijn als HomeRise puur equity is
    share_payoff = stake_pct * V_T

    # Floor payoff (minimale IRR)
    floor_payoff = stake_eur * (1 + r_min) ** T

    # Werkelijke payoff voor HomeRise
    homerise_payoff = max(floor_payoff, share_payoff)

    # IRR berekenen (cashflows: -stake_eur op t=0, payoff op t=T)
    cashflows = [-stake_eur] + [0.0] * (T - 1) + [homerise_payoff]
    irr = xirr(cashflows)

    # Equity voor huiseigenaar bij exit
    owner_exit_equity = V_T - homerise_payoff

    # Kosten voor de eigenaar (alleen Case 3: market CAGR < min IRR => floor > share)
    owner_cost_case3 = max(floor_payoff - share_payoff, 0.0)

    # Basic eigenaar-metrics (simple benadering)
    owner_equity_start = V0 - stake_eur  # woningwaarde - HomeRise-inbreng
    owner_equity_start = max(owner_equity_start, 0.0)

    owner_share_appreciation = owner_exit_equity - owner_equity_start
    total_appreciation = V_T - V0
    if total_appreciation > 0:
        owner_share_of_app = owner_share_appreciation / total_appreciation
    else:
        owner_share_of_app = None

    df_values = pd.DataFrame(
        {
            "year": years,
            "property_value": values,
        }
    )

    summary = {
        "initial_property_value": V0,
        "final_property_value": V_T,
        "stake_pct": stake_pct,
        "stake_eur": stake_eur,
        "floor_payoff": floor_payoff,
        "share_payoff": share_payoff,
        "homerise_payoff": homerise_payoff,
        "owner_exit_equity": owner_exit_equity,
        "irr": irr,
        "owner_cost_case3": owner_cost_case3,
        "owner_equity_start": owner_equity_start,
        "owner_share_appreciation": owner_share_appreciation,
        "total_appreciation": total_appreciation,
        "owner_share_of_app": owner_share_of_app,
    }

    return df_values, summary


# ---------- Streamlit UI ----------

st.set_page_config(page_title="HomeRise Simulator", layout="wide")

st.title("🏡 HomeRise Simulator")
st.write(
    "Vergelijk HomeRise met klassieke hypotheekopties. "
    "HomeRise biedt kapitaal zonder rente en aflossing; "
    "de gedefinieerde 'kosten' voor de eigenaar ontstaan alleen in het scenario waarin "
    "de woning minder hard groeit dan de minimum IRR van HomeRise (Case 3)."
)

# simple access gate (pas de code aan naar wat jij wilt)
access_code = st.sidebar.text_input("access code", type="password")

if access_code != "HR2025":
    st.warning("voer de juiste access code in om de simulator te gebruiken.")
    st.stop()

st.sidebar.header("Invoerparameters")

# Vrije invoer (intypen) voor woningwaarde en HomeRise-kapitaal
home_value = st.sidebar.number_input(
    "Woningwaarde vandaag (€)",
    min_value=1.0,
    value=500_000.0,
    step=10_000.0,
    format="%.0f",
)

stake_eur = st.sidebar.number_input(
    "HomeRise kapitaal / extra financiering (€)",
    min_value=0.0,
    value=100_000.0,
    step=10_000.0,
    format="%.0f",
)

tenure = st.sidebar.slider("Looptijd (jaar)", min_value=1, max_value=30, value=10)

market_cagr = st.sidebar.slider(
    "Woninggroei (% per jaar)",
    min_value=-5.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
)

min_irr = st.sidebar.slider(
    "Minimum IRR HomeRise (% per jaar)",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
)

mortgage_rate = st.sidebar.slider(
    "Hypotheekrente (% per jaar voor vergelijking)",
    min_value=2.0,
    max_value=7.0,
    value=4.5,
    step=0.1,
)

# ---------- Berekening & output ----------

try:
    df_values, summary = homerise_simulator(
        home_value=home_value,
        stake_eur=stake_eur,
        tenure=tenure,
        market_cagr=market_cagr,
        min_irr=min_irr,
    )

    col1, col2 = st.columns(2)

    # ----- HomeRise & eigenaar kernresultaten -----

    with col1:
        st.subheader("HomeRise kernresultaten")

        st.metric("Startwaarde woning", euro(summary["initial_property_value"]))
        st.metric("Eindwaarde woning", euro(summary["final_property_value"]))
        st.metric("Payoff HomeRise bij exit", euro(summary["homerise_payoff"]))

        if summary["irr"] is not None:
            st.metric("IRR HomeRise", f"{summary['irr'] * 100:,.2f}%".replace(",", "."))
        else:
            st.write("IRR kon niet worden berekend (check de input).")

        st.write(
            f"Stake als % van de woning vandaag: **{summary['stake_pct'] * 100:,.2f}%**".replace(
                ",", "."
            )
        )

        with st.expander("Detail payoff-componenten"):
            st.write(f"Floor payoff (min IRR): {euro(summary['floor_payoff'])}")
            st.write(
                f"Share payoff (stake % * eindwaarde): {euro(summary['share_payoff'])}"
            )
            st.write(f"Initiële HomeRise investering: {euro(summary['stake_eur'])}")

            if summary["owner_cost_case3"] > 0:
                st.write(
                    f"Extra waarde naar HomeRise in Case 3 (kosten eigenaar): "
                    f"{euro(summary['owner_cost_case3'])}"
                )
            else:
                st.write(
                    "In deze configuratie is er geen extra waardeoverdracht naar HomeRise "
                    "(kosten eigenaar = 0; Case 1 of 2)."
                )

    with col2:
        st.subheader("Huiseigenaar")

        st.metric("Equity eigenaar bij start", euro(summary["owner_equity_start"]))
        st.metric("Equity eigenaar bij exit", euro(summary["owner_exit_equity"]))

        if summary["total_appreciation"] > 0 and summary["owner_share_of_app"] is not None:
            st.metric(
                "Aandeel eigenaar in waardestijging",
                f"{summary['owner_share_of_app'] * 100:,.2f}%".replace(",", "."),
            )
        else:
            st.write("Geen positieve waardestijging; aandeel in waardestijging niet van toepassing.")

        st.subheader("Waardeontwikkeling woning")

        chart_df = df_values.rename(
            columns={"year": "jaar", "property_value": "woningwaarde"}
        )
        st.line_chart(chart_df.set_index("jaar"))

    st.subheader("Waarde per jaar")
    df_display = df_values.copy()
    df_display["property_value"] = df_display["property_value"].apply(euro)
    st.dataframe(df_display)

    # ---------- Hypotheekvergelijking: jaarlijkse kosten & cumulatief ----------

    st.subheader("Jaarlijkse kosten: HomeRise vs extra hypotheek")

    cost_df = mortgage_profiles(
        loan_amount=stake_eur,
        annual_rate=mortgage_rate / 100.0,
        years=tenure,
        owner_cost_case3=summary["owner_cost_case3"],
    )

    # Tabel met kosten per jaar (niet cumulatief)
    df_cost_display = cost_df.copy()
    for col in ["HomeRise (alleen Case 3)", "Aflossingsvrij", "Lineair", "Annuïtair"]:
        df_cost_display[col] = df_cost_display[col].apply(euro)

    st.write("Jaarlijkse cash out / kosten per optie (bedragen per jaar):")
    st.dataframe(df_cost_display)

    # Cumulatieve kosten voor de grafiek
    cost_df_cum = cost_df.copy()
    for col in ["HomeRise (alleen Case 3)", "Aflossingsvrij", "Lineair", "Annuïtair"]:
        cost_df_cum[col] = cost_df_cum[col].cumsum()

    st.write("Cumulatieve kosten over de looptijd:")
    st.line_chart(cost_df_cum.set_index("jaar"))

    st.caption(
        "De grafiek toont cumulatieve kosten: voor hypotheken zijn dit de som van rente en aflossing, "
        "voor HomeRise alleen een mogelijke 'kostenpiek' in het laatste jaar in Case 3 "
        "(wanneer de woning minder hard groeit dan de minimum IRR van HomeRise)."
    )

except Exception as e:
    st.error(f"Er is een fout opgetreden: {e}")

