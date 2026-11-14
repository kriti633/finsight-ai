import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="FinSight AI — Multi-Sector Cash-Flow Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .alert-box {
        background-color: #3b1219;
        border-left: 4px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        color: #fecaca;
    }
    .safe-box {
        background-color: #063729;
        border-left: 4px solid #10b981;
        padding: 16px 20px;
        border-radius: 8px;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "finsight_multisector.db"

# ---------------------------------------------------------
# 1. SECTOR SAMPLE DATA GENERATORS
# ---------------------------------------------------------
def generate_agri_data() -> pd.DataFrame:
    """90 days of smallholder farm / agri-enterprise transactions."""
    np.random.seed(42)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start_date, end_date, freq="D")

    records = []
    for d in dates:
        day = d.day
        if day in [4, 22] and np.random.rand() > 0.25:
            records.append([d.strftime("%Y-%m-%d"), "APMC Mandi Crop Sale Payout", round(np.random.uniform(28000, 48000), 2)])
        if day == 15:
            records.append([d.strftime("%Y-%m-%d"), "Milk Co-op / Dairy Settlement", 9500.00])

        if day == 5:
            records.append([d.strftime("%Y-%m-%d"), "Tractor & Harvester Loan EMI", -11500.00])
        if day == 10:
            records.append([d.strftime("%Y-%m-%d"), "Cold Storage Warehouse Rent", -6000.00])
        if day == 18:
            records.append([d.strftime("%Y-%m-%d"), "Irrigation Pump Electricity Bill", -2400.00])

        if np.random.rand() > 0.45:
            items = [
                ("Urea & NPK Fertilizer Depot", -1800, -4500),
                ("Hybrid Seeds & Bio-Pesticides", -1200, -3800),
                ("Diesel for Tractor & Pump", -800, -2200),
                ("Daily Harvest Labour Wages", -1500, -3500),
                ("Agri-Transport Freight", -900, -2500),
            ]
            m, low, high = items[np.random.randint(0, len(items))]
            records.append([d.strftime("%Y-%m-%d"), m, round(np.random.uniform(high, low), 2)])

    records.append([(end_date - timedelta(days=14)).strftime("%Y-%m-%d"), "Emergency Drip Pipeline Repair", -18500.00])
    records.append([(end_date - timedelta(days=5)).strftime("%Y-%m-%d"), "Unseasonal Pest Control Bulk Spray", -14200.00])
    return pd.DataFrame(records, columns=["date", "description", "amount"]).sort_values("date").reset_index(drop=True)


def generate_healthcare_data() -> pd.DataFrame:
    """90 days of independent clinic / pharmacy / diagnostic lab transactions."""
    np.random.seed(99)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start_date, end_date, freq="D")

    records = []
    for d in dates:
        day = d.day
        if day in [6, 21] and np.random.rand() > 0.2:
            records.append([d.strftime("%Y-%m-%d"), "Insurance TPA Claim Reimbursement", round(np.random.uniform(35000, 62000), 2)])
        if d.weekday() in [1, 4]:
            records.append([d.strftime("%Y-%m-%d"), "OPD Consultation & Pharmacy UPI", round(np.random.uniform(4500, 9500), 2)])

        if day == 3:
            records.append([d.strftime("%Y-%m-%d"), "Diagnostic Ultrasound Machine EMI", -16500.00])
        if day == 7:
            records.append([d.strftime("%Y-%m-%d"), "Clinic & Pharmacy Premises Rent", -22000.00])
        if day == 12:
            records.append([d.strftime("%Y-%m-%d"), "Bio-Medical Waste Disposal Fee", -1800.00])
        if day == 25:
            records.append([d.strftime("%Y-%m-%d"), "Clinic EHR & Lab Software SaaS", -2500.00])

        if np.random.rand() > 0.4:
            items = [
                ("Wholesale Pharma Drug Restock", -2500, -7500),
                ("Pathology Lab Reagents & Kits", -1800, -5200),
                ("Surgical Gloves & PPE Consumables", -900, -2400),
                ("Oxygen Cylinder Refill", -1100, -2800),
            ]
            m, low, high = items[np.random.randint(0, len(items))]
            records.append([d.strftime("%Y-%m-%d"), m, round(np.random.uniform(high, low), 2)])

    records.append([(end_date - timedelta(days=11)).strftime("%Y-%m-%d"), "Emergency Cold-Chain Vaccine Fridge Fix", -24000.00])
    records.append([(end_date - timedelta(days=3)).strftime("%Y-%m-%d"), "Urgent Specialty Drug Batch Purchase", -19500.00])
    return pd.DataFrame(records, columns=["date", "description", "amount"]).sort_values("date").reset_index(drop=True)


def generate_society_data() -> pd.DataFrame:
    """90 days of NGO, Self-Help Group (SHG), or Community Cooperative transactions."""
    np.random.seed(77)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start_date, end_date, freq="D")

    records = []
    for d in dates:
        day = d.day
        # Irregular CSR Grants & Government Welfare Tranches
        if day in [8, 24] and np.random.rand() > 0.25:
            records.append([d.strftime("%Y-%m-%d"), "CSR Foundation Grant Tranche", round(np.random.uniform(32000, 55000), 2)])
        if day == 14:
            records.append([d.strftime("%Y-%m-%d"), "SHG Member Micro-Savings Pool", 8500.00])

        # Fixed Community Center Rent, Field Worker Stipends & Utilities
        if day == 2:
            records.append([d.strftime("%Y-%m-%d"), "Community Field Workers Stipend", -18000.00])
        if day == 6:
            records.append([d.strftime("%Y-%m-%d"), "Community Learning Center Rent", -9500.00])
        if day == 16:
            records.append([d.strftime("%Y-%m-%d"), "Rural Outreach Van Lease EMI", -7200.00])

        # Variable Social Welfare & Relief Expenses
        if np.random.rand() > 0.45:
            items = [
                ("Community Kitchen Ration & Grains", -1500, -4200),
                ("Vocational Training Kits & Books", -900, -2800),
                ("Outreach Van Fuel & Village Travel", -700, -1900),
                ("Clean Water Filter Cartridges", -1100, -2600),
            ]
            m, low, high = items[np.random.randint(0, len(items))]
            records.append([d.strftime("%Y-%m-%d"), m, round(np.random.uniform(high, low), 2)])

    # Inject 2 social/emergency anomalies
    records.append([(end_date - timedelta(days=13)).strftime("%Y-%m-%d"), "Emergency Flood Relief Ration Drive", -21000.00])
    records.append([(end_date - timedelta(days=4)).strftime("%Y-%m-%d"), "Urgent Solar Borewell Pump Repair", -15500.00])
    return pd.DataFrame(records, columns=["date", "description", "amount"]).sort_values("date").reset_index(drop=True)


def generate_tech_data() -> pd.DataFrame:
    """90 days of Indie SaaS, Tech Startup, or Dev Agency transactions."""
    np.random.seed(108)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start_date, end_date, freq="D")

    records = []
    for d in dates:
        day = d.day
        # Irregular Client Milestone Invoices & Stripe SaaS Payouts
        if day in [5, 20] and np.random.rand() > 0.2:
            records.append([d.strftime("%Y-%m-%d"), "Client Milestone Invoice Settlement", round(np.random.uniform(42000, 75000), 2)])
        if day == 12:
            records.append([d.strftime("%Y-%m-%d"), "Stripe SaaS Subscription Payout", 14500.00])

        # Fixed Developer Payroll, Cloud Servers & SaaS Tools
        if day == 1:
            records.append([d.strftime("%Y-%m-%d"), "Junior Dev & Designer Retainer", -28000.00])
        if day == 4:
            records.append([d.strftime("%Y-%m-%d"), "AWS Cloud & RDS Hosting Bill", -8400.00])
        if day == 9:
            records.append([d.strftime("%Y-%m-%d"), "GitHub, Vercel & Figma Team Seats", -3200.00])
        if day == 19:
            records.append([d.strftime("%Y-%m-%d"), "Co-Working Desk & Fiber Internet", -6500.00])

        # Variable Tech & API Usage Costs
        if np.random.rand() > 0.45:
            items = [
                ("LLM & Vector DB API Token Usage", -1200, -4500),
                ("Digital Ads & Developer Outreach", -1500, -4000),
                ("Domain Renewals & SSL Certs", -800, -2200),
                ("Freelance QA Testing Sprint", -1800, -4800),
            ]
            m, low, high = items[np.random.randint(0, len(items))]
            records.append([d.strftime("%Y-%m-%d"), m, round(np.random.uniform(high, low), 2)])

    # Inject 2 tech cost anomalies
    records.append([(end_date - timedelta(days=10)).strftime("%Y-%m-%d"), "Unthrottled GPU / API Traffic Spike", -26500.00])
    records.append([(end_date - timedelta(days=2)).strftime("%Y-%m-%d"), "Emergency Security Audit & Patch", -19000.00])
    return pd.DataFrame(records, columns=["date", "description", "amount"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. SQLITE STORAGE LAYER
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            amount REAL,
            category TEXT,
            is_recurring INTEGER,
            is_anomaly INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()


# ---------------------------------------------------------
# 3. MULTI-SECTOR AUTO-CATEGORISER & RECURRING DETECTOR
# ---------------------------------------------------------
CATEGORY_RULES = {
    "Grants, Payouts & Revenue": ["mandi", "crop", "dairy", "milk", "insurance", "tpa", "opd", "payout", "reimbursement", "grant", "csr", "savings", "invoice", "stripe", "sale"],
    "EMIs, Rent & Core Payroll": ["emi", "rent", "loan", "tractor", "ultrasound", "storage", "warehouse", "stipend", "retainer", "co-working", "lease"],
    "Agri Inputs & Crop Care": ["fertilizer", "urea", "seed", "pesticide", "pest", "irrigation", "drip"],
    "Healthcare & Lab Supplies": ["pharma", "drug", "reagent", "surgical", "oxygen", "vaccine", "bio-medical", "clinic", "ehr"],
    "Social Welfare & Community Aid": ["ration", "kitchen", "vocational", "training", "relief", "borewell", "water", "outreach"],
    "Cloud, APIs & Tech Infra": ["aws", "cloud", "github", "vercel", "figma", "llm", "api", "gpu", "domain", "security", "qa", "ads"],
    "Utilities, Fuel & Field Ops": ["diesel", "electricity", "pump", "freight", "transport", "fuel", "labour", "wages", "repair"],
}


def categorise_transaction(desc: str, amount: float) -> str:
    if amount > 0:
        return "Grants, Payouts & Revenue"
    desc_lower = str(desc).lower()
    for cat, keywords in CATEGORY_RULES.items():
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "General Operational Expense"


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_merchant"] = (
        df["description"]
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.strip()
    )

    recurring_merchants = set()
    expenses = df[df["amount"] < 0]

    for merchant, group in expenses.groupby("clean_merchant"):
        if len(group) >= 2:
            std_ratio = group["amount"].std() / (abs(group["amount"].mean()) + 1e-5)
            if std_ratio < 0.15:
                recurring_merchants.add(merchant)

    df["is_recurring"] = df["clean_merchant"].isin(recurring_merchants).astype(int)
    df.drop(columns=["clean_merchant"], inplace=True)
    return df


# ---------------------------------------------------------
# 4. ML ANOMALY DETECTION (Isolation Forest)
# ---------------------------------------------------------
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly"] = 0
    expenses = df[df["amount"] < 0].copy()

    if len(expenses) >= 10:
        expenses["abs_amount"] = expenses["amount"].abs()
        iso = IsolationForest(contamination=0.08, random_state=42)
        preds = iso.fit_predict(expenses[["abs_amount"]])
        df.loc[expenses.index, "is_anomaly"] = (preds == -1).astype(int)

    return df


# ---------------------------------------------------------
# 5. ML CASH-FLOW FORECAST & STRESS-TEST ENGINE
# ---------------------------------------------------------
def forecast_cash_flow(
    df: pd.DataFrame,
    current_balance: float,
    horizon_days: int = 30,
    sim_purchase_amount: float = 0.0,
    sim_purchase_day: int = 5,
    sim_monthly_emi: float = 0.0,
    payout_delay_days: int = 0,
    defer_savings: float = 0.0,
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()

    all_days = pd.date_range(df["date"].min(), last_date, freq="D")
    daily = (
        df.groupby("date")["amount"]
        .sum()
        .reindex(all_days, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "date", "amount": "net_flow"})
    )
    daily["day_idx"] = np.arange(len(daily))
    daily["day_of_month"] = daily["date"].dt.day
    daily["is_weekend"] = (daily["date"].dt.dayofweek >= 5).astype(int)

    features = ["day_idx", "day_of_month", "is_weekend"]
    model = Ridge(alpha=10.0)
    model.fit(daily[features], daily["net_flow"])

    recurring_df = df[(df["is_recurring"] == 1) & (df["amount"] < 0)]
    recurring_schedule = {}
    if not recurring_df.empty:
        recurring_df = recurring_df.copy()
        recurring_df["dom"] = recurring_df["date"].dt.day
        recurring_schedule = recurring_df.groupby("dom")["amount"].mean().to_dict()

    major_inflows = df[df["amount"] >= 10000]
    minor_inflows = df[(df["amount"] > 0) & (df["amount"] < 10000)]

    major_schedule = {}
    if not major_inflows.empty:
        major_inflows = major_inflows.copy()
        major_inflows["dom"] = major_inflows["date"].dt.day
        major_schedule = major_inflows.groupby("dom")["amount"].mean().to_dict()

    minor_schedule = {}
    if not minor_inflows.empty:
        minor_inflows = minor_inflows.copy()
        minor_inflows["dom"] = minor_inflows["date"].dt.day
        minor_schedule = minor_inflows.groupby("dom")["amount"].mean().to_dict()

    daily_std = max(daily["net_flow"].std() * 0.35, 500.0)

    future_rows = []
    running_baseline = current_balance
    running_simulated = current_balance

    for step in range(1, horizon_days + 1):
        f_date = last_date + timedelta(days=step)
        dom = f_date.day
        is_wknd = int(f_date.dayofweek >= 5)
        day_idx = len(daily) + step

        X_pred = pd.DataFrame([[day_idx, dom, is_wknd]], columns=features)
        base_drift = float(np.clip(model.predict(X_pred)[0], -1800, 1800))

        scheduled_bill = recurring_schedule.get(dom, 0.0)
        base_inflow = (major_schedule.get(dom, 0.0) + minor_schedule.get(dom, 0.0)) * 0.85

        net_baseline = base_drift * 0.4 + scheduled_bill + base_inflow
        running_baseline += net_baseline

        delayed_dom = (f_date - timedelta(days=payout_delay_days)).day
        stressed_inflow = (major_schedule.get(delayed_dom, 0.0) + minor_schedule.get(dom, 0.0)) * 0.85
        net_simulated = base_drift * 0.4 + scheduled_bill + stressed_inflow

        sim_delta = 0.0
        if step == sim_purchase_day and sim_purchase_amount > 0:
            sim_delta -= sim_purchase_amount
        if dom == 5 and sim_monthly_emi > 0:
            sim_delta -= sim_monthly_emi
        if step == 3 and defer_savings > 0:
            sim_delta += defer_savings

        running_simulated += net_simulated + sim_delta

        uncertainty = daily_std * np.sqrt(step) * 0.45
        future_rows.append({
            "date": f_date,
            "baseline_balance": round(running_baseline, 2),
            "simulated_balance": round(running_simulated, 2),
            "upper_bound": round(running_simulated + uncertainty, 2),
            "lower_bound": round(running_simulated - uncertainty, 2),
            "daily_net_sim": round(net_simulated + sim_delta, 2),
        })

    return pd.DataFrame(future_rows)


# ---------------------------------------------------------
# 6. SECTOR-AWARE EXPLAINABLE AI ADVICE GENERATOR
# ---------------------------------------------------------
def generate_advice(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    current_balance: float,
    domain: str,
    payout_delay_days: int,
    api_key: str = "",
) -> str:
    min_bal = forecast_df["simulated_balance"].min()
    min_date = forecast_df.loc[forecast_df["simulated_balance"].idxmin(), "date"].strftime("%b %d")
    recurring_total = abs(df[(df["is_recurring"] == 1) & (df["amount"] < 0)]["amount"].sum()) / 3.0
    anomaly_count = int(df["is_anomaly"].sum())

    if api_key and GEMINI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            You are FinSight AI specializing in {domain} cash-flow intelligence.
            Current working capital: ₹{current_balance:,.0f}.
            Lowest predicted balance in next 30 days: ₹{min_bal:,.0f} on {min_date}.
            Inflow delay stress test: {payout_delay_days} days.
            Fixed monthly obligations: ₹{recurring_total:,.0f}.
            Unexpected operational cost spikes detected: {anomaly_count}.
            Give 3 crisp, domain-specific bullet points (under 65 words total) to prevent a liquidity crunch.
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception:
            pass

    tips = []
    if "Agriculture" in domain:
        if min_bal < 0:
            tips.append(
                f"**Pre-Harvest Cash Crunch ({min_date}):** Balance drops to **₹{min_bal:,.0f}** "
                f"{'(with ' + str(payout_delay_days) + '-day Mandi delay)' if payout_delay_days > 0 else 'before next Mandi payout'}. "
                f"Defer non-urgent inputs via the Action Planner or arrange **₹{abs(min_bal) + 3000:,.0f}** KCC buffer."
            )
        else:
            tips.append(f"**Crop Cycle Buffer Safe:** Minimum liquidity holds at **₹{min_bal:,.0f}** ({min_date}) across tractor EMI & cold-storage bills.")
        tips.append(f"**Fixed Agri Commitments:** **₹{recurring_total:,.0f}/month** is locked in equipment EMIs, storage, and pump power.")

    elif "Healthcare" in domain:
        if min_bal < 0:
            tips.append(
                f"**TPA Reimbursement Gap ({min_date}):** Balance hits **₹{min_bal:,.0f}** "
                f"{'(with ' + str(payout_delay_days) + '-day TPA claim delay)' if payout_delay_days > 0 else 'before insurance payouts clear'}. "
                f"Stagger bulk pharma restocking or reserve **₹{abs(min_bal) + 5000:,.0f}**."
            )
        else:
            tips.append(f"**Clinic Liquidity Healthy:** Working capital stays above **₹{min_bal:,.0f}** ({min_date}) through diagnostic EMI and rent cycles.")
        tips.append(f"**Fixed Clinical Overhead:** **₹{recurring_total:,.0f}/month** is committed to diagnostic EMIs, rent, and compliance.")

    elif "Society" in domain:
        if min_bal < 0:
            tips.append(
                f"**Grant Tranche Liquidity Gap ({min_date}):** Community fund drops to **₹{min_bal:,.0f}** "
                f"{'(with ' + str(payout_delay_days) + '-day CSR/grant delay)' if payout_delay_days > 0 else 'before next grant disbursement'}. "
                f"Tap SHG micro-reserve or phase non-emergency outreach spend by **₹{abs(min_bal) + 2500:,.0f}**."
            )
        else:
            tips.append(f"**Community Fund Resilient:** Treasury remains positive at **₹{min_bal:,.0f}** ({min_date}) while covering all field-worker stipends.")
        tips.append(f"**Fixed Social Commitments:** **₹{recurring_total:,.0f}/month** supports field-worker stipends, center rent, and outreach transport.")

    elif "Technology" in domain:
        if min_bal < 0:
            tips.append(
                f"**Runway Burn Crunch ({min_date}):** Balance dips to **₹{min_bal:,.0f}** "
                f"{'(with ' + str(payout_delay_days) + '-day client invoice delay)' if payout_delay_days > 0 else 'ahead of client milestone settlement'}. "
                f"Cap API token bursts or reserve **₹{abs(min_bal) + 5000:,.0f}** runway buffer."
            )
        else:
            tips.append(f"**Tech Runway Healthy:** Cash stays above **₹{min_bal:,.0f}** ({min_date}) across developer retainers and AWS cloud bills.")
        tips.append(f"**Fixed Tech Burn:** **₹{recurring_total:,.0f}/month** is committed to payroll retainers, cloud hosting, and SaaS seats.")

    else:
        tips.append(f"**Projected Minimum ({min_date}):** **₹{min_bal:,.0f}** with **₹{recurring_total:,.0f}/mo** in fixed obligations.")

    if anomaly_count > 0:
        tips.append(f"**Operational Shock Detected:** {anomaly_count} unexpected cost spike(s) flagged recently that narrowed your safety margin.")

    return "\n\n".join(tips)


# ---------------------------------------------------------
# 7. MAIN STREAMLIT UI
# ---------------------------------------------------------
def main():
    init_db()

    with st.sidebar:
        st.title("🌐 FinSight AI")
        st.caption("Agri · Healthcare · Society · Tech")
        st.divider()

        domain_choice = st.radio(
            "1. Select Domain / Data Source",
            [
                "🌾 Agriculture (Farm & Agri-Input)",
                "🏥 Healthcare (Clinic & Pharmacy)",
                "🤝 Society (NGO, SHG & Co-op)",
                "💻 Technology (SaaS & Dev Agency)",
                "📂 Upload Custom Bank CSV",
            ],
        )

        uploaded_file = None
        if "Upload" in domain_choice:
            uploaded_file = st.file_uploader("Upload CSV (date, description, amount)", type=["csv"])

        bal_map = {
            "Agriculture": 32000,
            "Healthcare": 45000,
            "Society": 28000,
            "Technology": 52000,
        }
        default_bal = next((v for k, v in bal_map.items() if k in domain_choice), 35000)

        current_balance = st.number_input(
            "2. Current Working Capital (₹)",
            min_value=0,
            max_value=50000000,
            value=default_bal,
            step=1000,
        )

        st.divider()
        st.subheader("⏱️ Inflow Delay Stress-Test")
        if "Agriculture" in domain_choice:
            delay_label = "Mandi / Harvest Payout Delay (Days)"
            purchase_label = "Bulk Seeds / Fertilizer / Repair (₹)"
        elif "Healthcare" in domain_choice:
            delay_label = "Insurance / TPA Claim Delay (Days)"
            purchase_label = "Bulk Pharma / Lab Equipment Outlay (₹)"
        elif "Society" in domain_choice:
            delay_label = "CSR / Govt Grant Disbursement Delay (Days)"
            purchase_label = "Emergency Relief / Community Drive (₹)"
        else:
            delay_label = "Client Invoice / Stripe Payout Delay (Days)"
            purchase_label = "GPU Cluster / Cloud / Audit Outlay (₹)"

        payout_delay = st.slider(delay_label, 0, 21, 0, step=1)

        st.divider()
        st.subheader("🎛️ Sector What-If Simulator")
        sim_purchase = st.slider(purchase_label, 0, 80000, 0, step=1000)
        sim_day = st.slider("Days From Today", 1, 28, 7)
        sim_emi = st.slider("New Monthly Recurring Commitment (₹)", 0, 25000, 0, step=500)

        st.divider()
        gemini_key = st.text_input("Optional Gemini API Key", type="password")

    # Load Sector Data
    if "Upload" in domain_choice and uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        raw_df.columns = [c.strip().lower() for c in raw_df.columns]
        if not {"date", "description", "amount"}.issubset(raw_df.columns):
            st.error("CSV must contain columns: `date`, `description`, `amount`")
            st.stop()
    elif "Healthcare" in domain_choice:
        raw_df = generate_healthcare_data()
    elif "Society" in domain_choice:
        raw_df = generate_society_data()
    elif "Technology" in domain_choice:
        raw_df = generate_tech_data()
    else:
        raw_df = generate_agri_data()

    # Intelligence Pipeline
    raw_df["category"] = raw_df.apply(lambda r: categorise_transaction(r["description"], r["amount"]), axis=1)
    processed_df = detect_recurring(raw_df)
    processed_df = detect_anomalies(processed_df)
    save_to_db(processed_df)

    st.title("FinSight AI — Cross-Sector Liquidity Forecaster")
    st.caption("BUILD BEYOND 26 · PS 2.2 Cash Flow · Agriculture, Healthcare, Society & Technology")

    # Sector-Specific Shortfall Mitigation Action Planner
    with st.expander("🛠️ Shortfall Action Planner (Simulate Cost-Optimization Moves)", expanded=False):
        act_col1, act_col2 = st.columns(2)
        defer_savings = 0.0
        if "Agriculture" in domain_choice:
            if act_col1.checkbox("Defer 50% Bulk Fertilizer Order until Mandi Payout (+₹6,500)"):
                defer_savings += 6500.0
            if act_col2.checkbox("Use Cooperative Tractor Sharing Instead of Spot Rental (+₹4,500)"):
                defer_savings += 4500.0
        elif "Healthcare" in domain_choice:
            if act_col1.checkbox("Split Wholesale Pharma Restock into Weekly Batches (+₹9,000)"):
                defer_savings += 9000.0
            if act_col2.checkbox("Defer Non-Urgent Lab Consumables Order (+₹5,500)"):
                defer_savings += 5500.0
        elif "Society" in domain_choice:
            if act_col1.checkbox("Phase Non-Urgent Training Kits to Next Grant Cycle (+₹5,000)"):
                defer_savings += 5000.0
            if act_col2.checkbox("Pool Village Outreach Transport Across Clusters (+₹3,500)"):
                defer_savings += 3500.0
        else:
            if act_col1.checkbox("Enable LLM Token Caching & Rate Limits (+₹6,000)"):
                defer_savings += 6000.0
            if act_col2.checkbox("Pause Non-Essential Ad Campaigns Until Invoice Clears (+₹7,500)"):
                defer_savings += 7500.0

    forecast_df = forecast_cash_flow(
        processed_df,
        current_balance=current_balance,
        horizon_days=30,
        sim_purchase_amount=sim_purchase,
        sim_purchase_day=sim_day,
        sim_monthly_emi=sim_emi,
        payout_delay_days=payout_delay,
        defer_savings=defer_savings,
    )

    min_sim_bal = forecast_df["simulated_balance"].min()
    min_sim_date = forecast_df.loc[forecast_df["simulated_balance"].idxmin(), "date"].strftime("%d %b %Y")
    recurring_df = processed_df[processed_df["is_recurring"] == 1]
    anomalies_df = processed_df[processed_df["is_anomaly"] == 1]

    avg_daily_burn = abs(processed_df[processed_df["amount"] < 0]["amount"].sum()) / 90.0
    cash_runway_days = int(current_balance / max(avg_daily_burn, 1.0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Working Capital", f"₹{current_balance:,.0f}", delta=f"~{cash_runway_days} Days Runway", delta_color="off")
    col2.metric(
        "30-Day Lowest Balance",
        f"₹{min_sim_bal:,.0f}",
        delta=f"On {min_sim_date}",
        delta_color="inverse" if min_sim_bal < 0 else "normal",
    )
    col3.metric(
        "Fixed Commitments",
        f"{recurring_df['description'].nunique()} Obligations",
        delta=f"₹{abs(recurring_df['amount'].sum()) / 3:,.0f} / mo",
        delta_color="off",
    )
    col4.metric(
        "Operational Shock Alerts",
        f"{len(anomalies_df)} Flagged",
        delta="Isolation Forest ML",
        delta_color="off",
    )

    if min_sim_bal < 0:
        st.markdown(
            f"""
            <div class="alert-box">
                <strong>⚠️ PREDICTED LIQUIDITY SHORTFALL ON {min_sim_date.upper()}:</strong>
                Your simulated working capital drops to <strong>₹{min_sim_bal:,.0f}</strong> due to timing mismatch between inflows and fixed obligations.
                Open the <strong>Shortfall Action Planner</strong> above to test cost-saving adjustments.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="safe-box">
                <strong>✅ WORKING CAPITAL SAFE:</strong>
                Liquidity remains positive over the next 30 days (lowest point: <strong>₹{min_sim_bal:,.0f}</strong> on {min_sim_date}).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    chart_col, advice_col = st.columns([2.2, 1])

    with chart_col:
        st.subheader("📈 30-Day Working Capital Forecast & Stress-Test Simulation")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Uncertainty Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["baseline_balance"],
                mode="lines",
                name="Baseline Forecast",
                line=dict(color="#94a3b8", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["simulated_balance"],
                mode="lines+markers",
                name="Stressed / Simulated Balance",
                line=dict(color="#10b981", width=3),
                marker=dict(size=5),
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#ef4444", annotation_text="₹0 Shortfall Threshold")
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Projected Working Capital (₹)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with advice_col:
        st.subheader("🧠 Sector AI Advisory")
        st.info(generate_advice(processed_df, forecast_df, current_balance, domain_choice, payout_delay, api_key=gemini_key))

        st.subheader("📊 Expense Breakdown")
        exp_by_cat = processed_df[processed_df["amount"] < 0].groupby("category")["amount"].sum().abs().reset_index()
        pie_fig = px.pie(exp_by_cat, values="amount", names="category", hole=0.45, color_discrete_sequence=px.colors.sequential.Emrld_r)
        pie_fig.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        pie_fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(pie_fig, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔁 Recurring Commitments & Payroll",
        "🚨 Operational Cost Shocks (Anomalies)",
        "📄 Full Sector Transaction Ledger",
        "🏦 Lender / Grant Audit Export",
    ])

    with tab1:
        if recurring_df.empty:
            st.write("No recurring obligations detected.")
        else:
            rec_summary = recurring_df.groupby(["description", "category"])["amount"].agg(Occurrences="count", Avg_Amount="mean").reset_index()
            rec_summary["Avg_Amount"] = rec_summary["Avg_Amount"].abs().round(2)
            st.dataframe(rec_summary, use_container_width=True, hide_index=True)

    with tab2:
        if anomalies_df.empty:
            st.success("No unexpected cost spikes detected.")
        else:
            st.dataframe(anomalies_df[["date", "description", "category", "amount"]], use_container_width=True, hide_index=True)

    with tab3:
        st.dataframe(processed_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

    with tab4:
        st.write("Download this 30-day cash-flow projection for **KCC Agri Loans**, **Clinic Working Capital**, **CSR Grant Audits**, or **Venture Runway Reports**.")
        csv_export = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download 30-Day Forecast CSV Report",
            data=csv_export,
            file_name="finsight_30day_sector_forecast.csv",
            mime="text/csv",
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()