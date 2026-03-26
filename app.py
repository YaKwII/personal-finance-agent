import streamlit as st
import pandas as pd
from difflib import get_close_matches

# ---- Page setup ----
st.set_page_config(page_title="Finance Agent", layout="wide")
st.title("💰 Personal Finance Agent")

# ---- Session state ----
for key in ["df", "analyzed_df", "summary", "plan", "critique", "controller_output"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# AGENT CLASSES
# ============================================================

class DataFetcherAgent:
    """Agent 1 – loads & normalises the CSV."""
    def fetch_data(self, uploaded_file):
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 File Preview")
        st.dataframe(df.head(), use_container_width=True)

        money_keys   = ["amount", "cost", "price", "debit", "credit", "charge", "total", "txn_amt"]
        desc_keys    = ["desc", "merchant", "payee", "description", "vendor", "name", "memo"]
        date_keys    = ["date", "period", "trans", "posted"]
        account_keys = ["account", "category", "type"]

        def find_best(colnames, keywords):
            for col in colnames:
                if any(k in col.lower() for k in keywords):
                    return col
            lower = [c.lower() for c in colnames]
            hit = get_close_matches(" ".join(lower), keywords, n=1, cutoff=0.6)
            if hit:
                for col in colnames:
                    if hit[0] in col.lower():
                        return col
            return None

        cols = list(df.columns)
        amount_col  = find_best(cols, money_keys)
        desc_col    = find_best(cols, desc_keys)
        date_col    = find_best(cols, date_keys)
        account_col = find_best(cols, account_keys)

        if amount_col is None:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols): amount_col = num_cols[0]
        if desc_col is None:
            obj_cols = df.select_dtypes(include="object").columns
            if len(obj_cols): desc_col = obj_cols[0]

        rename_map = {}
        if amount_col:  rename_map[amount_col]  = "Amount"
        if desc_col:    rename_map[desc_col]    = "Description"
        if date_col:    rename_map[date_col]    = "Date"
        if account_col: rename_map[account_col] = "Account"
        df = df.rename(columns=rename_map)

        if "Amount" not in df.columns or "Description" not in df.columns:
            st.error("❌ Could not find Amount/Description columns.")
            return None

        st.success(
            f"✅ Columns matched!\n\n"
            f"- Amount → `{amount_col}`\n"
            f"- Description → `{desc_col}`\n"
            f"- Date → `{date_col}`\n"
            f"- Account → `{account_col}`"
        )
        st.session_state.df = df
        return df


class AnalyzerAgent:
    """Agent 2 – computes income, expenses, and category breakdown."""
    def analyze(self, df):
        if df is None or df.empty:
            return "No data loaded yet. Upload a CSV first.", None

        data = df.copy()
        data["Category"] = data["Account"] if "Account" in data.columns else "General"

        if "Account" in data.columns:
            acc = data["Account"].str.lower()
            income   = data.loc[acc == "income",   "Amount"].sum()
            expenses = data.loc[acc == "expenses",  "Amount"].sum()
        else:
            income   = data.loc[data["Amount"] > 0, "Amount"].sum()
            expenses = data.loc[data["Amount"] < 0, "Amount"].sum()

        expenses_abs = abs(expenses)
        net = income - expenses_abs

        exp_rows = data[data["Amount"] < 0].copy()
        by_cat = (
            exp_rows.groupby("Category")["Amount"]
            .sum().abs()
            .sort_values(ascending=False)
        )

        summary = (
            f"### 💰 Summary\n\n"
            f"- **Income**: ${income:,.2f}\n"
            f"- **Expenses**: ${expenses_abs:,.2f}\n"
            f"- **Net**: ${net:,.2f}\n\n"
            f"### 🧾 Top Expense Categories\n"
        )
        if not by_cat.empty:
            for cat, amt in by_cat.items():
                summary += f"- {cat}: ${amt:,.2f}\n"
        else:
            summary += "- No expense rows detected.\n"

        return summary.strip(), data


class PlannerAgent:
    """Agent 3 – generates a simple budget plan."""
    def plan(self, df):
        if df is None or df.empty:
            return "Run Analyze first so I can see your spending."

        if "Account" in df.columns:
            total_exp = df.loc[df["Account"].str.lower() == "expenses", "Amount"].sum()
        else:
            total_exp = df.loc[df["Amount"] < 0, "Amount"].sum()

        total_exp     = abs(total_exp)
        monthly_exp   = total_exp / 3.0
        emergency_fund = monthly_exp * 3

        return (
            f"### 📈 Simple Financial Plan\n\n"
            f"- Quarterly expenses ≈ **${total_exp:,.2f}**\n"
            f"- Monthly expenses ≈ **${monthly_exp:,.2f}**\n"
            f"- Suggested emergency fund (3 mo): **${emergency_fund:,.2f}**\n\n"
            f"**Ideas:**\n"
            f"- Save 10–20% of income toward your emergency fund each month.\n"
            f"- Find your biggest expense category and cut it by 10%."
        ).strip()


class CriticAgent:
    """Agent 4 – reviews the plan and flags risks."""
    def critique(self, plan_text, df):
        if df is None:
            return "No data available for risk analysis."

        risks = []

        # Risk 1: very high total expenses
        if "Account" in df.columns:
            total_exp = df.loc[df["Account"].str.lower() == "expenses", "Amount"].sum()
        else:
            total_exp = df.loc[df["Amount"] < 0, "Amount"].sum()
        total_exp = abs(total_exp)

        if total_exp > 20000:
            risks.append("⚠️ **High expenses detected** – review your biggest categories.")

        # Risk 2: no income rows found
        if "Account" in df.columns:
            income = df.loc[df["Account"].str.lower() == "income", "Amount"].sum()
        else:
            income = df.loc[df["Amount"] > 0, "Amount"].sum()

        if income == 0:
            risks.append("⚠️ **No income rows found** – make sure your CSV has income data.")

        # Risk 3: negative net worth
        net = income - total_exp
        if net < 0:
            risks.append(f"⚠️ **Negative net (${net:,.2f})** – you are spending more than you earn!")

        feedback = "### 🧐 Critic Review\n\n"
        feedback += "\n".join(risks) if risks else "✅ Plan looks solid! No major risks detected."
        return feedback.strip()


class ControllerAgent:
    """Agent 5 – orchestrates the other agents via natural-language commands."""
    def process(self, command, df, analyzed_df):
        cmd = command.lower()
        if "analyze" in cmd:
            summary, new_df = analyzer.analyze(df)
            st.session_state.summary      = summary
            st.session_state.analyzed_df  = new_df
            return f"✅ Analyzer ran!\n\n{summary}"
        elif "plan" in cmd:
            if analyzed_df is None:
                return "⚠️ Run Analyze first before planning."
            result = planner.plan(analyzed_df)
            st.session_state.plan = result
            return f"✅ Planner ran!\n\n{result}"
        elif "critic" in cmd or "critique" in cmd or "review" in cmd:
            if analyzed_df is None:
                return "⚠️ Run Analyze first so the Critic has data."
            result = critic.critique(st.session_state.plan or "", analyzed_df)
            st.session_state.critique = result
            return f"✅ Critic ran!\n\n{result}"
        elif "reset" in cmd:
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        else:
            return "🤖 I understand: **analyze**, **plan**, **critique/review**, or **reset**."


# ---- Instantiate agents ----
fetcher    = DataFetcherAgent()
analyzer   = AnalyzerAgent()
planner    = PlannerAgent()
critic     = CriticAgent()
controller = ControllerAgent()

# ============================================================
# UI
# ============================================================

st.header("1️⃣ Upload Your CSV")
uploaded = st.file_uploader("Upload a CSV (e.g., john_doe_financial_report.csv)", type=["csv"])
if uploaded is not None:
    fetcher.fetch_data(uploaded)

st.header("2️⃣ Run the Agents")
col1, col2, col3, col4, col5 = st.columns(5)

# --- Analyze ---
with col1:
    if st.button("🔍 Analyze", use_container_width=True):
        summary, analyzed_df = analyzer.analyze(st.session_state.df)
        st.session_state.summary     = summary
        st.session_state.analyzed_df = analyzed_df

# --- Plan ---
with col2:
    if st.button("📈 Plan", use_container_width=True):
        if st.session_state.analyzed_df is None:
            st.warning("Run Analyze first.")
        else:
            st.session_state.plan = planner.plan(st.session_state.analyzed_df)

# --- Critique ---
with col3:
    if st.button("🧐 Critique", use_container_width=True):
        if st.session_state.analyzed_df is None:
            st.warning("Run Analyze first.")
        else:
            st.session_state.critique = critic.critique(
                st.session_state.plan or "", st.session_state.analyzed_df
            )

# --- Controller (NL command) ---
with col4:
    st.markdown("**🤖 Controller**")
    cmd_input = st.text_input("Type a command", placeholder="e.g. analyze, plan, critique", label_visibility="collapsed", key="cmd_input")
    if st.button("▶ Run Command", use_container_width=True):
        result = controller.process(cmd_input, st.session_state.df, st.session_state.analyzed_df)
        st.session_state.controller_output = result

# --- Reset ---
with col5:
    if st.button("🔄 Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================
# RESULTS
# ============================================================

st.header("3️⃣ Results")

if st.session_state.summary:
    st.markdown(st.session_state.summary)

if st.session_state.analyzed_df is not None:
    st.subheader("📄 Analyzed Data")
    st.dataframe(st.session_state.analyzed_df, use_container_width=True)

if st.session_state.plan:
    st.subheader("🗺️ Plan")
    st.markdown(st.session_state.plan)

if st.session_state.critique:
    st.subheader("🧐 Critic Feedback")
    st.markdown(st.session_state.critique)

if st.session_state.controller_output:
    st.subheader("🤖 Controller Output")
    st.markdown(st.session_state.controller_output)

st.caption("CIS 4394 – Multi-Agent Finance App ❤️")
