import streamlit as st
import pandas as pd
import pdfplumber
import re
import io
from difflib import get_close_matches

# ---- Page setup ----
st.set_page_config(page_title="Finance Agent", layout="wide")
st.title("💰 Personal Finance Agent")

# ---- Bank Statement Announcement Banner ----
st.info(
    "🏦 **This app reads real bank statements!**  \n"
    "Upload a **PDF or CSV** exported directly from your bank "
    "(Chase, Bank of America, Wells Fargo, Citi, Capital One, etc.).  \n"
    "The agent automatically detects your transaction columns — no reformatting needed."
)

# ---- Session state ----
for key in ["df", "analyzed_df", "summary", "plan", "critique", "controller_output"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# PDF PARSING HELPER
# ============================================================

def parse_pdf_statement(uploaded_file):
    """
    Universal bank PDF parser.
    Strategy:
      1. Try pdfplumber table extraction (works for most banks).
      2. Fall back to line-by-line regex parsing for text-only PDFs.
    Returns a raw DataFrame or None.
    """
    rows = []
    file_bytes = uploaded_file.read()

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # ── Pass 1: structured table extraction ──
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                # First non-empty row is treated as header
                header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[0])]
                for row in table[1:]:
                    if any(cell and str(cell).strip() for cell in row):
                        rows.append(dict(zip(header, [str(c).strip() if c else "" for c in row])))

        if rows:
            return pd.DataFrame(rows)

        # ── Pass 2: regex line parser (text-only PDFs) ──
        # Pattern: date  description  optional-debit  optional-credit  balance
        # Handles formats like:  03/15/2024  WALMART #1234  -52.43  1,234.56
        date_pat  = r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})'
        money_pat = r'(-?\$?[\d,]+\.\d{2})'
        line_re   = re.compile(
            rf'^{date_pat}\s+(.+?)\s+{money_pat}(?:\s+{money_pat})?(?:\s+{money_pat})?\s*$'
        )

        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                m = line_re.match(line)
                if m:
                    date, desc = m.group(1), m.group(2).strip()
                    amounts = [g for g in [m.group(3), m.group(4), m.group(5)] if g]
                    # Use the first money value as Amount; last as Balance if present
                    raw_amt = amounts[0].replace('$', '').replace(',', '') if amounts else '0'
                    balance = amounts[-1].replace('$', '').replace(',', '') if len(amounts) > 1 else ''
                    rows.append({
                        'Date': date,
                        'Description': desc,
                        'Amount': raw_amt,
                        'Balance': balance
                    })

    return pd.DataFrame(rows) if rows else None


# ============================================================
# AGENT CLASSES
# ============================================================

class DataFetcherAgent:
    """Agent 1 – loads & normalises CSV or PDF bank statements."""

    def fetch_data(self, uploaded_file):
        filename = uploaded_file.name.lower()

        # ── Route by file type ──
        if filename.endswith(".pdf"):
            st.info("📄 PDF detected — running bank statement parser...")
            df = parse_pdf_statement(uploaded_file)
            if df is None or df.empty:
                st.error(
                    "❌ Could not extract transactions from this PDF.  \n"
                    "Make sure it is a **text-based PDF** (not a scanned image).  \n"
                    "Try downloading the statement again from your bank's website."
                )
                return None
            st.success(f"✅ PDF parsed! Found {len(df)} rows across {len(df.columns)} columns.")
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("📊 File Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # ── Column normalisation (same logic for both CSV & PDF) ──
        money_keys   = ["amount", "cost", "price", "debit", "credit", "charge", "total", "txn_amt", "withdrawal", "deposit"]
        desc_keys    = ["desc", "merchant", "payee", "description", "vendor", "name", "memo", "details", "transaction"]
        date_keys    = ["date", "period", "trans", "posted", "value date"]
        account_keys = ["account", "category", "type", "class"]

        def find_best(colnames, keywords):
            for col in colnames:
                if any(k in col.lower() for k in keywords):
                    return col
            lower = [c.lower() for c in colnames]
            hit = get_close_matches(" ".join(lower), keywords, n=1, cutoff=0.5)
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
            st.error("❌ Could not find Amount/Description columns. The PDF layout may be unsupported.")
            return None

        # Clean Amount column: remove $, commas, convert to float
        df["Amount"] = (
            df["Amount"].astype(str)
            .str.replace(r'[\$,]', '', regex=True)
            .str.strip()
        )
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df = df.dropna(subset=["Amount"])

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
            return "No data loaded yet. Upload a file first.", None

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

        total_exp      = abs(total_exp)
        monthly_exp    = total_exp / 3.0
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

        if "Account" in df.columns:
            total_exp = df.loc[df["Account"].str.lower() == "expenses", "Amount"].sum()
        else:
            total_exp = df.loc[df["Amount"] < 0, "Amount"].sum()
        total_exp = abs(total_exp)

        if total_exp > 20000:
            risks.append("⚠️ **High expenses detected** – review your biggest categories.")

        if "Account" in df.columns:
            income = df.loc[df["Account"].str.lower() == "income", "Amount"].sum()
        else:
            income = df.loc[df["Amount"] > 0, "Amount"].sum()

        if income == 0:
            risks.append("⚠️ **No income rows found** – make sure your statement includes deposits.")

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

st.header("1️⃣ Upload Your Bank Statement")
uploaded = st.file_uploader(
    "Upload a PDF or CSV bank statement (Chase, BofA, Wells Fargo, Citi, Capital One, etc.)",
    type=["csv", "pdf"]
)
if uploaded is not None:
    fetcher.fetch_data(uploaded)

st.header("2️⃣ Run the Agents")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🔍 Analyze", use_container_width=True):
        summary, analyzed_df = analyzer.analyze(st.session_state.df)
        st.session_state.summary     = summary
        st.session_state.analyzed_df = analyzed_df

with col2:
    if st.button("📈 Plan", use_container_width=True):
        if st.session_state.analyzed_df is None:
            st.warning("Run Analyze first.")
        else:
            st.session_state.plan = planner.plan(st.session_state.analyzed_df)

with col3:
    if st.button("🧐 Critique", use_container_width=True):
        if st.session_state.analyzed_df is None:
            st.warning("Run Analyze first.")
        else:
            st.session_state.critique = critic.critique(
                st.session_state.plan or "", st.session_state.analyzed_df
            )

with col4:
    st.markdown("**🤖 Controller**")
    cmd_input = st.text_input("Type a command", placeholder="e.g. analyze, plan, critique",
                              label_visibility="collapsed", key="cmd_input")
    if st.button("▶ Run Command", use_container_width=True):
        result = controller.process(cmd_input, st.session_state.df, st.session_state.analyzed_df)
        st.session_state.controller_output = result

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
