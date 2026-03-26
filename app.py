import streamlit as st
import pandas as pd
import pdfplumber
import re
import io
from difflib import get_close_matches
import google.generativeai as genai

# OCR imports (for scanned image-based PDFs)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---- Page setup ----
st.set_page_config(page_title="Finance Agent", layout="wide")
st.title("💰 Personal Finance Agent")

# ---- Bank Statement Announcement Banner ----
st.info(
    "🏦 **This app reads real bank statements!**  \n"
    "Upload a **PDF or CSV** exported directly from your bank "
    "(Chase, Bank of America, Wells Fargo, Navy Federal, Citi, Capital One, etc.).  \n"
    "Supports both **digital PDFs** and **scanned paper statements** (OCR).  \n"
    "The agent automatically detects your transaction columns — no reformatting needed."
)

# ---- Gemini API Key (sidebar) ----
with st.sidebar:
    st.header("🤖 AI Settings")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here",
        help="Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.caption("Your key is never stored or sent anywhere except Google's API.")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        st.success("✅ Gemini connected!")
    else:
        st.warning("⚠️ Add your Gemini key to unlock the AI Critic.")

    st.divider()
    st.header("📷 OCR Settings")
    if OCR_AVAILABLE:
        st.success("✅ OCR ready (pytesseract installed)")
    else:
        st.warning("⚠️ OCR not available. Run: pip install pytesseract pdf2image")
    tesseract_path = st.text_input(
        "Tesseract Path (Windows only)",
        placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Leave blank on Mac/Linux. On Windows, paste the path to tesseract.exe"
    )
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ---- Session state ----
for key in ["df", "analyzed_df", "summary", "plan", "critique", "controller_output", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ============================================================
# HELPERS
# ============================================================

def clean_amount(raw):
    """
    Convert bank amount strings to signed floats.
    Handles:
      - Standard negatives:  '-23.43'  or  '($23.43)'
      - Navy Federal style:  '23.43-'  (minus at end)
      - Plain positives:     '418.00'
    """
    if raw is None:
        return None
    s = str(raw).strip().replace(',', '').replace('$', '').replace(' ', '')
    if not s:
        return None
    if s.endswith('-'):
        s = '-' + s[:-1]
    s = re.sub(r'^\((.+)\)$', r'-\1', s)
    try:
        return float(s)
    except ValueError:
        return None


def ocr_pdf_to_text(file_bytes):
    """
    Convert a scanned (image-based) PDF to text using OCR.
    Each page is rendered as an image, then pytesseract reads it.
    Returns a single string with all pages joined.
    """
    if not OCR_AVAILABLE:
        return None
    try:
        images = convert_from_bytes(file_bytes, dpi=300)
        full_text = ""
        for img in images:
            full_text += pytesseract.image_to_string(img) + "\n"
        return full_text
    except Exception as e:
        st.warning(f"⚠️ OCR failed: {e}")
        return None


def parse_text_to_transactions(text):
    """
    Parse raw text (from OCR or pdfplumber fallback) into transaction rows
    using regex. Handles multiple date formats and amount styles.
    """
    rows = []
    # Matches: MM-DD, MM/DD, MM-DD-YYYY, MM/DD/YYYY, Mon DD
    date_pat  = r'(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})'
    money_pat = r'(-?\$?[\d,]+\.\d{2}-?)'
    line_re   = re.compile(
        rf'^{date_pat}\s+(.+?)\s+{money_pat}(?:\s+{money_pat})?(?:\s+{money_pat})?\s*$'
    )
    for line in text.splitlines():
        line = line.strip()
        m = line_re.match(line)
        if m:
            date = m.group(1)
            desc = m.group(2).strip()
            raw_amounts = [g for g in [m.group(3), m.group(4), m.group(5)] if g]
            amt = clean_amount(raw_amounts[0]) if raw_amounts else None
            bal = clean_amount(raw_amounts[-1]) if len(raw_amounts) > 1 else None
            if amt is None:
                continue
            # Skip header rows OCR might misread as transactions
            if desc.lower() in ["transaction detail", "description", "details"]:
                continue
            rows.append({"Date": date, "Description": desc, "Amount": amt, "Balance": bal})
    return rows


def is_text_pdf(file_bytes):
    """Check if a PDF has real embedded text (not just images)."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.strip()) > 50:  # meaningful text found
                return True
    return False


# ============================================================
# PDF PARSING — 3-strategy pipeline
# ============================================================

def parse_pdf_statement(uploaded_file):
    """
    Extract transactions from any PDF bank statement.

    Strategy 1 — Structured table extraction  (digital PDFs with tables)
    Strategy 2 — Line-by-line regex            (digital PDFs, no tables)
    Strategy 3 — OCR + regex                   (scanned paper statements)

    Returns a cleaned DataFrame with columns:
        Date | Description | Amount | Balance
    """
    rows = []
    file_bytes = uploaded_file.read()

    # --------------------------------------------------------
    # STRATEGY 1 — structured table extraction
    # --------------------------------------------------------
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                raw_header = table[0]
                header = [
                    str(c).strip().lower() if c else f"col_{i}"
                    for i, c in enumerate(raw_header)
                ]
                has_date = any(
                    any(k in h for k in ["date", "col_0"]) for h in header
                )
                if not has_date:
                    continue

                for row in table[1:]:
                    if not any(cell and str(cell).strip() for cell in row):
                        continue
                    row_dict = {
                        header[i]: str(cell).strip() if cell else ""
                        for i, cell in enumerate(row)
                        if i < len(header)
                    }
                    date_val = desc_val = amt_val = bal_val = ""
                    for k, v in row_dict.items():
                        if "date" in k:
                            date_val = v
                        elif any(x in k for x in ["transaction detail", "description", "memo",
                                                   "merchant", "payee", "details", "desc", "name"]):
                            desc_val = v
                        elif any(x in k for x in ["amount($)", "amount", "debit",
                                                   "withdrawal", "charge"]):
                            if not amt_val:
                                amt_val = v
                        elif any(x in k for x in ["balance($)", "balance"]):
                            bal_val = v

                    if not date_val and "col_0" in row_dict: date_val = row_dict["col_0"]
                    if not desc_val and "col_1" in row_dict: desc_val = row_dict["col_1"]
                    if not amt_val  and "col_2" in row_dict: amt_val  = row_dict["col_2"]
                    if not bal_val  and "col_3" in row_dict: bal_val  = row_dict["col_3"]

                    if not date_val or date_val.lower() in ["date", ""]:
                        continue
                    if desc_val.lower() in ["beginning balance", "ending balance",
                                            "transaction detail"]:
                        continue
                    if not re.search(r'\d{1,2}[-/]\d{1,2}', date_val):
                        continue

                    cleaned_amt = clean_amount(amt_val)
                    cleaned_bal = clean_amount(bal_val)
                    if cleaned_amt is None:
                        continue

                    rows.append({
                        "Date":        date_val,
                        "Description": desc_val,
                        "Amount":      cleaned_amt,
                        "Balance":     cleaned_bal,
                    })

    if rows:
        df = pd.DataFrame(rows)
        df["Amount"]  = pd.to_numeric(df["Amount"],  errors="coerce")
        df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
        return df

    # --------------------------------------------------------
    # STRATEGY 2 — line-by-line regex (digital, no tables)
    # --------------------------------------------------------
    if is_text_pdf(file_bytes):
        st.info("📝 No tables found — trying line-by-line text parser...")
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"
        rows = parse_text_to_transactions(full_text)
        if rows:
            return pd.DataFrame(rows)

    # --------------------------------------------------------
    # STRATEGY 3 — OCR (scanned image-based PDFs)
    # --------------------------------------------------------
    if OCR_AVAILABLE:
        st.info("📷 No text found in PDF — running OCR on scanned pages (this may take 10–30 seconds)...")
        ocr_text = ocr_pdf_to_text(file_bytes)
        if ocr_text:
            rows = parse_text_to_transactions(ocr_text)
            if rows:
                st.success("✅ OCR complete! Transactions extracted from scanned statement.")
                return pd.DataFrame(rows)
        st.error("❌ OCR ran but could not extract transactions. The scan quality may be too low.")
    else:
        st.error(
            "❌ This appears to be a scanned PDF (image-based), but OCR is not installed.  \n"
            "To enable scanned statement support, run:  \n"
            "`pip install pytesseract pdf2image`  \n"
            "and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    return None


# ============================================================
# AGENT CLASSES
# ============================================================

class DataFetcherAgent:
    """Agent 1 – loads & normalises CSV or PDF bank statements."""
    def fetch_data(self, uploaded_file):
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            st.info("📄 PDF detected — running bank statement parser...")
            df = parse_pdf_statement(uploaded_file)
            if df is None or df.empty:
                st.error(
                    "❌ Could not extract transactions from this PDF.  \n"
                    "Make sure it is a text-based PDF (not a scanned image) or install pytesseract for OCR support."
                )
                return None
            st.success(f"✅ PDF parsed! Found {len(df)} transaction rows.")
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("📊 File Preview")
        st.dataframe(df.head(15), use_container_width=True)

        if filename.endswith(".pdf") and "Amount" in df.columns:
            st.session_state.df = df
            return df

        money_keys   = ["amount", "cost", "price", "debit", "credit", "charge", "total",
                        "txn_amt", "withdrawal", "deposit"]
        desc_keys    = ["desc", "merchant", "payee", "description", "vendor", "name",
                        "memo", "details", "transaction"]
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

        cols        = list(df.columns)
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
    """
    Agent 4 – AI-powered knowledge agent.
    - Auto-critique: flags risks in the financial data.
    - Q&A chat: answers any finance question using Gemini.
    """

    def critique(self, plan_text, df):
        if df is None:
            return "No data available for risk analysis."

        risks = []

        if "Account" in df.columns:
            total_exp = df.loc[df["Account"].str.lower() == "expenses", "Amount"].sum()
            income    = df.loc[df["Account"].str.lower() == "income",   "Amount"].sum()
        else:
            total_exp = df.loc[df["Amount"] < 0, "Amount"].sum()
            income    = df.loc[df["Amount"] > 0, "Amount"].sum()

        total_exp = abs(total_exp)
        net = income - total_exp

        if total_exp > 20000:
            risks.append("⚠️ **High expenses detected** – review your biggest categories.")
        if income == 0:
            risks.append("⚠️ **No income rows found** – make sure your statement includes deposits.")
        if net < 0:
            risks.append(f"⚠️ **Negative net (${net:,.2f})** – you are spending more than you earn!")

        feedback = "### 🧐 Critic Review\n\n"
        feedback += "\n".join(risks) if risks else "✅ Plan looks solid! No major risks detected."
        return feedback.strip()

    def ask_ai(self, question, df, api_key):
        if not api_key:
            return "❌ Please add your Gemini API key in the sidebar to use the AI Critic."

        if df is not None and not df.empty:
            if "Account" in df.columns:
                income   = df.loc[df["Account"].str.lower() == "income",   "Amount"].sum()
                expenses = df.loc[df["Account"].str.lower() == "expenses", "Amount"].sum()
            else:
                income   = df.loc[df["Amount"] > 0, "Amount"].sum()
                expenses = df.loc[df["Amount"] < 0, "Amount"].sum()

            expenses_abs = abs(expenses)
            net = income - expenses_abs
            top_desc = (
                df[df["Amount"] < 0]
                .nlargest(5, "Amount", keep="all")["Description"]
                .tolist()
            ) if "Description" in df.columns else []

            data_context = (
                f"The user's bank statement shows:\n"
                f"- Total income / deposits: ${income:,.2f}\n"
                f"- Total expenses / withdrawals: ${expenses_abs:,.2f}\n"
                f"- Net balance change: ${net:,.2f}\n"
                f"- Top spending transactions: {', '.join(str(d) for d in top_desc)}\n"
            )
        else:
            data_context = "No bank statement has been uploaded yet."

        system_prompt = (
            "You are an expert personal finance advisor and AI critic agent embedded in a finance app. "
            "You have access to the user's real bank statement data shown below. "
            "Answer the user's question clearly, helpfully, and in plain English. "
            "Give specific, actionable advice based on their actual numbers when possible.\n\n"
            f"{data_context}"
        )

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{system_prompt}\n\nUser question: {question}")
            return response.text
        except Exception as e:
            return f"❌ Gemini error: {str(e)}"


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
    "Upload a PDF or CSV bank statement (Chase, BofA, Wells Fargo, Navy Federal, Citi, Capital One, etc.)",
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

# ============================================================
# AI CRITIC CHAT (Section 4)
# ============================================================

st.header("🧐 4️⃣ Ask the AI Critic")
st.caption(
    "Ask anything about your finances — the AI Critic reads your actual statement data and answers like a personal finance advisor."
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask a finance question... e.g. 'Why are my expenses so high?' or 'How do I save more?'")
if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("🧐 AI Critic is thinking..."):
            answer = critic.ask_ai(
                user_q,
                st.session_state.analyzed_df or st.session_state.df,
                gemini_key
            )
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.caption("CIS 4394 – Multi-Agent Finance App ❤️")
