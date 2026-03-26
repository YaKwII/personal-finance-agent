import streamlit as st
import pandas as pd

# Your agents (fixed for local)
class DataFetcherAgent:
    def fetch_data(self, uploaded_file=None):
        if uploaded_file:
            return pd.read_csv(uploaded_file)
        mock_data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Description': ['Coffee', 'Groceries', 'Rent'],
            'Amount': [-5.50, -75.20, -1200.00]
        }
        return pd.DataFrame(mock_data)

class AnalyzerAgent:
    def analyze(self, df):
        if df is None: return "Upload data first!", None
        df = df.copy()
        df['Category'] = df['Description'].str.lower().map({
            'coffee': 'Food', 'groceries': 'Food', 'rent': 'Housing'
        }).fillna('Other')
        spending = df.groupby('Category')['Amount'].sum().abs().sort_values(ascending=False)
        summary = f"💳 Total: ${df['Amount'].sum().abs():.2f}\n" + '\n'.join([f"• {c}: ${a:.2f}" for c,a in spending.items()])
        return summary, df

class PlannerAgent:
    def plan(self, df):
        if df is None: return "Analyze first!"
        top_spend = df.groupby('Category')['Amount'].sum().abs().idxmax()
        return f"📊 Plan: Cut '{top_spend}' by 20% monthly. Save $200!"

# App
st.set_page_config(page_title="My Finance Agent", layout="wide")
st.title("💰 My Personal Finance Agent")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = None

fetcher = DataFetcherAgent()
analyzer = AnalyzerAgent()
planner = PlannerAgent()

uploaded = st.file_uploader("📁 Upload CSV", type='csv')
if uploaded:
    st.session_state.data = fetcher.fetch_data(uploaded)
    st.success("✅ Data loaded!")

cols = st.columns([1,1,1])
if cols[0].button("🔍 Analyze", use_container_width=True) and st.session_state.data is not None:
    summary, df = analyzer.analyze(st.session_state.data)
    st.session_state.analyzed = df
    st.balloons()
    st.text(summary)

if cols[1].button("📈 Plan", use_container_width=True) and st.session_state.analyzed is not None:
    plan = planner.plan(st.session_state.analyzed)
    st.success(plan)

if cols[2].button("🔄 Reset", use_container_width=True):
    for key in st.session_state.keys():
        del st.session_state[key]

if st.session_state.data is not None:
    st.subheader("📋 Data")
    st.dataframe(st.session_state.data)

st.caption("Made in VSCode ❤️")