import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Safe import for scikit-learn (helps with deployment errors)
try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    st.error("❌ scikit-learn is not installed. Please make sure 'scikit-learn' is in requirements.txt")
    st.stop()

st.set_page_config(page_title="GA Fraud Scout", layout="wide")
st.title("🕵️ GA Fraud Scout: Georgia Public Data Fraud Detector")
st.markdown("**Prototype v1.3** — Public Version | Educational Tool Only")

st.sidebar.header("📥 Data Source")
st.sidebar.markdown("""
**How to use:**
1. Go to [Open Georgia](https://open.ga.gov/)
2. Download Payments, Obligations, or Salary data
3. Upload the TXT/CSV file below
""")

uploaded_file = st.file_uploader("Upload your Open Georgia file (TXT or CSV)", type=["txt", "csv"])

if st.button("🧪 Use Demo Georgia Sample Data"):
    np.random.seed(42)
    data = {
        'Vendor_Name': ['ABC Construction', 'XYZ Consulting', 'ABC Construction', 'Tech Solutions Inc', 'VendorX LLC'] * 20,
        'Amount': np.random.normal(50000, 20000, 100).tolist(),
        'Agency': ['DOT'] * 50 + ['DOAS'] * 50,
        'Transaction_Date': pd.date_range(start='2024-01-01', periods=100, freq='D').tolist()
    }
    df = pd.DataFrame(data)
    df.loc[0:5, 'Amount'] = [250000, 300000, 1, 999999, 500000, 450000]
    st.success("✅ Demo data loaded with injected anomalies")
else:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} records")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = None
    else:
        df = None

if df is not None and not df.empty:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if 'Amount' in df.columns and 'Vendor_Name' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount']).reset_index(drop=True)

        # Rule-based flags
        st.subheader("🚩 Rule-Based Red Flags")
        col1, col2 = st.columns(2)
        with col1:
            high_value = df[df['Amount'] > df['Amount'].quantile(0.95)]
            st.write(f"**Top 5% High-Value Payments** ({len(high_value)})")
            cols = ['Vendor_Name', 'Amount', 'Agency'] if 'Agency' in df.columns else ['Vendor_Name', 'Amount']
            st.dataframe(high_value[cols].head(10))
        with col2:
            duplicates = df[df.duplicated(subset=['Vendor_Name', 'Amount'], keep=False)]
            st.write(f"**Duplicate Payments** ({len(duplicates)})")
            st.dataframe(duplicates[cols].head(10))

        # ML Anomaly Detection
        st.subheader("🤖 ML Anomaly Detection (15% sensitivity)")
        X = df[['Amount']].values
        model = IsolationForest(contamination=0.15, random_state=42)
        df['Anomaly_Score'] = model.fit_predict(X)
        df['Anomaly_Label'] = df['Anomaly_Score'].map({-1: '🚩 Fraud Flag', 1: '✅ Normal'})

        anomalies = df[df['Anomaly_Score'] == -1]
        st.write(f"**Potential Anomalies Detected: {len(anomalies)}**")

        # Interactive Plot
        st.subheader("📊 Interactive Anomalies")
        hover_cols = ['Vendor_Name', 'Amount', 'Anomaly_Label']
        if 'Agency' in df.columns:
            hover_cols.append('Agency')
        if 'Transaction_Date' in df.columns:
            hover_cols.append('Transaction_Date')

        fig = px.scatter(
            df, x=df.index, y='Amount',
            color='Anomaly_Label',
            hover_data=hover_cols,
            title="Hover over points for details • Red = Potential Fraud Flag",
            height=650
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download All Results with Fraud Flags", csv, "ga_fraud_flags.csv", "text/csv")

    else:
        st.warning("⚠️ Your file must contain at least **Vendor_Name** and **Amount** columns.")

else:
    st.info("👆 Upload a file or use the demo data to begin analysis.")

st.caption("⚠️ This is a prototype for educational and analysis purposes only. It does not prove fraud.")