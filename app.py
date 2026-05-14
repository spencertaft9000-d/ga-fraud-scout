import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="GA Fraud Scout", layout="wide")
st.title("🕵️ GA Fraud Scout: Georgia Public Data Fraud Detector")
st.markdown("**Lightweight Version** — Rule-based only (for deployment testing)")

st.sidebar.header("📥 Data Source")
st.sidebar.info("Upload Open Georgia Payments/Salary data")

uploaded_file = st.file_uploader("Upload TXT/CSV", type=["txt", "csv"])

if st.button("Use Demo Data"):
    np.random.seed(42)
    data = {
        'Vendor_Name': ['ABC Construction', 'XYZ Consulting'] * 50,
        'Amount': np.random.normal(50000, 20000, 100).tolist(),
        'Agency': ['DOT'] * 50 + ['DOAS'] * 50,
    }
    df = pd.DataFrame(data)
    df.loc[0:5, 'Amount'] = [250000, 300000, 1, 999999, 500000, 450000]
    st.success("Demo loaded")
else:
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep='\t' if uploaded_file.name.endswith('.txt') else ',')
            st.success(f"Loaded {len(df)} records")
        except Exception as e:
            st.error(f"Load error: {e}")
            df = None
    else:
        df = None

if df is not None:
    st.subheader("Raw Preview")
    st.dataframe(df.head(10))

    if 'Amount' in df.columns and 'Vendor_Name' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount'])

        st.subheader("🚩 Red Flags")
        high = df[df['Amount'] > df['Amount'].quantile(0.95)]
        st.write(f"High Value ({len(high)})")
        st.dataframe(high[['Vendor_Name', 'Amount']].head(10))

        duplicates = df[df.duplicated(['Vendor_Name', 'Amount'], keep=False)]
        st.write(f"Duplicates ({len(duplicates)})")
        st.dataframe(duplicates[['Vendor_Name', 'Amount']].head(10))

        csv = df.to_csv(index=False)
        st.download_button("Download Results", csv, "ga_flags.csv")

st.caption("✅ Basic version deployed. Once this works, we can add ML + charts.")