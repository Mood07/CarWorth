import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from predict import load_artifacts, build_input, predict_price

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarWorth — Used Car Price Estimator",
    page_icon="🚗",
    layout="wide",
)

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    model, encoders, explainer, features, metrics = get_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 CarWorth")
    st.caption("USA Used Car Price Estimator")
    st.divider()

    if model_loaded:
        st.success("Model loaded")
        col1, col2 = st.columns(2)
        col1.metric("R²",   f"{metrics.get('r2', 0):.3f}")
        col2.metric("MAE",  f"${metrics.get('mae', 0):,.0f}")
        st.caption(f"RMSE: ${metrics.get('rmse', 0):,.0f} | MAPE: {metrics.get('mape', 0):.1f}%")
    else:
        st.warning("Model not found. Run the training notebook first.")

    st.divider()
    st.markdown("""
**How it works:**
1. Fill in car details
2. Click **Estimate Price**
3. See prediction + SHAP explanation

**Model:** XGBoost
**Data:** ~300K USA listings (Craigslist)
""")
    st.divider()
    st.caption("Built by Berke Türk | [GitHub](https://github.com/Mood07/CarWorth)")

# ── Option lists ──────────────────────────────────────────────────────────────
MANUFACTURERS = sorted([
    'acura','audi','bmw','buick','cadillac','chevrolet','chrysler',
    'dodge','ferrari','ford','gmc','honda','hyundai','infiniti',
    'jaguar','jeep','kia','land rover','lexus','lincoln','mazda',
    'mercedes-benz','mitsubishi','nissan','pontiac','porsche',
    'ram','rover','saturn','subaru','tesla','toyota','volkswagen','volvo'
])
FUELS         = ['gas', 'diesel', 'hybrid', 'electric', 'other']
TRANSMISSIONS = ['automatic', 'manual', 'other']
DRIVES        = ['fwd', 'rwd', '4wd']
TYPES         = ['sedan', 'suv', 'pickup', 'truck', 'coupe', 'hatchback', 'van', 'minivan', 'convertible', 'wagon', 'offroad', 'bus', 'other']
CONDITIONS    = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
CYLINDERS     = ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders', 'other']
TITLE_STATUS  = ['clean', 'rebuilt', 'lien', 'salvage', 'missing', 'parts only']
STATES        = sorted(['al','ak','az','ar','ca','co','ct','de','fl','ga','hi','id','il','in','ia','ks','ky','la','me','md','ma','mi','mn','ms','mo','mt','ne','nv','nh','nj','nm','ny','nc','nd','oh','ok','or','pa','ri','sc','sd','tn','tx','ut','vt','va','wa','wv','wi','wy'])

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🚗 CarWorth — Used Car Price Estimator")
st.markdown("Enter your car details below to get an AI-powered price estimate with explanation.")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("car_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basic Info")
        manufacturer = st.selectbox("Manufacturer", MANUFACTURERS, index=MANUFACTURERS.index('toyota'))
        year         = st.slider("Year", 1990, 2024, 2018)
        odometer     = st.number_input("Odometer (miles)", min_value=0, max_value=350_000, value=60_000, step=1000)
        condition    = st.selectbox("Condition", CONDITIONS, index=3)

    with col2:
        st.subheader("Specs")
        fuel         = st.selectbox("Fuel", FUELS)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        drive        = st.selectbox("Drive", DRIVES)
        vehicle_type = st.selectbox("Type", TYPES, index=0)
        cylinders    = st.selectbox("Cylinders", CYLINDERS, index=3)

    with col3:
        st.subheader("Details")
        title_status = st.selectbox("Title Status", TITLE_STATUS)
        state        = st.selectbox("State", STATES, index=STATES.index('ca'))
        paint_color  = st.selectbox("Color", ['white','black','silver','blue','red','grey','green','brown','yellow','orange','purple','custom'])

    submitted = st.form_submit_button("🔍 Estimate Price", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    if not model_loaded:
        st.error("Model not loaded. Please run the training notebook first.")
        st.stop()

    user_input = {
        'manufacturer': manufacturer,
        'year':         year,
        'odometer':     odometer,
        'condition':    condition,
        'fuel':         fuel,
        'transmission': transmission,
        'drive':        drive,
        'type':         vehicle_type,
        'cylinders':    cylinders,
        'title_status': title_status,
        'state':        state,
    }

    df_input = build_input(user_input, encoders, features)
    result   = predict_price(model, df_input)

    st.divider()

    # Price display
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            f"""
            <div style='text-align:center; padding: 24px; background: #f0f4ff;
                        border-radius: 16px; border: 2px solid #4f8ef7;'>
                <p style='color:#555; font-size:16px; margin:0;'>Estimated Market Value</p>
                <h1 style='color:#1a3a8f; font-size:52px; margin:8px 0;'>
                    ${result['price']:,.0f}
                </h1>
                <p style='color:#777; font-size:14px; margin:0;'>
                    Range: <b>${result['low']:,.0f}</b> — <b>${result['high']:,.0f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # SHAP explanation
    st.subheader("Why this price? — SHAP Explanation")

    shap_values_single = explainer.shap_values(df_input)

    # Decode categorical values back to original strings for display
    display_data = []
    for feat in features:
        val = df_input[feat].values[0]
        if feat in encoders:
            try:
                val = encoders[feat].inverse_transform([int(val)])[0]
            except Exception:
                pass
        display_data.append(val)

    explanation = shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=display_data,
        feature_names=features
    )

    fig = plt.figure(figsize=(9, 5))
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.title(f"SHAP Waterfall — Prediction: ${result['price']:,.0f}", fontsize=11)
    plt.tight_layout()
    col_shap1, col_shap2, col_shap3 = st.columns([1, 3, 1])
    with col_shap2:
        st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "Each bar shows how much a feature pushed the price **up** (red) or **down** (blue) "
        "from the baseline average price. The baseline is the model's average prediction across all cars."
    )

    # Feature contribution table
    with st.expander("See raw SHAP values"):
        shap_df = pd.DataFrame({
            'Feature':     features,
            'Value':       df_input.values[0],
            'SHAP Impact': shap_values_single[0]
        }).sort_values('SHAP Impact', key=abs, ascending=False)
        shap_df['Direction'] = shap_df['SHAP Impact'].apply(lambda x: '↑ Increases price' if x > 0 else '↓ Decreases price')
        st.dataframe(shap_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("CarWorth uses XGBoost trained on ~300K USA used car listings from Craigslist. Predictions are estimates only.")
