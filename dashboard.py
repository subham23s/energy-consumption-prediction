import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("energy_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_resources()

# ─────────────────────────────────────────
# Title
# ─────────────────────────────────────────
st.title("⚡ Energy Consumption Prediction")
st.markdown("Adjust the building parameters below to get a **live prediction** of heating load.")
st.markdown("---")

# ─────────────────────────────────────────
# Input Panel (Sidebar)
# ─────────────────────────────────────────
st.sidebar.header("🏢 Building Parameters")
st.sidebar.markdown("Move the sliders to change input values:")

X1 = st.sidebar.slider("X1 — Relative Compactness", 0.62, 0.98, 0.76, step=0.01)
X2 = st.sidebar.slider("X2 — Surface Area (m²)",    514.5, 808.5, 671.7, step=1.0)
X3 = st.sidebar.slider("X3 — Wall Area (m²)",        245.0, 416.5, 318.5, step=1.0)
X4 = st.sidebar.slider("X4 — Roof Area (m²)",        110.25, 220.5, 176.6, step=0.25)
X5 = st.sidebar.slider("X5 — Overall Height (m)",    3.5, 7.0, 5.25, step=0.25)
X6 = st.sidebar.selectbox("X6 — Orientation", [2, 3, 4, 5])
X7 = st.sidebar.slider("X7 — Glazing Area",          0.0, 0.4, 0.1, step=0.1)
X8 = st.sidebar.selectbox("X8 — Glazing Distribution", [0, 1, 2, 3, 4, 5])

# ─────────────────────────────────────────
# Prediction (Live)
# ─────────────────────────────────────────
user_input = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])
input_scaled = scaler.transform(user_input)
prediction = model.predict(input_scaled, verbose=0)[0][0]

# ─────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔥 Predicted Heating Load", f"{prediction:.2f} kWh")

with col2:
    if prediction < 15:
        level = "🟢 Low"
    elif prediction < 30:
        level = "🟡 Medium"
    else:
        level = "🔴 High"
    st.metric("⚠️ Energy Level", level)

with col3:
    st.metric("📐 Building Height", f"{X5} m")

st.markdown("---")

# ─────────────────────────────────────────
# Input Summary Table
# ─────────────────────────────────────────
st.subheader("📋 Current Input Parameters")

input_df = pd.DataFrame({
    "Feature": ["Relative Compactness", "Surface Area", "Wall Area",
                "Roof Area", "Overall Height", "Orientation",
                "Glazing Area", "Glazing Distribution"],
    "Value": [X1, X2, X3, X4, X5, X6, X7, X8]
})
st.dataframe(input_df, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# Live Bar Chart of Inputs
# ─────────────────────────────────────────
st.subheader("📊 Input Feature Values (Normalized View)")
normalized_vals = scaler.transform(user_input)[0]
feature_names = ['X1','X2','X3','X4','X5','X6','X7','X8']

fig, ax = plt.subplots(figsize=(10, 3))
bars = ax.bar(feature_names, normalized_vals, color='steelblue')
ax.set_ylim(0, 1)
ax.set_ylabel("Normalized Value (0-1)")
ax.set_title("Your Input Features After Normalization")
for bar, val in zip(bars, normalized_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=9)
st.pyplot(fig)

# ─────────────────────────────────────────
# Show saved charts if available
# ─────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Model Training Results")

col4, col5 = st.columns(2)
try:
    with col4:
        st.image("loss_curve.png", caption="Training vs Validation Loss")
    with col5:
        st.image("actual_vs_predicted.png", caption="Actual vs Predicted")
except:
    st.info("Run train_model.py first to see training charts here.")

st.markdown("---")

