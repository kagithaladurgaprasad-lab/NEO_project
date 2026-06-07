import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #FF4B4B;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #A0AEC0;
    text-align: center;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. Header Section
# -------------------------------------------------------------------------
st.markdown(
    "<div class='main-title'>☄️ Near-Earth Objects (NEO)</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Asteroid Hazard Assessment & Classification Portal</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------------------------
# 3. Load Model
# -------------------------------------------------------------------------
available_models = [
    f for f in os.listdir(".")
    if f.endswith("_final_model.pkl")
]

if not available_models:
    st.error("No trained model found.")
    st.stop()

selected_model_file = st.sidebar.selectbox(
    "Select Model",
    available_models
)

@st.cache_resource
def load_pipeline(model_path):
    return joblib.load(model_path)

pipeline = load_pipeline(selected_model_file)

# -------------------------------------------------------------------------
# 4. Inputs
# -------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    est_dia_min = st.slider(
        "Minimum Diameter (km)",
        0.001, 5.0, 0.25
    )

    est_dia_max = st.slider(
        "Maximum Diameter (km)",
        0.001, 11.0, 0.55
    )

    abs_mag = st.number_input(
        "Absolute Magnitude",
        value=20.0
    )

with col2:
    rel_vel = st.number_input(
        "Relative Velocity",
        value=45000.0
    )

    miss_dist = st.number_input(
        "Miss Distance",
        value=35000000.0
    )

# -------------------------------------------------------------------------
# 5. Prediction
# -------------------------------------------------------------------------
if st.button("Analyze Threat Level"):

    input_data = pd.DataFrame([{
        "est_diameter_min": est_dia_min,
        "est_diameter_max": est_dia_max,
        "relative_velocity": rel_vel,
        "miss_distance": miss_dist,
        "absolute_magnitude": abs_mag
    }])

    prediction = pipeline.predict(input_data)[0]

    if prediction == 1:
        st.markdown("""
        <div style="padding:20px;border:2px solid red;border-radius:10px;">
            <h3>⚠️ HAZARDOUS THREAT DETECTED</h3>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:20px;border:2px solid green;border-radius:10px;">
            <h3>✅ CLASSIFIED AS SAFE</h3>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. Footer
# -------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

with st.expander("About the Data Metrics"):
    st.write("NEO hazard prediction model.")