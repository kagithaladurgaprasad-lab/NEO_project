import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="centered"
)

# -------------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #FF4B4B;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #A0AEC0;
    text-align: center;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.markdown(
    "<div class='main-title'>☄️ Near-Earth Objects (NEO)</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Asteroid Hazard Assessment</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------------------------
# Load Model
# -------------------------------------------------------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found in project folder")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.sidebar.success("✅ Loaded: model.pkl")

# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------
st.subheader("Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:
    est_diameter_min = st.number_input(
        "Estimated Diameter Min",
        min_value=0.0,
        value=0.25
    )

    est_diameter_max = st.number_input(
        "Estimated Diameter Max",
        min_value=0.0,
        value=0.55
    )

    absolute_magnitude = st.number_input(
        "Absolute Magnitude",
        value=20.0
    )

with col2:
    relative_velocity = st.number_input(
        "Relative Velocity",
        min_value=0.0,
        value=45000.0
    )

    miss_distance = st.number_input(
        "Miss Distance",
        min_value=0.0,
        value=35000000.0
    )

# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------
if st.button("Analyze Threat Level"):

    input_df = pd.DataFrame({
        "est_diameter_min": [est_diameter_min],
        "est_diameter_max": [est_diameter_max],
        "relative_velocity": [relative_velocity],
        "miss_distance": [miss_distance],
        "absolute_magnitude": [absolute_magnitude]
    })

    try:
        prediction = pipeline.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Hazardous Asteroid Detected")
        else:
            st.success("✅ Asteroid Classified as Safe")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------
with st.expander("About"):
    st.write(
        "This application predicts whether a Near-Earth Object "
        "is hazardous using a trained machine learning model."
    )
