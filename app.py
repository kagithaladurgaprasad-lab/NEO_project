import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------------------------------
# 1. Page Configuration & Futuristic Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Deep Space & Cyberpunk Accents Custom CSS
st.markdown("""
<style>
    /* Main Title Styling */
    .main-title {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2px;
        letter-spacing: 1px;
    }
    
    .subtitle {
        font-size: 16px;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Elegant Results Styling */
    .threat-banner {
        padding: 24px;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
        color: #ef4444;
        text-align: center;
    }
    
    .safe-banner {
        padding: 24px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.1);
        border: 2px solid #22c55e;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);
        color: #22c55e;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. Header Section
# -------------------------------------------------------------------------
st.markdown("<div class='main-title'>☄️ NEO HAZARD PREDICTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Planetary Defense Intelligence</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. Sidebar & Model Loading
# -------------------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Control Panel")
available_models = [f for f in os.listdir(".") if f.endswith("model.pkl")]

if not available_models:
    st.sidebar.error("❌ No trained model (*model.pkl) found in workspace.")
    st.error("Please place your trained model pipeline file in the directory to proceed.")
    st.stop()

selected_model_file = st.sidebar.selectbox(
    "Select AI Core Model",
    available_models,
    help="Choose the model pipeline instance for active inference."
)

@st.cache_resource
def load_pipeline(model_path):
    return joblib.load(model_path)

try:
    pipeline = load_pipeline(selected_model_file)
    st.sidebar.success("🤖 Model Engine Loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------------------------------------------------------
# 4. Interactive Input Dashboard (All Manual + Step Button Layout)
# -------------------------------------------------------------------------
st.markdown("### 🛰️ Telemetry Input Matrix")

tab1, tab2 = st.tabs(["📏 Physical Properties", "🌌 Orbital Dynamics"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # Changed from slider to dynamic number input box with increments
        est_dia_min = st.number_input(
            "Minimum Estimated Diameter (km)",
            min_value=0.000, max_value=15.000, value=0.250, step=0.005,
            format="%.3f",
            help="Type manually or use +/- to adjust by 0.005 km increments."
        )
        # Changed from slider to dynamic number input box with increments
        est_dia_max = st.number_input(
            "Maximum Estimated Diameter (km)",
            min_value=0.000, max_value=30.000, value=0.550, step=0.005,
            format="%.3f",
            help="Type manually or use +/- to adjust by 0.005 km increments."
        )
    with col2:
        # Changed from slider to dynamic number input box with increments
        abs_mag = st.number_input(
            "Absolute Magnitude (H)",
            min_value=5.0, max_value=40.0, value=20.0, step=0.1,
            format="%.1f",
            help="The visual magnitude an object would have if it were 1 AU from both the Sun and Earth. Adjust by 0.1."
        )

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        rel_vel = st.number_input(
            "Relative Velocity (km/h)",
            min_value=0.0, max_value=300000.0, value=45000.0, step=500.0, 
            format="%.1f",
            help="Adjust manually or use +/- to change by 500.0 km/h."
        )
    with col4:
        miss_dist = st.number_input(
            "Miss Distance (km)",
            min_value=0.0, max_value=150000000.0, value=35000000.0, step=50000.0, 
            format="%.1f",
            help="Distance by which the object misses Earth. Adjust by 50,000 km vectors."
        )

st.markdown("---")

# -------------------------------------------------------------------------
# 5. Summary Telemetry Metrics
# -------------------------------------------------------------------------
with st.container():
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Diameter", f"{(est_dia_min + est_dia_max)/2:.3f} km")
    m2.metric("Velocity", f"{rel_vel:,.0f} km/h")
    m3.metric("Miss Distance", f"{miss_dist:,.0f} km")

st.markdown("<br>", unsafe_allow_html=True)

# Validation check to ensure consistency before launching inference
if est_dia_min > est_dia_max:
    st.warning("⚠️ Telemetry Alert: Minimum diameter should not exceed maximum diameter parameters.")

# -------------------------------------------------------------------------
# 6. Real-time Threat Analysis & Inference
# -------------------------------------------------------------------------
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    analyze_triggered = st.button("🚨 RUN THREAT ANALYSIS EVALUATION", use_container_width=True)

if analyze_triggered:
    input_data = pd.DataFrame([{
        "est_diameter_min": est_dia_min,
        "est_diameter_max": est_dia_max,
        "relative_velocity": rel_vel,
        "miss_distance": miss_dist,
        "absolute_magnitude": abs_mag
    }])

    with st.spinner("Processing trajectory arrays..."):
        try:
            prediction = pipeline.predict(input_data)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                <div class='threat-banner'>
                    <h2 style='margin:0; font-weight:800;'>⚠️ CRITICAL THREAT DETECTED</h2>
                    <p style='margin:10px 0 0 0; font-size:15px; opacity:0.9;'>
                        This Near-Earth Object meets all structural criteria to be classified as a <b>Potentially Hazardous Asteroid (PHA)</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='safe-banner'>
                    <h2 style='margin:0; font-weight:800;'>✅ CLASSIFIED AS SAFE</h2>
                    <p style='margin:10px 0 0 0; font-size:15px; opacity:0.9;'>
                        Object trajectory and physical metrics present no immediate threat to planetary safety vectors.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# -------------------------------------------------------------------------
# 7. Information Architecture / Documentation Footer
# -------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📘 Telemetry Feature Documentation & Methodology"):
    st.markdown("""
    ### Understanding the Metrics
    * **Estimated Diameters:** Calculated values based on the object's absolute magnitude and expected albedo profile.
    * **Absolute Magnitude ($H$):** Measures the intrinsic brightness of the asteroid. Lower values typically mean larger object sizing.
    * **Relative Velocity:** The speed of the celestial body relative to Earth at its closest approach vector.
    * **Miss Distance:** The proximity gap between the center of Earth and the passing object's peak trajectory point.
    """)
