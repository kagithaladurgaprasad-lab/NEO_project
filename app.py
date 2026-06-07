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
# 4. Interactive Input Dashboard (Fully Manual & Adjustable)
# -------------------------------------------------------------------------
st.markdown("### 🛰️ Telemetry Input Matrix")

tab1, tab2 = st.tabs(["📏 Physical Properties", "🌌 Orbital Dynamics"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        est_dia_min = st.slider(
            "Minimum Estimated Diameter (km)",
            min_value=0.001, max_value=5.0, value=0.25, step=0.001,
            format="%.3f"
        )
        est_dia_max = st.slider(
            "Maximum Estimated Diameter (km)",
            min_value=0.001, max_value=11.0, value=0.55, step=0.001,
            format="%.3f"
        )
    with col2:
        # Changed to a slider for quick, manual tuning adjustments
        abs_mag = st.slider(
            "Absolute Magnitude (H)",
            min_value=10.0, max_value=35.0, value=20.0, step=0.1,
            help="The visual magnitude an object would have if it were 1 AU from both the Sun and Earth. Lower means brighter/larger."
        )

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        # Added explicit min/max boundaries and active stepping so it's easily adjustable
        rel_vel = st.number_input(
            "Relative Velocity (km/h)",
            min_value=1000.0, max_value=200000.0, value=45000.0, step=500.0, 
            format="%.1f",
            help="Adjust manually using the '+' or '-' buttons, or type your custom speed value directly."
        )
    with col4:
        # Added explicit min/max boundaries and active stepping so it's easily adjustable
        miss_dist = st.number_input(
            "Miss Distance (km)",
            min_value=100000.0, max_value=80000000.0, value=35000000.0, step=50000.0, 
            format="%.1f",
            help="Distance by which the object misses Earth. Adjust using buttons or type manually."
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
