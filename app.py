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

# Custom CSS for an ultra-modern, dark space theme with glowing elements
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 0 0 10px rgba(255,75,75,0.3);
    }
    .subtitle {
        font-size: 18px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. Header Section
# -------------------------------------------------------------------------
st.markdown("<div class='main-title'>☄️ Near-Earth Objects (NEO)</div>", unsafe_html=True)
st.markdown("<div class='subtitle'>AI-Powered Asteroid Hazard Assessment & Classification Portal</div>", unsafe_html=True)
st.markdown("---")

# -------------------------------------------------------------------------
# 3. Model Loading Logic
# -------------------------------------------------------------------------
# Automatically find the first available .pkl model in your directory
available_models = [f for f in os.listdir('.') if f.endswith('_final_model.pkl')]

if not available_models:
    st.error("🚨 No trained model found! Please run your training script first to generate a `_final_model.pkl` file.")
    st.stop()

# Let user pick which trained model architecture to use from the sidebar
st.sidebar.header("⚙️ Configuration")
selected_model_file = st.sidebar.selectbox("Select Trained Model Architecture", available_models)

@st.cache_resource
def load_pipeline(model_path):
    return joblib.load(model_path)

pipeline = load_pipeline(selected_model_file)
st.sidebar.success(f"Loaded: {selected_model_file.split('_')[0]}")

# -------------------------------------------------------------------------
# 4. User Inputs Layout (Organized into Visual Cards/Columns)
# -------------------------------------------------------------------------
st.subheader("📋 Asteroid Metrics & Physical Characteristics")
st.write("Modify the sliders and values below based on telemetry observation data:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📐 Estimated Proportions**")
    est_dia_min = st.slider("Minimum Diameter (km)", 0.001, 5.0, 0.25, step=0.001, help="The estimated minimum physical size of the asteroid.")
    # Derive a realistic maximum based on standard ratios or let them explicitly choose
    est_dia_max = st.slider("Maximum Diameter (km)", 0.001, 11.0, 0.55, step=0.001, help="The estimated maximum physical size of the asteroid.")
    
    st.markdown("<br>", unsafe_html=True)
    st.markdown("**✨ Visual Properties**")
    abs_mag = st.number_input("Absolute Magnitude (H)", min_value=5.0, max_value=35.0, value=20.0, step=0.1,
                              help="The intrinsic brightness of the celestial object. Lower values indicate a larger, brighter object.")

with col2:
    st.markdown("**🚀 Orbital Dynamics**")
    rel_vel = st.number_input("Relative Velocity (km/h)", min_value=100.0, max_value=150000.0, value=45000.0, step=500.0,
                               help="The speed of the asteroid relative to Earth.")
    miss_dist = st.number_input("Miss Distance (km)", min_value=10000.0, max_value=80000000.0, value=35000000.0, step=50000.0,
                                help="The distance by which the asteroid misses Earth's orbit trajectory.")

# -------------------------------------------------------------------------
# 5. Prediction Execution
# -------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔥 Analyze Threat Level", use_container_width=True):
    
    # Format the explicit feature matrix exact to how the models were trained
    input_data = pd.DataFrame([{
        'est_diameter_min': est_dia_min,
        'est_diameter_max': est_dia_max,
        'relative_velocity': rel_vel,
        'miss_distance': miss_dist,
        'absolute_magnitude': abs_mag
    }])
    
    # Run predictions safely through your pipeline (handles Scaling automatically!)
    prediction = pipeline.predict(input_data)[0]
    
    # Display results with high-contrast, visually pleasing alerts
    st.markdown("### 📊 Assessment Summary")
    
    if prediction == 1:
        st.markdown("""
            <div style="background-color: rgba(239, 68, 68, 0.2); border: 2px solid #EF4444; padding: 25px; border-radius: 10px;">
                <h3 style="color: #EF4444; margin-top:0;">⚠️ HAZARDOUS THREAT DETECTED</h3>
                <p style="color: #FCA5A5; font-size: 16px; margin-bottom:0;">
                    <strong>Alert:</strong> Based on the geometric configuration, velocity, and close proximity thresholds, 
                    this object is classified as a <strong>Potentially Hazardous Asteroid (PHA)</strong>. Continuous telemetry monitoring is highly recommended.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: rgba(34, 197, 94, 0.2); border: 2px solid #22C55E; padding: 25px; border-radius: 10px;">
                <h3 style="color: #22C55E; margin-top:0;">✅ CLASSIFIED AS SAFE</h3>
                <p style="color: #86EFAC; font-size: 16px; margin-bottom:0;">
                    <strong>Status:</strong> Nominal. The asteroid's tracking profile indicates its size, velocity, and distance 
                    pose no immediate risk or atmospheric impact threats to Earth.
                </p>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. Extra UI Elements (Explainer Footer)
# -------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_html=True)
with st.expander("ℹ️ About the Data Metrics"):
    st.markdown("""
    - **Absolute Magnitude:** Measures luminosity. Every decrease of 5 magnitudes implies a tenfold increase in object diameter.
    - **Miss Distance:** Earth-moon distance averages roughly $384,400\text{ km}$. Values near or below this require high priority safety checks.
    - **Pipeline Handling:** The input vectors are automatically transformed by whichever preprocessing method (`StandardScaler` or `MinMaxScaler`) your Optuna optimization trial flagged as ideal.
    """)
