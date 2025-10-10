import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # already set
    pass

import streamlit as st
import numpy as np
import logging
from typing import Optional, Tuple, Dict
import os
import pickle

# Lightweight top-level config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature definitions for homelessness risk model
# Order: Gender, Age, Drug, Mental, Indigenous, DV, ACT, NSW, NT, QLD, SA, TAS, VIC, WA, SHS_Client
FEATURE_NAMES = [
    'Gender', 'Age', 'Drug', 'Mental', 'Indigenous', 'DV',
    'ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA',
    'SHS_Client'
]

BINARY_FEATURES = ['Gender', 'Drug', 'Mental', 'Indigenous', 'DV', 'SHS_Client']
LOCATION_FEATURES = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'Gender': 'Gender (0: Female, 1: Male)',
    'Age': 'Age in years',
    'Drug': 'Drug use risk factor',
    'Mental': 'Mental health risk factor',
    'Indigenous': 'Indigenous status',
    'DV': 'Domestic violence risk factor',
    'Location': 'Australian state/territory',
    'SHS_Client': 'Specialist Homelessness Services client'
}


@st.cache_resource
def get_scaler():
    """Load the age scaler from pickle file"""
    try:
        scaler_path = 'model/age_scaler.pkl'
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler file not found at {scaler_path}")
            return None

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Age scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")
        return None


def normalize_features(feature_dict: dict, scaler) -> np.ndarray:
    """
    Normalize Age feature using the scaler and create feature array.
    Returns array in correct order: Gender, Age, Drug, Mental, Indigenous, DV,
    ACT, NSW, NT, QLD, SA, TAS, VIC, WA, SHS_Client
    """
    # Create feature array in correct order
    features = []

    # Gender
    features.append(feature_dict['Gender'])

    # Age (normalize if scaler available)
    age = feature_dict['Age']
    if scaler is not None:
        try:
            age_scaled = scaler.transform([[age]])[0][0]
            features.append(age_scaled)
        except Exception as e:
            logger.warning(f"Age normalization failed: {e}, using raw value")
            features.append(age)
    else:
        features.append(age)

    # Binary risk factors
    features.append(feature_dict['Drug'])
    features.append(feature_dict['Mental'])
    features.append(feature_dict['Indigenous'])
    features.append(feature_dict['DV'])

    # Location (one-hot encoded)
    selected_location = feature_dict['Location']
    for loc in LOCATION_FEATURES:
        features.append(1 if loc == selected_location else 0)

    # SHS_Client
    features.append(feature_dict['SHS_Client'])

    return np.array([features], dtype=float)


def load_model() -> Optional[object]:
    """Lazy-load Keras model for homelessness risk prediction"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        import h5py

        # Try .keras file first (new format), then .h5 (old format)
        model_paths = [
            'model/homelessness_risk_model.keras',
            'model/homelessness_risk_model.h5'
        ]

        model = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # Check if it's actually an HDF5 file
                    with h5py.File(model_path, 'r') as f:
                        # If we can open it as HDF5, load with legacy method
                        logger.info(f"Detected HDF5 format model at {model_path}")
                        model = keras.models.load_model(model_path, compile=True)
                        logger.info(f"Model loaded successfully from {model_path}")
                        break
                except OSError:
                    # Not an HDF5 file, try as new Keras format
                    try:
                        model = keras.models.load_model(model_path)
                        logger.info(f"Model loaded successfully from {model_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load {model_path}: {e}")
                        continue

        if model is None:
            logger.error("Could not find or load model file")
            return None

        # Warm up model
        test_input = np.zeros((1, 15))
        _ = model.predict(test_input, verbose=0)
        logger.info("Model warmed up successfully")
        return model
    except Exception as exc:
        logger.error(f"Error loading model: {exc}", exc_info=True)
        logger.warning("Running in demo mode - predictions will use mock data")
        st.session_state['model_error'] = str(exc)
        st.session_state['model_error_traceback'] = __import__('traceback').format_exc()
        return None


@st.cache_resource
def get_model():
    """Returns cached model instance or None."""
    return load_model()


def predict_with_model(model, input_array: np.ndarray) -> float:
    """Predict using the provided model. Returns probability of homelessness."""
    prediction = model.predict(input_array, verbose=0)
    return float(prediction[0][0])


def get_mock_prediction() -> float:
    """Return deterministic mock predictions for demo/fallback"""
    return 0.5  # 50% probability


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Homelessness Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-box-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🏠 Homelessness Risk Prediction")
st.markdown("*Powered by TensorFlow & Scikit-Learn*")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Model status
    st.subheader("📊 Model Status")
    model = get_model()
    scaler = get_scaler()

    if model is not None:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model unavailable")

    if scaler is not None:
        st.success("✅ Age scaler loaded")
    else:
        st.warning("⚠️ Age scaler unavailable")

    st.divider()

    # Developer tools
    with st.expander("🔧 Developer Tools"):
        if st.button("Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.code(f"Python {__import__('sys').version.split()[0]}")

        # File system check
        st.write("**Files:**")
        st.write(f"📁 CWD: `{os.getcwd()}`")
        st.write(f"{'✅' if os.path.exists('model/homelessness_risk_model.h5') else '❌'} homelessness_risk_model.h5")
        st.write(f"{'✅' if os.path.exists('model/homelessness_risk_model.keras') else '❌'} homelessness_risk_model.keras")
        st.write(f"{'✅' if os.path.exists('model/age_scaler.pkl') else '❌'} age_scaler.pkl")

        # Show error if model failed
        if model is None and 'model_error' in st.session_state:
            with st.expander("Error Details"):
                st.error(st.session_state['model_error'])
                st.code(st.session_state['model_error_traceback'])

    st.divider()

    # Info
    with st.expander("ℹ️ About"):
        st.write("""
        This app predicts homelessness risk based on demographic and risk factors.

        **Features:**
        - Gender, Age
        - Risk factors: Drug use, Mental health, Indigenous status, Domestic violence
        - Location (Australian state/territory)
        - SHS client status
        - Binary classification (at-risk vs not at-risk)
        - Age normalization using StandardScaler
        """)

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("👤 Individual Information")

    # Feature inputs with form
    with st.form("prediction_form"):
        feature_dict = {}

        # Create tabs for organized input
        tab1, tab2 = st.tabs(["📋 Demographics", "⚠️ Risk Factors"])

        with tab1:
            st.write("**Basic Information**")

            # Gender
            gender_option = st.radio(
                "Gender",
                options=["Female", "Male"],
                horizontal=True,
                help="Select gender"
            )
            feature_dict['Gender'] = 1 if gender_option == "Male" else 0

            # Age
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
                step=1,
                help="Enter age in years"
            )
            feature_dict['Age'] = age

            st.divider()

            # Location
            st.write("**Location**")
            location = st.selectbox(
                "Australian State/Territory",
                options=LOCATION_FEATURES,
                index=1,  # Default to NSW
                help="Select the state or territory"
            )
            feature_dict['Location'] = location

        with tab2:
            st.write("**Risk Factors**")
            st.caption("Select all that apply")

            # Binary risk factors
            feature_dict['Drug'] = 1 if st.checkbox(
                "Drug Use",
                help="Drug use risk factor"
            ) else 0

            feature_dict['Mental'] = 1 if st.checkbox(
                "Mental Health Issues",
                help="Mental health risk factor"
            ) else 0

            feature_dict['Indigenous'] = 1 if st.checkbox(
                "Indigenous",
                help="Indigenous status"
            ) else 0

            feature_dict['DV'] = 1 if st.checkbox(
                "Domestic Violence",
                help="Domestic violence risk factor"
            ) else 0

            st.divider()

            st.write("**Service Information**")
            feature_dict['SHS_Client'] = 1 if st.checkbox(
                "SHS Client",
                help="Currently receiving Specialist Homelessness Services"
            ) else 0

        submitted = st.form_submit_button("🎯 Predict Risk", use_container_width=True, type="primary")

with col_right:
    st.subheader("📊 Results")

    # Prediction results
    if submitted:
        try:
            # Prepare input array from feature dictionary
            input_array = normalize_features(feature_dict, scaler)
            logger.info(f"Feature input array shape: {input_array.shape}")
            logger.info(f"Feature values: {input_array}")

            # Make prediction
            if model is None:
                risk_probability = get_mock_prediction()
                st.warning("⚠️ Using mock predictions (model unavailable)")
            else:
                risk_probability = predict_with_model(model, input_array)
                logger.info(f"Risk probability: {risk_probability}")

            # Determine risk level
            risk_percentage = risk_probability * 100
            is_high_risk = risk_probability > 0.5

            # Display prediction with color coding
            if risk_probability < 0.3:
                box_class = "prediction-box-low"
                risk_label = "Low Risk"
                icon = "✅"
            elif risk_probability < 0.7:
                box_class = "prediction-box"
                risk_label = "Moderate Risk"
                icon = "⚠️"
            else:
                box_class = "prediction-box-high"
                risk_label = "High Risk"
                icon = "🚨"

            st.markdown(f"""
            <div class='{box_class}'>
                <h2>{icon} {risk_label}</h2>
                <h3>{risk_percentage:.1f}% Risk Probability</h3>
            </div>
            """, unsafe_allow_html=True)

            # Risk visualization
            st.write("**Risk Assessment:**")

            # Progress bar for risk
            st.progress(risk_probability, text=f"Homelessness Risk: {risk_percentage:.1f}%")

            # Risk meter visualization
            import pandas as pd
            risk_data = pd.DataFrame({
                'Category': ['Not At Risk', 'At Risk'],
                'Probability': [1 - risk_probability, risk_probability]
            })
            st.bar_chart(risk_data.set_index('Category'), color='#667eea', height=200)

            # Interpretation
            st.write("**Interpretation:**")
            if risk_probability < 0.3:
                st.success("This individual shows a low probability of homelessness risk based on the provided factors.")
            elif risk_probability < 0.7:
                st.warning("This individual shows a moderate probability of homelessness risk. Consider preventive interventions.")
            else:
                st.error("This individual shows a high probability of homelessness risk. Immediate support services may be needed.")

            # Detailed analysis
            with st.expander("📈 Detailed Analysis"):
                st.json({
                    "Risk Level": risk_label,
                    "Risk Probability": f"{risk_probability:.4f}",
                    "Risk Percentage": f"{risk_percentage:.2f}%",
                    "Classification": "At Risk" if is_high_risk else "Not At Risk"
                })

                # Feature summary
                st.write("**Input Features:**")
                feature_summary = {
                    "Gender": "Male" if feature_dict['Gender'] == 1 else "Female",
                    "Age": feature_dict['Age'],
                    "Location": feature_dict['Location'],
                    "Drug Use": "Yes" if feature_dict['Drug'] == 1 else "No",
                    "Mental Health": "Yes" if feature_dict['Mental'] == 1 else "No",
                    "Indigenous": "Yes" if feature_dict['Indigenous'] == 1 else "No",
                    "Domestic Violence": "Yes" if feature_dict['DV'] == 1 else "No",
                    "SHS Client": "Yes" if feature_dict['SHS_Client'] == 1 else "No"
                }
                st.json(feature_summary)

        except Exception as e:
            logger.exception("Prediction error")
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    else:
        st.info("👈 Enter individual information and click **Predict Risk** to get results")

        # Show example
        st.write("**Example:**")
        st.code("""
Demographics:
- Age: 30
- Gender: Female
- Location: NSW

Risk Factors:
- Mental Health Issues
- SHS Client
        """)