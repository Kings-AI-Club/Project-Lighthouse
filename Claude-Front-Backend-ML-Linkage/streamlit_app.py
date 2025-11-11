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

from focal_loss import BinaryFocalLoss

# Lightweight top-level config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature definitions for homelessness risk model (20 features)
# Order: Gender, Drug, Mental, Indigenous, DV, ACT, NSW, NT, QLD, SA, TAS, VIC, WA, Age one-hot (7)
FEATURE_NAMES = [
    'Gender', 'Age', 'Drug', 'Mental', 'Indigenous', 'DV',
    'ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'
]

BINARY_FEATURES = ['Gender', 'Drug', 'Mental', 'Indigenous', 'DV']
LOCATION_FEATURES = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']

# Age groups for one-hot encoding - MUST match training data
AGE_GROUPS = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'Gender': 'Gender (0: Female, 1: Male)',
    'Age': 'Age group',
    'Drug': 'Drug use risk factor',
    'Mental': 'Mental health risk factor',
    'Indigenous': 'Indigenous status',
    'DV': 'Domestic violence risk factor',
    'Location': 'Australian state/territory'
}


# No scaler needed - all features are binary/one-hot encoded


def create_feature_array(feature_dict: dict) -> np.ndarray:
    """
    Create feature array with one-hot encoding for Age and Location.

    Feature order matches training data:
    - Gender (1 feature)
    - Drug, Mental, Indigenous, DV (4 features)
    - Location one-hot: ACT, NSW, NT, QLD, SA, TAS, VIC, WA (8 features)
    - Age one-hot: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+ (7 features)

    Total: 20 features (all binary)
    """
    features = []

    # Gender (1 feature)
    features.append(feature_dict['Gender'])

    # Binary risk factors (4 features)
    features.append(feature_dict['Drug'])
    features.append(feature_dict['Mental'])
    features.append(feature_dict['Indigenous'])
    features.append(feature_dict['DV'])

    # Location one-hot encoded (8 features)
    selected_location = feature_dict['Location']
    for loc in LOCATION_FEATURES:
        features.append(1 if loc == selected_location else 0)

    # Age one-hot encoded (7 features)
    # Order: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    selected_age_group = feature_dict['Age']
    for age_group in AGE_GROUPS:
        features.append(1 if age_group == selected_age_group else 0)

    logger.info(f"Created feature array with {len(features)} features")
    logger.info(f"Age group '{selected_age_group}' one-hot encoded")

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

        # Warm up model with 20 features (all binary/one-hot encoded)
        test_input = np.zeros((1, 20))
        _ = model.predict(test_input, verbose=0)
        logger.info("Model warmed up successfully (20 features)")
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

    if model is not None:
        st.success("✅ Model loaded")
        st.info("ℹ️ Using one-hot encoding (no scaling required)")
    else:
        st.error("❌ Model unavailable")

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
        - Gender, Age group (one-hot encoded)
        - Risk factors: Drug use, Mental health, Indigenous status, Domestic violence
        - Location (Australian state/territory, one-hot encoded)
        - Binary classification (at-risk vs not at-risk)
        - All features are binary (no scaling required)

        **Model:**
        - 20 input features (all binary/one-hot encoded)
        - Feedforward Neural Network
        - Binary classification output
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

            # Age Group
            st.write("**Age Group**")
            age_group = st.selectbox(
                "Select age group",
                options=AGE_GROUPS,
                index=2,  # Default to 25-34
                help="Age group of the individual (must match training data format)"
            )
            feature_dict['Age'] = age_group

            # Show one-hot encoding info
            st.caption(f"ℹ️ Age group will be one-hot encoded (7 binary features)")

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

            # (SHS Client field removed to match 20-feature model)

        submitted = st.form_submit_button("🎯 Predict Risk", use_container_width=True, type="primary")

with col_right:
    st.subheader("📊 Results")

    # Prediction results
    if submitted:
        try:
            # Prepare input array from feature dictionary
            input_array = create_feature_array(feature_dict)
            logger.info(f"Feature input array shape: {input_array.shape}")
            logger.info(f"Feature values: {input_array}")

            # Make prediction
            if model is None:
                risk_probability = get_mock_prediction()
                st.warning("⚠️ Using mock predictions (model unavailable)")
            else:
                risk_probability = predict_with_model(model, input_array)
                logger.info(f"Confidence Score: {risk_probability}")

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
                <h3>{risk_percentage:.1f} Confidence Score</h3>
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
                    "Confidence Score": f"{risk_probability:.4f}",
                    "Risk Percentage": f"{risk_percentage:.2f}%",
                    "Classification": "At Risk" if is_high_risk else "Not At Risk"
                })

                # Feature summary
                st.write("**Input Features:**")
                age_group = feature_dict['Age']
                feature_summary = {
                    "Gender": "Male" if feature_dict['Gender'] == 1 else "Female",
                    "Age Group": age_group,
                    "Location": feature_dict['Location'],
                    "Drug Use": "Yes" if feature_dict['Drug'] == 1 else "No",
                    "Mental Health": "Yes" if feature_dict['Mental'] == 1 else "No",
                    "Indigenous": "Yes" if feature_dict['Indigenous'] == 1 else "No",
                    "Domestic Violence": "Yes" if feature_dict['DV'] == 1 else "No"
                }
                st.json(feature_summary)

                # Show the feature array structure
                st.write("**Feature Array (20 binary features):**")
                st.code(f"Shape: {input_array.shape}\nValues: {input_array[0]}")

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
- Age Group: 25-34
- Gender: Female
- Location: NSW

Risk Factors:
- Mental Health Issues
        """)
