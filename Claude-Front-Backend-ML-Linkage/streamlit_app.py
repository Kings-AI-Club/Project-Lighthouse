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

# Lightweight top-level config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINE_FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]
WINE_CLASSES = ['Class 0', 'Class 1', 'Class 2']

# Feature ranges from the Wine dataset (for slider bounds and normalization reference)
FEATURE_RANGES = {
    'alcohol': (0.0, 15.0),
    'malic_acid': (0.0, 6.0),
    'ash': (0.0, 3.5),
    'alcalinity_of_ash': (0.0, 30.0),
    'magnesium': (0.0, 170.0),
    'total_phenols': (0.0, 4.0),
    'flavanoids': (0.0, 5.5),
    'nonflavanoid_phenols': (0.0, 1.0),
    'proanthocyanins': (0.0, 4.0),
    'color_intensity': (0.0, 13.0),
    'hue': (0.0, 2.0),
    'od280/od315_of_diluted_wines': (0.0, 4.5),
    'proline': (0.0, 1700.0)
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'alcohol': 'Alcohol content (%)',
    'malic_acid': 'Malic acid (g/L)',
    'ash': 'Ash content (g/L)',
    'alcalinity_of_ash': 'Alkalinity of ash',
    'magnesium': 'Magnesium (mg/L)',
    'total_phenols': 'Total phenols',
    'flavanoids': 'Flavanoids',
    'nonflavanoid_phenols': 'Non-flavanoid phenols',
    'proanthocyanins': 'Proanthocyanins',
    'color_intensity': 'Color intensity',
    'hue': 'Hue',
    'od280/od315_of_diluted_wines': 'OD280/OD315 ratio',
    'proline': 'Proline (mg/L)'
}


def load_sample_data():
    """Load sample data from sample-data.txt file"""
    try:
        with open('sample-data.txt', 'r') as f:
            lines = f.readlines()
        
        x_data = None
        for i, line in enumerate(lines):
            if line.strip().startswith('X:'):
                data_line_idx = i + 2
                if data_line_idx < len(lines):
                    data_line = lines[data_line_idx].strip()
                else:
                    logger.error("No data line found after X:")
                    return None
                
                if data_line:
                    values = [float(x.strip()) for x in data_line.split(',')]
                    if len(values) != 13:
                        logger.error(f"Sample data has {len(values)} values, expected 13")
                        return None
                    x_data = values
                    break
        
        if x_data is None:
            logger.error("Could not find X: data in sample-data.txt")
            return None
            
        return x_data
        
    except FileNotFoundError:
        logger.warning("sample-data.txt not found")
        return None
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return None


@st.cache_resource
def get_scaler():
    """Load or create a StandardScaler for feature normalization"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import load_wine
        
        # Load wine dataset to fit scaler
        wine_data = load_wine()
        scaler = StandardScaler()
        scaler.fit(wine_data.data)
        logger.info("Scaler fitted on wine dataset")
        return scaler
    except Exception as e:
        logger.warning(f"Could not create scaler: {e}")
        return None


def normalize_features(features: np.ndarray, scaler) -> np.ndarray:
    """Normalize features using the provided scaler"""
    if scaler is None:
        return features
    try:
        return scaler.transform(features)
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
        return features


def load_model() -> Optional[object]:
    """Lazy-load Keras model"""
    try:
        from model import ThreeClassFNN
        import tensorflow as tf
        
        model = ThreeClassFNN()
        model.build((None, 13))
        model.load_weights('model/wine.weights.h5')
        logger.info("Weights loaded successfully")
        
        test_input = np.zeros((1, 13))
        _ = model.predict(test_input, verbose=0)
        logger.info("Model loaded and warmed up successfully.")
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


def predict_with_model(model, input_array: np.ndarray) -> np.ndarray:
    """Predict using the provided model"""
    return model.predict(input_array, verbose=0)


def get_mock_prediction() -> np.ndarray:
    """Return deterministic mock predictions for demo/fallback"""
    return np.array([[0.0, 1.0, 0.0]])  # 100% Class 1


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Wine Classification AI",
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
st.title("🍷 Wine Classification AI")
st.markdown("*Powered by Deep Learning & Scikit-Learn*")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Normalization toggle
    use_normalization = st.toggle(
        "Use Feature Normalization",
        value=True,
        help="Normalize features using StandardScaler fitted on the Wine dataset"
    )
    
    st.divider()
    
    # Model status
    st.subheader("📊 Model Status")
    model = get_model()
    scaler = get_scaler() if use_normalization else None
    
    if model is not None:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model unavailable")
    
    if use_normalization:
        if scaler is not None:
            st.success("✅ Scaler active")
        else:
            st.warning("⚠️ Scaler unavailable")
    
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
        st.write(f"{'✅' if os.path.exists('model.py') else '❌'} model.py")
        st.write(f"{'✅' if os.path.exists('model/wine.weights.h5') else '❌'} wine.weights.h5")
        
        # Show error if model failed
        if model is None and 'model_error' in st.session_state:
            with st.expander("Error Details"):
                st.error(st.session_state['model_error'])
                st.code(st.session_state['model_error_traceback'])
    
    st.divider()
    
    # Info
    with st.expander("ℹ️ About"):
        st.write("""
        This app classifies wines into three categories based on chemical properties.
        
        **Features:**
        - 13 chemical measurements
        - Deep learning classification
        - StandardScaler normalization
        - Real-time predictions
        """)

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔬 Wine Characteristics")
    
    # Quick test buttons
    button_cols = st.columns(3)
    with button_cols[0]:
        if st.button("📋 Load Sample", use_container_width=True):
            sample_data = load_sample_data()
            if sample_data:
                for i, value in enumerate(sample_data):
                    st.session_state[f"feat_{i}"] = float(value)
                st.rerun()
            else:
                st.error("Sample data file not found")
    
    with button_cols[1]:
        if st.button("🔄 Reset All", use_container_width=True):
            for i in range(len(WINE_FEATURES)):
                st.session_state[f"feat_{i}"] = 0.0
            st.rerun()
    
    with button_cols[2]:
        if st.button("🎲 Random", use_container_width=True):
            for i, feature in enumerate(WINE_FEATURES):
                min_val, max_val = FEATURE_RANGES[feature]
                st.session_state[f"feat_{i}"] = float(np.random.uniform(min_val, max_val))
            st.rerun()
    
    st.divider()
    
    # Feature inputs with sliders
    with st.form("wine_form"):
        inputs = []
        
        # Create tabs for organized input
        tab1, tab2, tab3 = st.tabs(["🍇 Composition", "🎨 Color & Taste", "🧪 Advanced"])
        
        with tab1:
            # First set of features
            for i, feature in enumerate(WINE_FEATURES[:5]):
                default_value = st.session_state.get(f"feat_{i}", 0.0)
                min_val, max_val = FEATURE_RANGES[feature]
                
                value = st.slider(
                    FEATURE_DESCRIPTIONS[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_value),
                    step=(max_val - min_val) / 100,
                    format="%.2f"
                )
                inputs.append(value)
        
        with tab2:
            # Second set of features
            for i, feature in enumerate(WINE_FEATURES[5:10], start=5):
                default_value = st.session_state.get(f"feat_{i}", 0.0)
                min_val, max_val = FEATURE_RANGES[feature]
                
                value = st.slider(
                    FEATURE_DESCRIPTIONS[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_value),
                    step=(max_val - min_val) / 100,
                    format="%.2f"
                )
                inputs.append(value)
        
        with tab3:
            # Last set of features
            for i, feature in enumerate(WINE_FEATURES[10:], start=10):
                default_value = st.session_state.get(f"feat_{i}", 0.0)
                min_val, max_val = FEATURE_RANGES[feature]
                
                value = st.slider(
                    FEATURE_DESCRIPTIONS[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_value),
                    step=(max_val - min_val) / 100,
                    format="%.2f"
                )
                inputs.append(value)
        
        submitted = st.form_submit_button("🎯 Classify Wine", use_container_width=True, type="primary")

with col_right:
    st.subheader("📊 Results")
    
    # Prediction results
    if submitted:
        try:
            # Validate inputs
            for i, (feature, value) in enumerate(zip(WINE_FEATURES, inputs)):
                if not np.isfinite(value):
                    st.error(f"Invalid value for {feature}")
                    st.stop()
            
            # Prepare input
            input_array = np.array([inputs], dtype=float)
            logger.info(f"Raw input: {input_array}")
            
            # Normalize if enabled
            if use_normalization and scaler is not None:
                normalized_array = normalize_features(input_array, scaler)
                logger.info(f"Normalized input: {normalized_array}")
                prediction_input = normalized_array
            else:
                prediction_input = input_array
            
            # Make prediction
            if model is None:
                predictions = get_mock_prediction()
                st.warning("⚠️ Using mock predictions")
            else:
                predictions = predict_with_model(model, prediction_input)
                logger.info(f"Predictions: {predictions}")
            
            predictions = np.asarray(predictions)
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
            
            # Get results
            predicted_class_index = int(np.argmax(predictions[0]))
            predicted_class = WINE_CLASSES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            # Display prediction
            st.markdown(f"""
            <div class='prediction-box'>
                <h2>🎯 {predicted_class}</h2>
                <h3>{confidence:.1%} Confidence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability visualization
            st.write("**Class Probabilities:**")
            
            # Bar chart
            import pandas as pd
            prob_data = pd.DataFrame({
                'Class': WINE_CLASSES,
                'Probability': [float(predictions[0][i]) for i in range(len(WINE_CLASSES))]
            })
            st.bar_chart(prob_data.set_index('Class'), color='#667eea', height=200)
            
            # Progress bars
            for i, class_name in enumerate(WINE_CLASSES):
                prob = float(predictions[0][i])
                st.progress(prob, text=f"{class_name}: {prob:.1%}")
            
            # Detailed metrics
            with st.expander("📈 Detailed Analysis"):
                st.json({
                    "Predicted Class": predicted_class,
                    "Confidence": f"{confidence:.4f}",
                    "All Probabilities": {
                        WINE_CLASSES[i]: f"{float(predictions[0][i]):.4f}"
                        for i in range(len(WINE_CLASSES))
                    },
                    "Normalized": use_normalization
                })
                
                # Feature summary
                st.write("**Input Features:**")
                st.dataframe({
                    "Feature": [FEATURE_DESCRIPTIONS[f] for f in WINE_FEATURES],
                    "Value": [f"{v:.2f}" for v in inputs]
                }, use_container_width=True)
        
        except Exception as e:
            logger.exception("Prediction error")
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 Enter wine characteristics and click **Classify Wine** to get predictions")
        
        # Show example
        st.write("**Example Values:**")
        st.code("""
Alcohol: 13.2%
Malic Acid: 2.3 g/L
Ash: 2.4 g/L
...
        """)