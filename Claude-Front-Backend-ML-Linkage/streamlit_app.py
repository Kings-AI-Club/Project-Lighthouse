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

# Lightweight top-level config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINE_FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]
WINE_CLASSES = ['class_0', 'class_1', 'class_2']

# Example data for testing
# Load from external file
def load_example_data():
    """Load example data from sample-data.txt file"""
    try:
        with open('sample-data.txt', 'r') as f:
            data_str = f.read().strip()
            # Parse comma-separated values
            values = [float(x.strip()) for x in data_str.split(',')]
            if len(values) != 13:
                logger.error(f"Example data has {len(values)} values, expected 13")
                return None
            return values
    except FileNotFoundError:
        logger.warning("sample-data.txt not found")
        return None
    except Exception as e:
        logger.error(f"Error loading example data: {e}")
        return None


def load_model() -> Optional[object]:
    """
    Lazy-load your Keras model. Import heavy libraries inside this function
    to avoid Streamlit watcher threading + native-extension initialization issues.
    Returns the model or None on failure.
    """
    try:
        # Import heavy libs here
        # The 'model' module should provide ThreeClassFNN
        from model import ThreeClassFNN  # local file in same project
        # Import TensorFlow/Keras only now
        import tensorflow as tf  # noqa: F401
        
        model = ThreeClassFNN()
        model.build((None, 13))  # Input shape: (batch_size, 13 features)
        model.load_weights('model/wine.weights.h5')
        logger.info("Weights loaded successfully")
        
        # Verify model architecture by doing a test prediction
        # This ensures the model is fully initialized
        test_input = np.zeros((1, 13))
        _ = model.predict(test_input, verbose=0)
        logger.info(f"Model expects input shape: (None, 13)")
        logger.info(f"Model output shape: (None, 3)")
        logger.info("Model loaded and warmed up successfully.")
        return model
    except Exception as exc:
        logger.error(f"Error loading model: {exc}")
        logger.warning("Running in demo mode - predictions will use mock data")
        return None


@st.cache_resource
def get_model():
    """Returns cached model instance or None."""
    return load_model()


def predict_with_model(model, input_array: np.ndarray) -> np.ndarray:
    """Predict using the provided model, or raise."""
    return model.predict(input_array, verbose=0)


def get_mock_prediction() -> np.ndarray:
    """Return deterministic mock predictions for demo/fallback."""
    # Create realistic-looking mock predictions
    return np.array([[0.2, 0.7, 0.1]])  # Favor class_1


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Wine Classification AI", layout="centered")

st.title("Wine Classification AI (Streamlit)")
st.write("Enter wine characteristics below and click **Predict**.")

# Sidebar / developer tools
with st.sidebar:
    st.header("Developer")
    st.write("Model loader status and debug controls.")
    if st.button("Reload model"):
        # Clear cache then load (workaround: re-run by calling get_model with clear_cache)
        st.cache_resource.clear()
        _ = get_model()
        st.rerun()
    st.write("Python & environment info:")
    st.code({
        "python_version": "".join(map(str, __import__("sys").version.splitlines())),
    })

# Example Data Button
st.subheader("Quick Test")
if st.button("Fill Example Data", use_container_width=True, type="secondary"):
    example_data = load_example_data()
    if example_data:
        for i, value in enumerate(example_data):
            st.session_state[f"feat_{i}"] = float(value)
        st.rerun()
    else:
        st.error("Could not load example data. Create a file named 'sample-data.txt' with 13 comma-separated values.")

st.divider()

# Form inputs (use number_input for numeric validation)
with st.form("wine_form"):
    st.subheader("Wine Characteristics")
    cols = st.columns(2)
    inputs = []
    for i, feature in enumerate(WINE_FEATURES):
        col = cols[i % 2]
        # Get value from session state if available, otherwise default to 0.0
        default_value = st.session_state.get(f"feat_{i}", 0.0)
        
        # Don't use key inside form - it causes state conflicts
        # The form will manage its own state
        value = col.number_input(
            label=feature.replace('_', ' ').capitalize(),
            value=float(default_value),
            format="%.6f",
            step=0.1
        )
        inputs.append(value)
    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

# Show model load status
model = get_model()
if model is None:
    st.warning("⚠️ Model not loaded — running in demo mode (mock predictions).")
else:
    st.success("✅ Model loaded successfully.")

# Perform prediction when submitted
if submitted:
    try:
        # FRONTEND DATA VALIDATION
        # ========================
        # Validate all inputs before processing
        for i, (feature, value) in enumerate(zip(WINE_FEATURES, inputs)):
            try:
                float_value = float(value)
                if not np.isfinite(float_value):
                    raise ValueError(f"Value must be a finite number")
            except (ValueError, TypeError) as e:
                st.error(f"❌ Invalid value for {feature}: {value}")
                logger.error(f"Validation error for {feature}: {e}")
                st.stop()

        # FRONTEND-BACKEND DATA TRANSFORMATION
        # ===================================
        # Reshape for model input: model expects shape (batch_size, 13)
        input_array = np.array([inputs], dtype=float)  # Shape: (1, 13)
        logger.info(f"Model input shape: {input_array.shape}")
        logger.info(f"Model input values: {input_array}")
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        # Show input summary
        with st.expander("View Input Features", expanded=False):
            input_dict = dict(zip(WINE_FEATURES, inputs))
            st.json(input_dict)
            st.caption(f"Input shape: {input_array.shape}")

        # MODEL PREDICTION
        # ===============
        if model is None:
            # DEMO MODE: Generate mock predictions when model unavailable
            predictions = get_mock_prediction()
            logger.warning("Using mock predictions - model not loaded")
            logger.info("Using mock predictions for demonstration")
        else:
            # Generate prediction using the loaded model
            # model.predict() returns probability scores for each class
            predictions = predict_with_model(model, input_array)
            logger.info(f"Raw model predictions: {predictions}")

        # Ensure predictions shape is as expected
        if predictions is None:
            raise RuntimeError("Model returned no predictions.")
        predictions = np.asarray(predictions)
        if predictions.ndim != 2 or predictions.shape[1] != len(WINE_CLASSES):
            # Try to reshape/truncate if model returned vector
            if predictions.ndim == 1 and predictions.shape[0] == len(WINE_CLASSES):
                predictions = predictions.reshape(1, -1)
            else:
                raise RuntimeError(f"Unexpected prediction shape: {predictions.shape}")

        # Convert predictions to interpretable format
        predicted_class_index = int(np.argmax(predictions[0]))  # Index of highest probability
        predicted_class = WINE_CLASSES[predicted_class_index]  # Human-readable class name
        confidence = float(predictions[0][predicted_class_index])  # Confidence score

        # Display main prediction with confidence
        st.success(f"### 🎯 Predicted Class: **{predicted_class}**")
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Display all probabilities
        st.write("#### All Class Probabilities:")
        probs = {WINE_CLASSES[i]: float(predictions[0][i]) for i in range(len(WINE_CLASSES))}
        
        # Create columns for probability display
        prob_cols = st.columns(3)
        for i, (class_name, prob) in enumerate(probs.items()):
            with prob_cols[i]:
                st.metric(
                    label=class_name,
                    value=f"{prob:.2%}",
                    delta=None
                )
        
        # Detailed probability breakdown
        with st.expander("Detailed Probability Breakdown"):
            st.json(probs)
            st.bar_chart(probs)
        
        logger.info(f"Prediction complete: {predicted_class} (confidence: {confidence:.2f})")

    except Exception as e:
        logger.exception("Prediction error")
        st.error(f"❌ **Prediction error:** {str(e)}")
        st.write("Please check your inputs and try again.")

# Provide additional info and troubleshooting tips
with st.expander("ℹ️ About This App"):
    st.write("""
    **Model Details:**
    - Input Features (13): alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
      total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
      color_intensity, hue, od280/od315_of_diluted_wines, proline
    - Output Classes (3): class_0, class_1, class_2 (wine quality/type categories)
    - Architecture: TensorFlow/Keras neural network for wine classification
    
    **Technical Notes:**
    - Create a file named `sample-data.txt` in the same directory with 13 comma-separated values
    - Example format: `14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0`
    - The model is loaded lazily to avoid issues with native C/C++ extensions during Streamlit's watcher threads.
    - If you get a crash on startup, run a minimal app to test Streamlit:
      ```
      cat > hello.py <<'PY'
      import streamlit as st
      st.write('hello')
      PY
      streamlit run hello.py
      ```
    - If minimal app crashes, create a clean conda environment (recommended on macOS) with Python 3.11 and install packages from conda-forge:
      ```
      conda create -n st-env python=3.11 -y
      conda activate st-env
      conda config --set channel_priority strict
      conda install -c conda-forge streamlit numpy -y
      # and if you need TF:
      conda install -c conda-forge tensorflow -y
      streamlit run streamlit_app.py
      ```
    """)

# Show last model load traceback (if failed)
if model is None:
    with st.expander("⚠️ Model Load Logs"):
        st.write("Model failed to load. Check your local `model.py`, `model/wine.weights.h5` and TensorFlow installation.")
        st.info("The app will continue to run with mock predictions for demonstration purposes.")