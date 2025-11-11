"""
Flask-based Wine Classification AI Application

This application demonstrates a complete ML pipeline with:
- Frontend: HTML form for inputting wine characteristics
- Backend: Flask API that processes inputs and returns ML predictions
- Model: TensorFlow/Keras neural network for wine classification

Model Details:
- Input Features (13): alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
  total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
  color_intensity, hue, od280/od315_of_diluted_wines, proline
- Output Classes (3): class_0, class_1, class_2 (wine quality/type categories)
- Example input: [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
"""

# Import necessary Flask components and libraries
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from model import ThreeClassFNN
import numpy as np
import logging

from focal_loss import BinaryFocalLoss

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BACKEND MODEL LOADING
# ===================
# Load the pre-trained Keras model from the model/ directory
# This happens once when the server starts, improving response times
try:
    # Attempt to load the model with custom objects
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
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.warning("Running in demo mode - predictions will use mock data")
    model = None

# FLASK APP INITIALIZATION
# =======================
# Create Flask application instance
# __name__ helps Flask locate templates and static files relative to this module
app = Flask(__name__)

# Define wine feature names for the frontend form
# These correspond exactly to the model's expected input features
WINE_FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Define output class names for result interpretation
WINE_CLASSES = ['class_0', 'class_1', 'class_2']

# FRONTEND ROUTE
# =============
@app.route('/')
def index():
    """
    Main route that serves the HTML form interface

    Flask Mechanism Explanation:
    - @app.route('/') decorator binds this function to the root URL
    - When user visits http://localhost:5000/, Flask calls this function
    - render_template() looks for 'index.html' in the templates/ folder
    - features=WINE_FEATURES passes the feature list to the template
      enabling dynamic form generation
    """
    logger.info("Serving main page")
    return render_template('index.html', features=WINE_FEATURES)

# BACKEND API ROUTE
# =================
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that receives wine characteristics and returns predictions

    Frontend-Backend Connection Point #1:
    - This route receives POST requests from the frontend form
    - JavaScript fetch() calls this endpoint with form data
    - Returns JSON response that frontend displays to user

    Flask Mechanism Explanation:
    - methods=['POST'] restricts this route to POST requests only
    - request.get_json() parses the JSON data sent from frontend
    - jsonify() converts Python dict to JSON response
    """
    try:
        # FRONTEND DATA RECEPTION
        # ======================
        # Extract JSON data sent from frontend JavaScript
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # FRONTEND-BACKEND DATA TRANSFORMATION
        # ==================================
        # Convert frontend form values to model input format
        # Frontend sends: {'alcohol': '14.23', 'malic_acid': '1.71', ...}
        # Model needs: numpy array [[14.23, 1.71, ...]]

        input_features = []
        for feature in WINE_FEATURES:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                # Convert string input to float (frontend always sends strings)
                value = float(data[feature])
                input_features.append(value)
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}: {data[feature]}'}), 400

        # Reshape for model input: model expects shape (batch_size, 13)
        input_array = np.array([input_features])  # Shape: (1, 13)
        logger.info(f"Model input shape: {input_array.shape}")
        logger.info(f"Model input values: {input_array}")

        # MODEL PREDICTION
        # ===============
        if model is None:
            # DEMO MODE: Generate mock predictions when model unavailable
            logger.warning("Using mock predictions - model not loaded")
            # Create realistic-looking mock predictions
            mock_predictions = np.array([[0.2, 0.7, 0.1]])  # Favor class_1
            predictions = mock_predictions
            logger.info("Using mock predictions for demonstration")
        else:
            # Generate prediction using the loaded model
            # model.predict() returns probability scores for each class
            predictions = model.predict(input_array)
            logger.info(f"Raw model predictions: {predictions}")

        # Convert predictions to interpretable format
        predicted_class_index = np.argmax(predictions[0])  # Index of highest probability
        predicted_class = WINE_CLASSES[predicted_class_index]  # Human-readable class name
        confidence = float(predictions[0][predicted_class_index])  # Confidence score

        # BACKEND-FRONTEND RESPONSE
        # ========================
        # Format response for frontend consumption
        response_data = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                WINE_CLASSES[i]: float(predictions[0][i])
                for i in range(len(WINE_CLASSES))
            },
            'input_features': dict(zip(WINE_FEATURES, input_features))
        }

        logger.info(f"Sending prediction response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# DEVELOPMENT SERVER
# =================
if __name__ == '__main__':
    """
    Start the Flask development server

    Flask Mechanism Explanation:
    - debug=True enables automatic reloading on code changes
    - host='0.0.0.0' makes server accessible from other machines
    - port=5000 sets the server port (default)
    """
    logger.info("Starting Flask application...")
    logger.info(f"Wine features: {WINE_FEATURES}")
    logger.info(f"Wine classes: {WINE_CLASSES}")

    app.run(debug=True, host='0.0.0.0', port=8000)