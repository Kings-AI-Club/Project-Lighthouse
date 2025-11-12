# 🍷 Wine Classification AI - Flask ML Application

A comprehensive Flask-based web application that demonstrates machine learning model deployment with a complete frontend-backend architecture for wine classification.

## 🎯 Application Overview

This application showcases a full-stack ML pipeline:
- **Frontend**: Interactive HTML form with 13 wine characteristic inputs
- **Backend**: Flask API that processes form data and serves ML predictions
- **Model**: Pre-trained TensorFlow/Keras neural network for wine classification
- **UI/UX**: Modern responsive design with real-time feedback

## 🏗️ Architecture & Frontend-Backend Connections

### Key Connection Points

1. **Static File Serving** (`templates/index.html:12`)
   ```html
   <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
   ```
   - Flask automatically serves `static/style.css` via `url_for()` template function
   - Ensures proper asset loading regardless of deployment environment

2. **Dynamic Form Generation** (`templates/index.html:30-50`)
   ```html
   {% for feature in features %}
   <input name="{{ feature }}" id="{{ feature }}" ...>
   {% endfor %}
   ```
   - Backend `WINE_FEATURES` list (`app.py:51`) drives frontend form creation
   - Jinja2 templating ensures form inputs match model expectations exactly
   - Guarantees frontend-backend data contract compliance

3. **API Communication** (`templates/index.html:186-193`)
   ```javascript
   const response = await fetch('/predict', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(data)
   });
   ```
   - Frontend JavaScript POSTs JSON data to Flask `/predict` endpoint
   - Backend processes with `request.get_json()` (`app.py:98`)
   - Seamless data flow from HTML form → JavaScript → Flask → ML model

4. **Data Transformation Pipeline** (`app.py:110-122`)
   ```python
   input_features = []
   for feature in WINE_FEATURES:
       value = float(data[feature])  # Convert frontend strings to model floats
       input_features.append(value)
   input_array = np.array([input_features])  # Reshape for model input
   ```
   - Transforms frontend form data into ML model input format
   - Handles type conversion and tensor reshaping
   - Validates all required features are present

5. **Prediction Response** (`app.py:144-155`)
   ```python
   response_data = {
       'predicted_class': predicted_class,
       'confidence': confidence,
       'all_probabilities': {...}
   }
   return jsonify(response_data)
   ```
   - Backend formats ML predictions as JSON for frontend consumption
   - Frontend JavaScript displays results (`templates/index.html:208-235`)
   - Complete round-trip: User Input → Model Prediction → Visual Results

## 🧪 Model Specifications

### Input Features (13 wine characteristics):
- `alcohol` - Alcohol content percentage
- `malic_acid` - Malic acid concentration
- `ash` - Ash content
- `alcalinity_of_ash` - Alcalinity of ash
- `magnesium` - Magnesium content
- `total_phenols` - Total phenolic compounds
- `flavanoids` - Flavanoid compounds
- `nonflavanoid_phenols` - Non-flavanoid phenolic compounds
- `proanthocyanins` - Proanthocyanin content
- `color_intensity` - Color intensity measurement
- `hue` - Hue measurement
- `od280/od315_of_diluted_wines` - OD280/OD315 ratio of diluted wines
- `proline` - Proline amino acid content

### Output Classes (3 wine categories):
- `class_0` - Wine category 0
- `class_1` - Wine category 1
- `class_2` - Wine category 2

### Sample Input:
```json
{
  "alcohol": 14.23,
  "malic_acid": 1.71,
  "ash": 2.43,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127,
  "total_phenols": 2.8,
  "flavanoids": 3.06,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.29,
  "color_intensity": 5.64,
  "hue": 1.04,
  "od280/od315_of_diluted_wines": 3.92,
  "proline": 1065
}
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (included in project)
- TensorFlow/Keras compatible system

### Installation & Running

1. **Activate Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Start the Application**
   ```bash
   python app.py
   ```

3. **Access the Application**
   - Open browser to: `http://localhost:5000`
   - Fill in wine characteristics or use "Fill Sample Data" button
   - Click "Predict Wine Class" to get ML predictions

## 📁 Project Structure

```
Flask-Backend-Test/
├── app.py                 # Main Flask application with ML integration
├── model/
│   └── wine.keras        # Pre-trained TensorFlow/Keras model
├── templates/
│   └── index.html        # Frontend HTML with JavaScript
├── static/
│   └── style.css         # Responsive CSS styling
├── .venv/                # Python virtual environment
├── CLAUDE.md             # Development guidance for Claude Code
└── README.md             # This documentation
```

## 🔧 Flask Mechanisms Explained

### Route Handling
- `@app.route('/')` - Serves HTML form interface
- `@app.route('/predict', methods=['POST'])` - Processes ML predictions
- Flask automatically maps URLs to Python functions

### Template Rendering
```python
return render_template('index.html', features=WINE_FEATURES)
```
- `render_template()` uses Jinja2 engine to process HTML templates
- Variables passed as arguments become available in templates
- Enables dynamic content generation

### JSON Processing
```python
data = request.get_json()        # Parse incoming JSON
return jsonify(response_data)    # Return JSON response
```
- `request.get_json()` automatically parses JSON from frontend
- `jsonify()` converts Python objects to JSON responses
- Handles content-type headers automatically

### Model Integration
```python
model = tf.keras.models.load_model('model/wine.keras')  # Load once at startup
predictions = model.predict(input_array)               # Use for predictions
```
- Model loaded once during app initialization for efficiency
- Predictions generated on-demand for each request
- Error handling prevents crashes if model unavailable

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Validation**: Input validation with visual feedback
- **Loading States**: Shows progress during backend processing
- **Error Handling**: User-friendly error messages
- **Sample Data**: Pre-filled example for quick testing
- **Probability Visualization**: Animated bars showing prediction confidence

## 🔍 Development Notes

### Debugging
- Flask runs in debug mode (`debug=True`) for development
- Console logging shows request/response data flow
- Browser developer tools show JavaScript console output

### Extending the Model
1. Replace `model/wine.keras` with your trained model
2. Update `WINE_FEATURES` list in `app.py` to match new input features
3. Update `WINE_CLASSES` list for different output categories
4. Frontend form will automatically adapt to new features

### Security Considerations
- Input validation prevents invalid data submission
- Error handling prevents sensitive information leakage
- HTTPS recommended for production deployment

## 🚀 Production Deployment

For production use:
1. Set `debug=False` in `app.run()`
2. Use production WSGI server (Gunicorn, uWSGI)
3. Add proper error logging
4. Implement rate limiting for API endpoints
5. Add authentication if needed

## 📊 API Endpoints

### GET `/`
Returns the main HTML interface

### POST `/predict`
**Request Body:**
```json
{
  "alcohol": 14.23,
  "malic_acid": 1.71,
  // ... all 13 wine features
}
```

**Response:**
```json
{
  "predicted_class": "class_1",
  "confidence": 0.85,
  "all_probabilities": {
    "class_0": 0.10,
    "class_1": 0.85,
    "class_2": 0.05
  },
  "input_features": { /* echo of input data */ }
}
```

## 🤝 Contributing

This project demonstrates best practices for Flask-ML integration. Feel free to extend with additional features like:
- Model comparison functionality
- Batch prediction support
- Historical prediction tracking
- Advanced visualization options