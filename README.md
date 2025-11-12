# Project-Lighthouse

> A multi-application machine learning demonstration project by **Kings AI Club** showcasing practical ML deployment with Flask and Streamlit interfaces.

## Overview

Project-Lighthouse demonstrates production-ready machine learning applications through two distinct use cases:
1. **Wine Classification** - Flask web application for classifying wines into quality categories
2. **Homelessness Risk Prediction** - Streamlit application for predicting homelessness risk based on demographic factors

Each application showcases different ML deployment approaches, UI frameworks, and real-world problem-solving scenarios.

---

## Applications

### 1. Wine Classification (Flask)

A full-stack Flask web application that classifies wines into three quality categories using a custom neural network.

#### Features
- Interactive web form with 13 wine characteristic inputs
- Real-time ML predictions with confidence scores
- Responsive design with sample data pre-fill
- RESTful API endpoint for predictions
- Modern UI with probability visualization

#### Tech Stack
- **Backend**: Flask 3.1.2
- **ML Framework**: TensorFlow 2.20.0, Keras 3.11.3
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: Custom 4-layer FNN (ThreeClassFNN)

#### Wine Features (13 inputs)
```
alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
color_intensity, hue, od280/od315_of_diluted_wines, proline
```

#### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Flask application
python app.py

# Access at http://localhost:8000
```

#### API Endpoint

**POST `/predict`**
```json
// Request
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

// Response
{
  "predicted_class": "class_1",
  "confidence": 0.85,
  "all_probabilities": {
    "class_0": 0.10,
    "class_1": 0.85,
    "class_2": 0.05
  },
  "input_features": { /* echo of inputs */ }
}
```

---

### 2. Homelessness Risk Prediction (Streamlit)

An interactive Streamlit application that predicts homelessness risk based on demographic and social factors using synthetic data generated via Iterative Proportional Fitting (IPF).

#### Features
- Interactive sidebar with demographic inputs
- Real-time risk assessment with visual indicators
- Educational tool for understanding homelessness risk factors
- One-hot encoded categorical variables (Location, Age groups)
- Binary risk factor analysis

#### Tech Stack
- **Frontend/Backend**: Streamlit
- **ML Framework**: TensorFlow 2.20.0, Keras 3.11.3
- **Loss Function**: Focal Loss (for handling class imbalance)
- **Model**: Feedforward Neural Network with L2 regularization

#### Risk Factors (14 inputs)
```
Demographics:
  - Gender (binary)
  - Age (7 groups: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+)

Risk Indicators:
  - Drug use
  - Mental health
  - Indigenous status
  - Domestic violence

Location (one-hot):
  - ACT, NSW, NT, QLD, SA, TAS, VIC, WA
```

#### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Streamlit application
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

#### Model Architecture
- **Input Layer**: 20 features (after one-hot encoding)
- **Hidden Layers**: 64 → 32 → 16 neurons with ReLU activation
- **Regularization**: L2 regularization + Dropout (30%, 20%)
- **Output**: Binary classification (homeless/not homeless)
- **Training**: Early stopping, learning rate reduction, stratified splits

---

## Project Structure

```
Project-Lighthouse/
├── app.py                          # Flask application for wine classification
├── streamlit_app.py                # Streamlit app for homelessness prediction
├── model.py                        # ThreeClassFNN model definition
├── focal_loss.py                   # Focal loss implementation
├── model/
│   ├── wine.weights.h5            # Wine classification model weights
│   └── homelessness_risk_model.h5 # Homelessness prediction model
├── templates/
│   └── index.html                 # Flask frontend template
├── static/
│   └── style.css                  # CSS styling for Flask app
├── sample-data.txt                # Sample wine data for testing
├── .venv/                         # Python virtual environment
├── CLAUDE.md                      # AI assistant development guide
├── LICENSE                        # Project license
└── README.md                      # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum (for TensorFlow)

### Virtual Environment Setup

The project includes a pre-configured virtual environment in `.venv/`:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### Dependencies

Key packages (installed in `.venv/`):
- Flask 3.1.2
- TensorFlow 2.20.0
- Keras 3.11.3
- Streamlit (for homelessness app)
- NumPy 2.3.3
- focal-loss 0.0.7

To verify installation:
```bash
pip list | grep -E "Flask|tensorflow|streamlit|keras"
```

---

## Usage Examples

### Wine Classification Example

**Using the Web Interface:**
1. Navigate to `http://localhost:8000`
2. Click "Fill Sample Data" for pre-populated values
3. Adjust any wine characteristics as needed
4. Click "Predict Wine Class"
5. View prediction results with confidence scores

**Using the API:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample-data.txt
```

### Homelessness Risk Prediction Example

1. Navigate to `http://localhost:8501` (after running Streamlit)
2. Select demographic information in the sidebar:
   - Gender
   - Age group
   - Location (Australian state)
3. Toggle risk factors:
   - Drug use
   - Mental health issues
   - Indigenous status
   - Domestic violence
4. View real-time risk prediction with probability score

---

## Model Details

### Wine Classification Model (ThreeClassFNN)

```python
Architecture:
  Input(13) → Dense(32, ReLU) → Dropout(0.3) →
  Dense(32, ReLU) → Dropout(0.3) →
  Dense(32, ReLU) → Dropout(0.3) →
  Dense(32, ReLU) → Softmax(3)

Optimizer: Adam
Loss: Categorical Cross-Entropy
Metrics: Accuracy
```

### Homelessness Risk Model

```python
Architecture:
  Input(20) → Dense(64, ReLU, L2) → Dropout(0.3) →
  Dense(32, ReLU, L2) → Dropout(0.2) →
  Dense(16, ReLU) → Sigmoid(1)

Optimizer: Adam (legacy, lr=0.001)
Loss: Binary Focal Loss
Metrics: Accuracy, AUC
Training: Early stopping + LR reduction
```

**Training Data**: Synthetic data generated using IPF (Iterative Proportional Fitting) to match realistic demographic patterns while maintaining privacy.

---

## Development

### Running in Development Mode

Both applications run in debug/development mode by default:

```bash
# Flask (debug=True, auto-reload enabled)
python app.py

# Streamlit (watch mode enabled)
streamlit run streamlit_app.py
```

### Logging

Both applications use Python's `logging` module:
- Flask: Logs requests, predictions, and errors to console
- Streamlit: Logs model loading and prediction events

### Extending the Models

**To add new wine features:**
1. Update `WINE_FEATURES` list in `app.py:61-65`
2. Retrain model with new input dimension
3. Update frontend form (auto-generates from `WINE_FEATURES`)

**To modify homelessness risk factors:**
1. Update feature lists in `streamlit_app.py:22-28`
2. Adjust `create_feature_array()` function
3. Retrain model with new architecture

---

## Architecture Notes

### Flask Application Architecture

```
User Browser
    ↓
HTML Form (templates/index.html)
    ↓ [JavaScript fetch()]
Flask Route: POST /predict (app.py:89)
    ↓
Data Validation & Transformation
    ↓
ThreeClassFNN.predict()
    ↓
JSON Response → Frontend Display
```

**Key Connection Points:**
- **Template Rendering**: Jinja2 engine generates HTML with Python data
- **Static Files**: Flask serves CSS/JS via `url_for('static', ...)`
- **API Communication**: JavaScript `fetch()` ↔ Flask `request.get_json()`
- **Model Integration**: Loaded once at startup, used for all predictions

### Streamlit Application Architecture

```
Streamlit Frontend (Auto-generated)
    ↓
User Inputs (Sidebar widgets)
    ↓ [st.session_state]
Feature Engineering (one-hot encoding)
    ↓
Model Prediction (cached)
    ↓
Real-time UI Update (st.write, st.metric)
```

**Key Features:**
- **Session State**: Maintains model in memory across reruns
- **Caching**: `@st.cache_resource` prevents model reloading
- **Reactive**: UI updates automatically on input change
- **No API**: Direct Python function calls (no REST layer)

---

## Production Deployment

### Flask Application

**Recommended Setup:**
```bash
# Use production WSGI server
pip install gunicorn

# Run with gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Or with uWSGI
uwsgi --http :8000 --wsgi-file app.py --callable app
```

**Production Checklist:**
- [ ] Set `debug=False` in `app.run()`
- [ ] Use HTTPS/TLS certificates
- [ ] Implement rate limiting (Flask-Limiter)
- [ ] Add request logging (Flask-Logging)
- [ ] Set up monitoring (Sentry, DataDog)
- [ ] Configure environment variables for secrets
- [ ] Enable CORS if needed (Flask-CORS)

### Streamlit Application

**Deployment Options:**
1. **Streamlit Cloud** (Recommended for quick deployment)
   ```bash
   # Push to GitHub, deploy via streamlit.io/cloud
   ```

2. **Docker Container**
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "streamlit_app.py"]
   ```

3. **Custom Server**
   ```bash
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```

---

## Security Considerations

### Input Validation
- Flask app validates all 13 wine features are present and numeric
- Streamlit enforces input types via widget constraints
- Both apps sanitize inputs before model inference

### Error Handling
- Flask returns appropriate HTTP status codes (400, 500)
- Sensitive error details hidden in production
- Fallback to mock predictions if model unavailable

### Model Security
- Models loaded from local filesystem (no remote fetch)
- No user-uploaded models accepted
- Prediction inputs bounded by frontend validation

---

## Educational Use Cases

This project is ideal for:
- **ML Engineering Students**: Learn Flask/Streamlit deployment patterns
- **Data Science Courses**: Understand model serving architecture
- **Web Development**: See frontend-backend ML integration
- **Social Impact Research**: Explore homelessness risk modeling

---

## Contributing

This is a Kings AI Club educational project. Contributions welcome!

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test locally
4. Commit with clear messages (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to functions
- Include logging for debugging
- Update README for new features

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Acknowledgments

- **Kings AI Club** - Project development and maintenance
- **UCI Machine Learning Repository** - Wine dataset inspiration
- **Synthetic Data Generation** - IPF methodology for homelessness data
- **TensorFlow/Keras Team** - ML framework support
- **Flask & Streamlit Communities** - Web framework documentation

---

## Support & Contact

For questions, issues, or contributions:
- **Repository**: [Kings-AI-Club/Project-Lighthouse](https://github.com/Kings-AI-Club/Project-Lighthouse)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Documentation**: See `CLAUDE.md` for AI-assisted development guidance

---

## Version History

### Current Version
- Flask wine classification with custom ThreeClassFNN
- Streamlit homelessness risk prediction
- Focal loss for handling class imbalance
- Comprehensive documentation and examples

### Future Roadmap
- [ ] Add model performance metrics dashboard
- [ ] Implement user authentication
- [ ] Create REST API for homelessness model
- [ ] Add data visualization for feature importance
- [ ] Deploy both apps to cloud platforms
- [ ] Create Docker compose for local development

---

**Built with ❤️ by Kings AI Club**
