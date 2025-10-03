# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

This is a Flask web application with machine learning capabilities using TensorFlow/Keras.

### Virtual Environment
The project uses a Python virtual environment located in `.venv/`. Always activate it before development:
```bash
source .venv/bin/activate
```

### Running the Application
```bash
python app.py
```
The Flask app runs in debug mode by default on localhost:5000.

## Project Structure

- `app.py` - Main Flask application with minimal setup (single route)
- `model/` - Contains machine learning models
  - `wine.keras` - Pre-trained Keras model for wine classification/prediction
- `templates/` - HTML templates for the web interface
  - `index.html` - Basic HTML boilerplate template
- `static/` - Static assets (currently empty)

## Key Dependencies

The project includes:
- Flask 3.1.2 - Web framework
- TensorFlow 2.20.0 - Machine learning framework
- Keras 3.11.3 - High-level neural networks API
- NumPy 2.3.3 - Numerical computing

## Architecture Notes

This appears to be a basic Flask application structure set up for machine learning model serving. The presence of a Keras model file suggests this is intended for ML model inference via web API, though the current implementation only has a basic "Hello, World!" route.