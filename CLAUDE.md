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

## FYI: Model Structure File

"""
TensorFlow Keras FNN for Homelessness Risk Prediction
======================================================
Trains on IPF-generated synthetic data with realistic demographic patterns.

Data features:
- Gender (binary)
- Age (numeric)
- Drug, Mental, Indigenous, DV (binary risk factors)
- Location (one-hot encoded: ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
- SHS_Client (binary indicator)
- Target: Homeless (binary)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Core ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("HOMELESSNESS RISK PREDICTION MODEL")
print("="*80)

# ============================================================================
# 1) Load IPF-generated data
# ============================================================================
print("\n1. Loading data...")
csv_path = Path("../data/synthetic_homelessness_data.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Couldn't find {csv_path}. Please run ipf.py first.")

df = pd.read_csv(csv_path)
print(f"   Loaded {len(df):,} samples")
print(f"   Features: {df.shape[1]-1}")

# ============================================================================
# 2) Prepare features and target
# ============================================================================
print("\n2. Preparing features and target...")

# Target variable
y = df['Homeless'].values
print(f"   Target distribution: {y.sum():,} homeless ({y.mean()*100:.1f}%), {(~y.astype(bool)).sum():,} not homeless ({(1-y.mean())*100:.1f}%)")

# Features
feature_cols = [col for col in df.columns if col != 'Homeless']
X = df[feature_cols].copy()

# Convert all boolean columns to int (for TensorFlow compatibility)
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Identify numeric columns to scale
numeric_cols = ['Age']  # Only Age needs scaling, rest are binary/one-hot

print(f"   Features ({len(feature_cols)}): {feature_cols}")

# ============================================================================
# 3) Train/validation/test split
# ============================================================================
print("\n3. Splitting data...")

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Test: {len(X_test):,} samples")

# ============================================================================
# 4) Scale numeric features
# ============================================================================
print("\n4. Scaling numeric features...")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled['Age'] = scaler.fit_transform(X_train[['Age']])
X_val_scaled['Age'] = scaler.transform(X_val[['Age']])
X_test_scaled['Age'] = scaler.transform(X_test[['Age']])

print(f"   Scaled Age feature (mean: {scaler.mean_[0]:.1f}, std: {np.sqrt(scaler.var_[0]):.1f})")

# ============================================================================
# 5) Build FNN model
# ============================================================================
print("\n5. Building FNN model...")

def build_model(input_dim):
    """Build a feedforward neural network for binary classification"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model

input_dim = X_train_scaled.shape[1]
model = build_model(input_dim)

print(f"   Model architecture:")
model.summary()

# ============================================================================
# 6) Train model
# ============================================================================
print("\n6. Training model...")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# Convert to numpy arrays for TensorFlow
X_train_array = X_train_scaled.values
X_val_array = X_val_scaled.values
X_test_array = X_test_scaled.values

# Train
history = model.fit(
    X_train_array, y_train,
    validation_data=(X_val_array, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================================
# 7) Evaluate model
# ============================================================================
print("\n" + "="*80)
print("7. MODEL EVALUATION")
print("="*80)

# Training set
train_loss, train_acc, train_auc = model.evaluate(X_train_array, y_train, verbose=0)
print(f"\nTraining Set:")
print(f"   Loss: {train_loss:.4f}")
print(f"   Accuracy: {train_acc:.4f}")
print(f"   AUC: {train_auc:.4f}")

# Validation set
val_loss, val_acc, val_auc = model.evaluate(X_val_array, y_val, verbose=0)
print(f"\nValidation Set:")
print(f"   Loss: {val_loss:.4f}")
print(f"   Accuracy: {val_acc:.4f}")
print(f"   AUC: {val_auc:.4f}")

# Test set (final evaluation)
test_loss, test_acc, test_auc = model.evaluate(X_test_array, y_test, verbose=0)
print(f"\nTest Set:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   AUC: {test_auc:.4f}")

# Predictions on test set
y_pred_proba = model.predict(X_test_array, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"\n" + "="*80)
print("CLASSIFICATION REPORT (Test Set)")
print("="*80)
print(classification_report(y_test, y_pred, target_names=['Not Homeless', 'Homeless']))

print(f"\n" + "="*80)
print("CONFUSION MATRIX (Test Set)")
print("="*80)
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted")
print(f"              Not Homeless  Homeless")
print(f"Actual Not Homeless    {cm[0,0]:8d}    {cm[0,1]:8d}")
print(f"       Homeless        {cm[1,0]:8d}    {cm[1,1]:8d}")

# ============================================================================
# 8) Plot training history
# ============================================================================
print("\n8. Generating training plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUC
axes[2].plot(history.history['auc'], label='Train')
axes[2].plot(history.history['val_auc'], label='Validation')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].set_title('Model AUC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("   Saved training plots to: training_history.png")
plt.show()

# ============================================================================
# 9) Save model
# ============================================================================
print("\n9. Saving model...")
model.save('homelessness_risk_model.keras')
print("   Model saved to: homelessness_risk_model.keras")

# Save scaler
import pickle
with open('age_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   Scaler saved to: age_scaler.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)


