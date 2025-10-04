"""
🍷 Wine Classification AI - Streamlit Application

This Streamlit application replaces the previous Flask version and provides:
- Interactive web interface for wine classification
- Real-time ML predictions using TensorFlow/Keras model
- User-friendly input forms and result visualization
- Built-in sample data for quick testing

Model Details:
- Input Features (13): alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
  total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
  color_intensity, hue, od280/od315_of_diluted_wines, proline
- Output Classes (3): class_0, class_1, class_2 (wine quality/type categories)
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from model import ThreeClassFNN
import logging
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🍷 Wine Classification AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Wine feature definitions for form inputs
WINE_FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Feature descriptions for better user understanding
FEATURE_DESCRIPTIONS = {
    'alcohol': 'Alcohol content percentage',
    'malic_acid': 'Malic acid concentration',
    'ash': 'Ash content',
    'alcalinity_of_ash': 'Alcalinity of ash',
    'magnesium': 'Magnesium content',
    'total_phenols': 'Total phenolic compounds',
    'flavanoids': 'Flavanoid compounds',
    'nonflavanoid_phenols': 'Non-flavanoid phenolic compounds',
    'proanthocyanins': 'Proanthocyanin content',
    'color_intensity': 'Color intensity measurement',
    'hue': 'Hue measurement',
    'od280/od315_of_diluted_wines': 'OD280/OD315 ratio of diluted wines',
    'proline': 'Proline amino acid content'
}

# Define output class names
WINE_CLASSES = ['Class 0', 'Class 1', 'Class 2']

# Sample data for quick testing
SAMPLE_DATA = {
    'alcohol': 14.23,
    'malic_acid': 1.71,
    'ash': 2.43,
    'alcalinity_of_ash': 15.6,
    'magnesium': 127.0,
    'total_phenols': 2.8,
    'flavanoids': 3.06,
    'nonflavanoid_phenols': 0.28,
    'proanthocyanins': 2.29,
    'color_intensity': 5.64,
    'hue': 1.04,
    'od280/od315_of_diluted_wines': 3.92,
    'proline': 1065.0
}

@st.cache_resource
def load_model():
    """
    Load the pre-trained TensorFlow/Keras model
    Uses Streamlit caching to avoid reloading on every interaction
    """
    try:
        model = ThreeClassFNN()
        model.build((None, 13))  # Input shape: (batch_size, 13 features)
        model.load_weights('model/wine.weights.h5')
        
        # Verify model with test prediction
        test_input = np.zeros((1, 13))
        _ = model.predict(test_input, verbose=0)
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"⚠️ Error loading model: {e}")
        return None

def create_input_form():
    """
    Create the input form for wine characteristics
    """
    st.subheader("🔬 Wine Characteristics Input")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Store form values
    form_values = {}
    
    # First column inputs
    with col1:
        form_values['alcohol'] = st.number_input(
            "Alcohol (%)", 
            value=SAMPLE_DATA['alcohol'],
            help=FEATURE_DESCRIPTIONS['alcohol'],
            format="%.2f"
        )
        form_values['malic_acid'] = st.number_input(
            "Malic Acid", 
            value=SAMPLE_DATA['malic_acid'],
            help=FEATURE_DESCRIPTIONS['malic_acid'],
            format="%.2f"
        )
        form_values['ash'] = st.number_input(
            "Ash", 
            value=SAMPLE_DATA['ash'],
            help=FEATURE_DESCRIPTIONS['ash'],
            format="%.2f"
        )
        form_values['alcalinity_of_ash'] = st.number_input(
            "Alcalinity of Ash", 
            value=SAMPLE_DATA['alcalinity_of_ash'],
            help=FEATURE_DESCRIPTIONS['alcalinity_of_ash'],
            format="%.2f"
        )
        form_values['magnesium'] = st.number_input(
            "Magnesium", 
            value=SAMPLE_DATA['magnesium'],
            help=FEATURE_DESCRIPTIONS['magnesium'],
            format="%.1f"
        )
        form_values['total_phenols'] = st.number_input(
            "Total Phenols", 
            value=SAMPLE_DATA['total_phenols'],
            help=FEATURE_DESCRIPTIONS['total_phenols'],
            format="%.2f"
        )
        form_values['flavanoids'] = st.number_input(
            "Flavanoids", 
            value=SAMPLE_DATA['flavanoids'],
            help=FEATURE_DESCRIPTIONS['flavanoids'],
            format="%.2f"
        )
    
    # Second column inputs
    with col2:
        form_values['nonflavanoid_phenols'] = st.number_input(
            "Non-flavanoid Phenols", 
            value=SAMPLE_DATA['nonflavanoid_phenols'],
            help=FEATURE_DESCRIPTIONS['nonflavanoid_phenols'],
            format="%.2f"
        )
        form_values['proanthocyanins'] = st.number_input(
            "Proanthocyanins", 
            value=SAMPLE_DATA['proanthocyanins'],
            help=FEATURE_DESCRIPTIONS['proanthocyanins'],
            format="%.2f"
        )
        form_values['color_intensity'] = st.number_input(
            "Color Intensity", 
            value=SAMPLE_DATA['color_intensity'],
            help=FEATURE_DESCRIPTIONS['color_intensity'],
            format="%.2f"
        )
        form_values['hue'] = st.number_input(
            "Hue", 
            value=SAMPLE_DATA['hue'],
            help=FEATURE_DESCRIPTIONS['hue'],
            format="%.2f"
        )
        form_values['od280/od315_of_diluted_wines'] = st.number_input(
            "OD280/OD315 Ratio", 
            value=SAMPLE_DATA['od280/od315_of_diluted_wines'],
            help=FEATURE_DESCRIPTIONS['od280/od315_of_diluted_wines'],
            format="%.2f"
        )
        form_values['proline'] = st.number_input(
            "Proline", 
            value=SAMPLE_DATA['proline'],
            help=FEATURE_DESCRIPTIONS['proline'],
            format="%.1f"
        )
    
    return form_values

def make_prediction(model, form_values):
    """
    Make prediction using the loaded model
    """
    try:
        # Convert form values to model input format
        input_features = [form_values[feature] for feature in WINE_FEATURES]
        input_array = np.array([input_features])
        
        logger.info(f"Model input: {input_array}")
        
        if model is None:
            # Demo mode with mock predictions
            st.warning("⚠️ Model not loaded - using mock predictions for demonstration")
            predictions = np.array([[0.2, 0.7, 0.1]])
        else:
            # Real prediction
            predictions = model.predict(input_array, verbose=0)
        
        # Process results
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = WINE_CLASSES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Create probability dictionary
        probabilities = {
            WINE_CLASSES[i]: float(predictions[0][i]) 
            for i in range(len(WINE_CLASSES))
        }
        
        return predicted_class, confidence, probabilities, input_features
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"❌ Prediction error: {e}")
        return None, None, None, None

def display_results(predicted_class, confidence, probabilities):
    """
    Display prediction results with visualizations
    """
    st.subheader("🎯 Prediction Results")
    
    # Main result
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            label="Predicted Wine Class", 
            value=predicted_class,
            help="The most likely wine classification"
        )
    
    with col2:
        st.metric(
            label="Confidence", 
            value=f"{confidence:.1%}",
            help="Model's confidence in the prediction"
        )
    
    with col3:
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
        st.metric(
            label="Confidence Level", 
            value=confidence_color,
            help="Visual confidence indicator"
        )
    
    # Probability visualization
    st.subheader("📊 Class Probabilities")
    
    # Create two types of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            title="Probability Distribution",
            labels={'x': 'Wine Class', 'y': 'Probability'},
            color=list(probabilities.values()),
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=list(probabilities.values()),
            names=list(probabilities.keys()),
            title="Class Distribution"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed probabilities table
    st.subheader("📋 Detailed Probabilities")
    prob_df = pd.DataFrame([
        {
            'Wine Class': class_name,
            'Probability': prob,
            'Percentage': f"{prob:.1%}",
            'Confidence Bar': '█' * int(prob * 20)  # Visual bar
        }
        for class_name, prob in probabilities.items()
    ])
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

def display_input_summary(form_values):
    """
    Display a summary of input values
    """
    st.subheader("📝 Input Summary")
    
    # Create a formatted dataframe
    input_df = pd.DataFrame([
        {
            'Feature': feature.replace('_', ' ').title(),
            'Value': form_values[feature],
            'Description': FEATURE_DESCRIPTIONS[feature]
        }
        for feature in WINE_FEATURES
    ])
    
    st.dataframe(input_df, use_container_width=True, hide_index=True)

def main():
    """
    Main Streamlit application
    """
    # Header
    st.title("🍷 Wine Classification AI")
    st.markdown("*Powered by TensorFlow/Keras Neural Network*")
    
    st.markdown("""
    This application uses machine learning to classify wines into three categories based on 13 chemical characteristics.
    Simply input the wine's properties below and get an instant AI-powered classification with confidence scores.
    """)
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model Information:**
        - Neural Network Architecture
        - 13 Input Features
        - 3 Wine Classes
        - TensorFlow/Keras Backend
        
        **Features:**
        - Real-time predictions
        - Confidence scoring
        - Interactive visualizations
        - Sample data loading
        """)
        
        st.header("🎯 Quick Actions")
        if st.button("🔄 Load Sample Data", help="Fill form with example wine data"):
            st.rerun()
        
        if st.button("🗑️ Clear All Fields", help="Reset all input fields"):
            for feature in WINE_FEATURES:
                if f"{feature}_input" in st.session_state:
                    del st.session_state[f"{feature}_input"]
            st.rerun()
    
    # Load model
    model = load_model()
    
    # Create input form
    form_values = create_input_form()
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Wine Class", 
            type="primary",
            use_container_width=True,
            help="Click to classify the wine based on input characteristics"
        )
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("🧠 Analyzing wine characteristics..."):
            predicted_class, confidence, probabilities, input_features = make_prediction(model, form_values)
            
            if predicted_class is not None:
                st.success("✅ Prediction completed!")
                
                # Display results
                display_results(predicted_class, confidence, probabilities)
                
                # Show input summary in expander
                with st.expander("📋 View Input Summary", expanded=False):
                    display_input_summary(form_values)
                
                # Additional insights
                st.subheader("💡 Insights")
                if confidence > 0.8:
                    st.success(f"🎯 **High Confidence Prediction**: The model is very confident this wine belongs to {predicted_class}")
                elif confidence > 0.6:
                    st.warning(f"⚡ **Moderate Confidence**: The model suggests {predicted_class} but consider reviewing the characteristics")
                else:
                    st.error(f"⚠️ **Low Confidence**: The model is uncertain. This wine's characteristics may be borderline between classes.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🍷 Wine Classification AI | Powered by Streamlit & TensorFlow
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()