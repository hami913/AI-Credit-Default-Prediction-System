import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Import custom modules
from config import PAGE_CONFIG, STYLES, FEATURE_GROUPS
from models.loader import load_model_artifacts
from models.predictor import HousePricePredictor
from ui.input_form import render_input_form
from ui.results import render_prediction_results, render_batch_results
from utils.feature_handler import FeatureHandler

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Load custom CSS
st.markdown(STYLES, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
@st.cache_resource
def initialize_app():
    """Load model artifacts once at startup"""
    artifacts = load_model_artifacts()
    predictor = HousePricePredictor(artifacts)
    feature_handler = FeatureHandler(artifacts)
    return predictor, feature_handler, artifacts

# Initialize
predictor, feature_handler, artifacts = initialize_app()

# Initialize session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "form_data" not in st.session_state:
    st.session_state.form_data = None

# ============================================================================
# HEADER & HERO SECTION
# ============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏠 House Price Predictor</h1>
        <p class="hero-subtitle">Powered by Advanced XGBoost Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric(
        label="Model Ready",
        value="✓ Active",
        delta="222 Features",
        delta_color="off"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN INTERFACE - TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "📊 Batch Upload",
    "📈 History & Insights",
    "ℹ️ How It Works"
])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================
with tab1:
    st.markdown("### Enter Property Details")
    
    col_form_left, col_form_right = st.columns([2, 1])
    
    with col_form_left:
        # Render input form
        form_data = render_input_form(feature_handler, FEATURE_GROUPS)
        
        # Prediction button
        if st.button("🚀 Predict Price", use_container_width=True, 
                     help="Generate price prediction based on entered details"):
            
            # Build full feature set
            full_features = feature_handler.build_feature_vector(form_data)
            
            # Get prediction
            prediction, confidence, feature_impacts = predictor.predict(full_features)
            
            # Store in history
            st.session_state.prediction_history.append({
                "timestamp": datetime.now(),
                "features": form_data,
                "prediction": prediction,
                "confidence": confidence,
                "feature_impacts": feature_impacts
            })
            
            # Display results
            render_prediction_results(
                prediction=prediction,
                confidence=confidence,
                feature_impacts=feature_impacts,
                input_features=form_data
            )
    
    with col_form_right:
        st.markdown("""
        <div class="info-box">
            <p><strong>💡 Tip</strong></p>
            <p>Adjust property characteristics to see how they affect the predicted price in real-time.</p>
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p><strong>📊 Model Info</strong></p>
            <p>XGBoost Regressor<br>Features: 222<br>Algorithm: Gradient Boosting</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: BATCH UPLOAD
# ============================================================================
with tab2:
    st.markdown("### Upload CSV for Batch Predictions")
    
    col_upload_left, col_upload_right = st.columns([2, 1])
    
    with col_upload_left:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should contain the same columns as the form above"
        )
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                
                st.success(f"✓ Loaded {len(input_df)} records")
                
                # Show preview
                with st.expander("📋 Preview Input Data"):
                    st.dataframe(input_df.head(10), use_container_width=True)
                
                # Process batch
                if st.button("⚡ Process Batch", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    confidences = []
                    
                    for idx, row in input_df.iterrows():
                        # Build feature vector for each row
                        row_dict = row.to_dict()
                        full_features = feature_handler.build_feature_vector(row_dict)
                        
                        # Get prediction
                        pred, conf, _ = predictor.predict(full_features)
                        predictions.append(pred)
                        confidences.append(conf)
                        
                        # Update progress
                        progress = (idx + 1) / len(input_df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(input_df)}")
                    
                    # Create results dataframe
                    results_df = input_df.copy()
                    results_df["predicted_price"] = predictions
                    results_df["confidence"] = confidences
                    
                    # Display results
                    render_batch_results(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col_upload_right:
        st.markdown("""
        <div class="info-box">
            <p><strong>📝 Requirements</strong></p>
            <p>• CSV format<br>• Numeric columns<br>• Same structure as form</p>
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p><strong>🚀 Performance</strong></p>
            <p>Processes 100+ records in seconds</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: HISTORY & INSIGHTS
# ============================================================================
with tab3:
    if st.session_state.prediction_history:
        st.markdown(f"### Prediction History ({len(st.session_state.prediction_history)} records)")
        
        # Summary metrics
        predictions = [h["prediction"] for h in st.session_state.prediction_history]
        confidences = [h["confidence"] for h in st.session_state.prediction_history]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(predictions))
        col2.metric("Avg Price", f"${np.mean(predictions):,.0f}")
        col3.metric("Price Range", f"${np.max(predictions) - np.min(predictions):,.0f}")
        col4.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
        
        st.markdown("---")
        
        # Distribution visualization
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Price Distribution", "Confidence Distribution")
        )
        
        fig.add_trace(
            go.Histogram(x=predictions, name="Price", nbinsx=20, 
                        marker_color="#0066cc"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=confidences, name="Confidence", nbinsx=20,
                        marker_color="#00cc66"),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Confidence (%)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False, hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history table
        st.markdown("### Detailed Prediction Log")
        history_data = []
        for h in st.session_state.prediction_history:
            history_data.append({
                "Timestamp": h["timestamp"].strftime("%H:%M:%S"),
                "Predicted Price": f"${h['prediction']:,.0f}",
                "Confidence": f"{h['confidence']:.1%}",
                "Square Ft": h["features"].get("GrLivArea", "N/A"),
                "Bedrooms": h["features"].get("BedroomAbvGr", "N/A")
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        # Clear history
        if st.button("🗑️ Clear History", help="Remove all predictions from this session"):
            st.session_state.prediction_history = []
            st.rerun()
    
    else:
        st.info("📊 No predictions yet. Make a prediction in the 'Single Prediction' tab to see history here.")

# ============================================================================
# TAB 4: HOW IT WORKS
# ============================================================================
with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🤖 Model Architecture
        
        This application uses an **XGBoost Regressor** trained on house price data.
        
        **Key Components:**
        
        1. **Input Processing** - Your inputs are processed and normalized
        2. **Feature Engineering** - 222 engineered features are automatically generated
        3. **Model Inference** - XGBoost generates price predictions
        4. **Confidence Scoring** - Uncertainty quantification for each prediction
        
        ### 📊 Feature Information
        
        - **Total Features**: 222 engineered features
        - **Input Features**: ~25 user-controllable parameters
        - **Auto-populated**: Remaining features from learned defaults
        
        ### 🎯 Use Cases
        
        ✅ **Real Estate Valuation** - Quick property valuations  
        ✅ **Investment Analysis** - Compare property values across markets  
        ✅ **Appraisal Support** - Data-driven pricing insights  
        ✅ **Market Research** - Batch analysis of property listings  
        
        ### 📈 Model Performance
        
        - **Algorithm**: Gradient Boosting Regressor
        - **Training Data**: Comprehensive house price dataset
        - **Prediction Type**: Continuous (price in $)
        - **Output**: Predicted price with confidence interval
        
        """)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <p><strong>🔍 Data Flow</strong></p>
            <p>
            User Input<br>
            ↓<br>
            Feature Engineering<br>
            ↓<br>
            XGBoost Model<br>
            ↓<br>
            Price Prediction
            </p>
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p><strong>⚙️ Technical Stack</strong></p>
            <p>Streamlit<br>XGBoost<br>Pandas<br>NumPy<br>Plotly</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>🏗️ Built with Streamlit & XGBoost | Machine Learning Model v1.0</p>
    <p>© 2024 | House Price Prediction System</p>
</div>
""", unsafe_allow_html=True)
