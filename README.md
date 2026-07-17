# 🏠 House Price Prediction App

A modern, production-ready Streamlit application for predicting house prices using XGBoost machine learning. Features a beautiful UI, batch processing, and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Setup Instructions](#-setup-instructions)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
- [File Descriptions](#-file-descriptions)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 🎯 Core Features
- **Single Prediction** - Predict house price for individual properties
- **Batch Processing** - Upload CSV files for bulk predictions
- **Real-time Results** - Instant price predictions with confidence scores
- **Feature Importance** - Visualize which features impact price most
- **Prediction History** - Track all predictions in current session

### 💻 UI/UX
- **Modern Design** - Clean, professional interface with gradient styling
- **Responsive Layout** - Works on desktop and mobile devices
- **Interactive Charts** - Plotly visualizations for data exploration
- **Organized Form** - Features grouped by category for easy navigation
- **Dark/Light Theme** - Professional color scheme

### 🔧 Technical
- **222 Features** - Engineered feature support with auto-fill defaults
- **XGBoost Model** - Fast, accurate gradient boosting regressor
- **Session Caching** - Fast inference with model caching
- **Error Handling** - Comprehensive validation and error messages
- **Batch Export** - Download predictions as CSV

---

## 📁 Project Structure

```
house-price-app/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration & styling
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── __init__.py
│   ├── loader.py                  # Load XGBoost model & artifacts
│   └── predictor.py               # Prediction engine & inference
│
├── ui/
│   ├── __init__.py
│   ├── input_form.py              # User input form components
│   └── results.py                 # Results display & visualization
│
├── utils/
│   ├── __init__.py
│   └── feature_handler.py         # Feature engineering & management
│
└── artifacts/                      # Model files (create this folder)
    ├── xgb_model.json             # XGBoost model (from Colab)
    ├── columns.pkl                # Feature columns (from Colab)
    ├── feature_names.pkl          # Feature names (from Colab)
    └── default_values.csv         # Default feature values (from Colab)
```

---

## 🚀 Installation

### Step 1: Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 2: Clone or Download Project
```bash
# Option A: Clone (if using git)
git clone https://github.com/yourusername/house-price-app.git
cd house-price-app

# Option B: Download and extract ZIP file manually
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Setup Instructions

### Step 1: Create Artifacts Folder
Create an `artifacts` folder in your project root:
```
house-price-app/
└── artifacts/  <-- Create this folder
```

### Step 2: Add Model Files
Copy your 4 files from Google Colab into the `artifacts` folder:

- `xgb_model.json` - Your XGBoost model
- `columns.pkl` - Feature column names
- `feature_names.pkl` - Feature names
- `default_values.csv` - Default feature values

Your folder should look like:
```
house-price-app/
├── app.py
├── config.py
├── requirements.txt
└── artifacts/
    ├── xgb_model.json
    ├── columns.pkl
    ├── feature_names.pkl
    └── default_values.csv
```

### Step 3: Verify Setup
```bash
# Check if all files are in place
python -c "
import os
files = ['artifacts/xgb_model.json', 'artifacts/columns.pkl', 
         'artifacts/feature_names.pkl', 'artifacts/default_values.csv']
for f in files:
    print(f'{f}: {'✓' if os.path.exists(f) else '✗'}')"
```

---

## ▶️ Running the App

### Start the Streamlit Server
```bash
streamlit run app.py
```

### Expected Output
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Open in Browser
Click the link or manually open:
```
http://localhost:8501
```

### Stop the App
Press `Ctrl + C` in your terminal

---

## 📖 Usage Guide

### 🎯 Tab 1: Single Prediction

1. **Enter Property Details**
   - Use the organized form with 7 feature groups
   - Adjust sliders for numeric values
   - Values automatically range to reasonable bounds

2. **Click "Predict Price"**
   - Model processes all 222 features
   - Results display instantly

3. **View Results**
   - **Predicted Price** - Main price prediction
   - **Confidence Level** - How confident the model is (0-100%)
   - **Estimated Range** - Lower and upper price bounds
   - **Top Factors** - Chart showing most important features
   - **Property Insights** - Price/sqft, quality rating, age

### 📊 Tab 2: Batch Upload

1. **Prepare CSV File**
   - Column names should match form fields
   - One property per row
   - Example:
     ```
     LotArea,GrLivArea,OverallQual,BedroomAbvGr,FullBath
     5000,1500,7,3,2
     6000,2000,8,4,2.5
     ```

2. **Upload File**
   - Click "Choose a CSV file"
   - Select your prepared CSV

3. **Process Batch**
   - Click "Process Batch"
   - Watch progress bar
   - Results display in table

4. **Download Results**
   - Click "Download Results (CSV)"
   - File includes all original data + predictions + confidence

### 📈 Tab 3: History & Insights

- View all predictions from current session
- See price and confidence distribution charts
- Analyze prediction trends
- Clear history if needed

### ℹ️ Tab 4: How It Works

- Understand the model architecture
- Learn about features and preprocessing
- See technical stack details
- Understand use cases

---

## 📄 File Descriptions

### Core Files

#### `app.py` - Main Application
- Entry point for the Streamlit app
- Handles tab navigation
- Manages session state and prediction history
- Renders all UI components

#### `config.py` - Configuration & Styling
- Page settings and metadata
- Custom CSS for modern design
- Feature group definitions
- Display settings and paths

#### `models/loader.py` - Model Loading
- Loads XGBoost model from JSON
- Loads pickle files (columns, features)
- Loads default values from CSV
- Validates artifacts on startup

#### `models/predictor.py` - Prediction Engine
- Makes price predictions
- Calculates confidence scores
- Computes feature importance
- Handles batch predictions

#### `ui/input_form.py` - Input Components
- Renders organized form
- Creates feature group sections
- Handles slider and input validation
- Optional quick presets

#### `ui/results.py` - Results Display
- Shows prediction results with metrics
- Creates visualizations (Plotly charts)
- Displays feature importance
- Batch results table and export

#### `utils/feature_handler.py` - Feature Engineering
- Manages 222 features
- Auto-fills defaults from CSV
- Builds feature vectors
- Handles log transformations
- Validates features before prediction

---

## 🏗️ Architecture

### Data Flow
```
User Input
    ↓
Input Form (ui/input_form.py)
    ↓
Feature Handler (utils/feature_handler.py)
    ├─ Load defaults (222 features)
    ├─ Override with user input (~25 features)
    └─ Build feature vector
    ↓
Model Predictor (models/predictor.py)
    ├─ Load XGBoost model
    ├─ Make prediction
    ├─ Calculate confidence
    └─ Get feature importance
    ↓
Results Display (ui/results.py)
    ├─ Show prediction & metrics
    ├─ Visualize feature importance
    └─ Display insights
```

### Session State Management
```
streamlit.session_state
├── prediction_history[]     # List of all predictions
└── form_data               # Current form inputs
```

### Caching Strategy
```
@st.cache_resource
├── load_model_artifacts()  # Load once per session
└── HousePricePredictor()   # Reuse instance
```

---

## 🔍 Troubleshooting

### Error: "artifacts folder not found"
**Solution:**
```bash
# Create artifacts folder in project root
mkdir artifacts

# Copy your 4 files into it:
# - xgb_model.json
# - columns.pkl
# - feature_names.pkl
# - default_values.csv
```

### Error: "ModuleNotFoundError"
**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or individually:
pip install streamlit xgboost pandas numpy plotly
```

### Error: "Feature dimension mismatch"
**Solution:**
- Verify all 4 artifact files are present
- Check that columns.pkl and feature_names.pkl match (should both have 222 items)
- Ensure default_values.csv has correct format (feature name in column 0, value in column 1)

### App Runs But Shows Blank Screen
**Solution:**
```bash
# Check for errors in terminal
# Try:
streamlit run app.py --logger.level=debug

# Look for error messages
# Ensure artifacts/ folder and files are readable
chmod 755 artifacts/*  # Linux/macOS
```

### Predictions Seem Unrealistic
**Solution:**
- Check input ranges match your training data
- Verify log transformation is correctly applied
- Review feature defaults match your dataset
- Consider retraining with new data

### Batch Upload Not Working
**Solution:**
- Verify CSV column names match form fields exactly
- Check for missing values (they should be handled or filled)
- Ensure numeric columns are actual numbers, not text
- Keep CSV file under 100MB for best performance

---

## 🎨 Customization

### Change Color Scheme
Edit `config.py` and modify the color palette in the STYLES variable:
```python
--primary: #0066cc;        # Blue
--accent: #00cc66;         # Green
--warning: #ff6633;        # Orange
```

### Add More Features
1. Edit `config.py` and add to `FEATURE_GROUPS`
2. Update `default_values.csv` with new defaults
3. Ensure your model was trained on these features

### Modify UI Layout
Edit sections in `app.py` to rearrange tabs or components.

### Change Model
Replace files in `artifacts/` with new XGBoost model files.

---

## 📊 Portfolio Highlights

This app demonstrates:

✅ **Full-Stack Development** - Backend model integration + modern frontend  
✅ **ML Engineering** - Feature engineering, model deployment, inference  
✅ **Software Architecture** - Modular, SOLID-principle code structure  
✅ **UI/UX Design** - Professional, responsive design with Streamlit  
✅ **Data Processing** - Batch handling, validation, transformations  
✅ **Error Handling** - Comprehensive validation and user feedback  
✅ **Performance** - Model caching, efficient inference  
✅ **Documentation** - Clear code comments and user guides  

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repo
4. Deploy with one click

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### AWS EC2 / Azure VM
Install Python and Streamlit, then run the app on a cloud instance.

---

## 📞 Support

### Common Questions

**Q: Can I use my own model?**  
A: Yes, replace the files in `artifacts/` with your model files.

**Q: How do I add more features?**  
A: Add to `FEATURE_GROUPS` in `config.py` and update defaults.

**Q: Can I change the styling?**  
A: Yes, edit the CSS in `config.py` STYLES variable.

**Q: What if model predictions seem wrong?**  
A: Verify input ranges, check log transformations, and validate defaults against training data.

---

## 📝 License

This project is provided as-is for educational and portfolio purposes.

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Add model files to `artifacts/` folder
3. ✅ Run the app: `streamlit run app.py`
4. ✅ Test single and batch predictions
5. ✅ Share link or deploy to Streamlit Cloud
6. ✅ Customize colors and features as needed

---

**Happy Predicting! 🏠📊**

Built with ❤️ using Streamlit & XGBoost
