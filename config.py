"""
Configuration and styling for House Price Prediction App
"""

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
PAGE_CONFIG = {
    "page_title": "🏠 House Price Predictor",
    "page_icon": "🏠",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

# ============================================================================
# CUSTOM CSS STYLING (Modern Design System)
# ============================================================================
STYLES = """
<style>
    /* Color Palette */
    :root {
        --primary: #0066cc;
        --primary-light: #e6f2ff;
        --accent: #00cc66;
        --accent-light: #e6ffe6;
        --warning: #ff6633;
        --success: #00aa55;
        --bg-light: #f8f9fa;
        --bg-card: #ffffff;
        --border: #e0e0e0;
        --text-primary: #1a1a1a;
        --text-secondary: #666666;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background-color: var(--bg-light);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                     'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                     sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, var(--primary) 0%, #0052a3 100%);
        padding: 40px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 18px;
        font-weight: 400;
        opacity: 0.95;
        letter-spacing: 0.3px;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 24px 0;
    }
    
    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border-color: var(--primary);
        transform: translateY(-2px);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--accent-light) 100%);
        border-left: 4px solid var(--primary);
        padding: 20px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        color: var(--text-primary);
    }
    
    .info-box strong {
        color: var(--primary);
        display: block;
        margin-bottom: 8px;
    }
    
    /* Metric Cards */
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Form Elements */
    .stNumberInput > label,
    .stSelectbox > label,
    .stSlider > label {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 14px;
        margin-bottom: 8px;
        letter-spacing: 0.3px;
    }
    
    .stNumberInput input,
    .stSelectbox select,
    .stSlider input {
        border: 2px solid var(--border);
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stNumberInput input:focus,
    .stSelectbox select:focus,
    .stSlider input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primary-light);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        text-transform: none;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 102, 204, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-light);
        border-bottom: 2px solid var(--border);
        padding: 12px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
        color: var(--text-secondary);
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom: 3px solid var(--primary);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background-color: transparent;
    }
    
    /* Expander */
    .stExpander {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    .stExpander > div > button {
        padding: 16px;
        font-weight: 600;
        color: var(--primary);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: var(--accent-light);
        color: var(--success);
        border-left: 4px solid var(--success);
        padding: 16px;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #ffe6e6;
        color: #cc0000;
        border-left: 4px solid #cc0000;
        padding: 16px;
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: #ffe6cc;
        color: var(--warning);
        border-left: 4px solid var(--warning);
        padding: 16px;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: var(--primary-light);
        color: var(--primary);
        border-left: 4px solid var(--primary);
        padding: 16px;
        border-radius: 8px;
    }
    
    /* Data Table */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 32px 0;
        margin-top: 48px;
        border-top: 1px solid var(--border);
        color: var(--text-secondary);
        font-size: 12px;
    }
    
    .footer p {
        margin: 4px 0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px;
        }
        
        .hero-subtitle {
            font-size: 16px;
        }
        
        .card {
            padding: 16px;
        }
        
        .info-box {
            font-size: 13px;
        }
    }
    
    /* Animation - Smooth transitions */
    * {
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
"""

# ============================================================================
# FEATURE GROUPS FOR ORGANIZED INPUT
# ============================================================================
FEATURE_GROUPS = {
    "Property Basics": {
        "LotArea": {
            "type": "slider",
            "label": "Lot Area (sq ft)",
            "min": 1300,
            "max": 215000,
            "step": 100,
            "help": "Total lot size in square feet"
        },
        "OverallQual": {
            "type": "slider",
            "label": "Overall Quality (1-10)",
            "min": 1,
            "max": 10,
            "step": 1,
            "help": "Overall material and finish quality"
        },
        "OverallCond": {
            "type": "slider",
            "label": "Overall Condition (1-10)",
            "min": 1,
            "max": 10,
            "step": 1,
            "help": "Overall condition of the property"
        },
    },
    
    "Living Space": {
        "GrLivArea": {
            "type": "slider",
            "label": "Gross Living Area (sq ft)",
            "min": 300,
            "max": 6000,
            "step": 50,
            "help": "Above grade (ground) living area square feet"
        },
        "TotRmsAbvGrd": {
            "type": "slider",
            "label": "Total Rooms (Above Ground)",
            "min": 1,
            "max": 15,
            "step": 1,
            "help": "Total rooms above ground level"
        },
        "BedroomAbvGr": {
            "type": "slider",
            "label": "Bedrooms (Above Ground)",
            "min": 0,
            "max": 8,
            "step": 1,
            "help": "Number of bedrooms"
        },
    },
    
    "Building Features": {
        "YearBuilt": {
            "type": "slider",
            "label": "Year Built",
            "min": 1872,
            "max": 2024,
            "step": 1,
            "help": "Original construction year"
        },
        "YearRemodAdd": {
            "type": "slider",
            "label": "Year Remodeled",
            "min": 1872,
            "max": 2024,
            "step": 1,
            "help": "Remodel date (same as construction if no remodel)"
        },
        "BsmtFinSF1": {
            "type": "slider",
            "label": "Basement Finished Area (sq ft)",
            "min": 0,
            "max": 6000,
            "step": 50,
            "help": "Type 1 finished basement area"
        },
    },
    
    "Garage & Parking": {
        "GarageArea": {
            "type": "slider",
            "label": "Garage Area (sq ft)",
            "min": 0,
            "max": 1500,
            "step": 50,
            "help": "Size of garage in square feet"
        },
        "GarageCars": {
            "type": "slider",
            "label": "Garage Capacity",
            "min": 0,
            "max": 5,
            "step": 1,
            "help": "Number of cars garage can hold"
        },
    },
    
    "Utilities & Features": {
        "Fireplaces": {
            "type": "slider",
            "label": "Number of Fireplaces",
            "min": 0,
            "max": 4,
            "step": 1,
            "help": "Number of fireplaces in the house"
        },
        "FullBath": {
            "type": "slider",
            "label": "Full Bathrooms",
            "min": 0,
            "max": 4,
            "step": 1,
            "help": "Full bathrooms above ground"
        },
        "HalfBath": {
            "type": "slider",
            "label": "Half Bathrooms",
            "min": 0,
            "max": 2,
            "step": 1,
            "help": "Half bathrooms above ground"
        },
    },
    
    "Exterior & Land": {
        "LotFrontage": {
            "type": "slider",
            "label": "Lot Frontage (ft)",
            "min": 20,
            "max": 300,
            "step": 5,
            "help": "Linear feet of street connected to property"
        },
        "1stFlrSF": {
            "type": "slider",
            "label": "1st Floor Area (sq ft)",
            "min": 300,
            "max": 5000,
            "step": 50,
            "help": "First floor square footage"
        },
        "2ndFlrSF": {
            "type": "slider",
            "label": "2nd Floor Area (sq ft)",
            "min": 0,
            "max": 2500,
            "step": 50,
            "help": "Second floor square footage"
        },
    },
    
    "Special Features": {
        "WoodDeckSF": {
            "type": "slider",
            "label": "Wood Deck Area (sq ft)",
            "min": 0,
            "max": 1000,
            "step": 25,
            "help": "Wood deck area in square feet"
        },
        "OpenPorchSF": {
            "type": "slider",
            "label": "Open Porch Area (sq ft)",
            "min": 0,
            "max": 600,
            "step": 25,
            "help": "Open porch area in square feet"
        },
        "PoolArea": {
            "type": "slider",
            "label": "Pool Area (sq ft)",
            "min": 0,
            "max": 750,
            "step": 25,
            "help": "Pool area in square feet"
        },
    },
}

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    "model": "artifacts/xgb_model.json",
    "columns": "artifacts/columns.pkl",
    "feature_names": "artifacts/feature_names.pkl",
    "default_values": "artifacts/default_values.csv",
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
DISPLAY_SETTINGS = {
    "price_decimals": 0,
    "confidence_decimals": 2,
    "max_history": 100,
    "chart_height": 400,
}
