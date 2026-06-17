# Configuration settings for Bengaluru Traffic Violation Detection System

# Bounding Box Color Mapping (RGB format)
COLOR_MAP = {
    'numberPlate': (255, 0, 0),          # Red
    'faceWithNoHelmet': (0, 255, 0),      # Green
    'faceWithGoodHelmet': (0, 0, 255),    # Blue
    'faceWithBadHelmet': (255, 255, 0),   # Yellow
    'rider': (255, 0, 255),               # Magenta
    'triple_riding': (255, 165, 0),       # Orange
    'person_no_seatbelt': (255, 0, 0),    # Red
    'person_seatbelt': (0, 0, 255),       # Blue
    'License_Plate': (0, 255, 255),       # Cyan
}

# Vehicle classes (rendered with white bounding boxes)
VEHICLE_CLASSES = {
    'Truck', 'bicycle', 'bus', 'car', 'lcv', 'three-wheeler', 'two-wheeler', 'vehicle'
}

# Default color for unrecognized classes (Light Gray)
DEFAULT_BOX_COLOR = (200, 200, 200)

# Violation classes that must be highlighted
VIOLATION_CLASSES = {
    'faceWithNoHelmet',
    'faceWithBadHelmet',
    'triple_riding',
    'person_no_seatbelt',
    'no_helmet'
}

# Bengaluru Traffic Police Colors
PRIMARY_NAVY = "#1a237e"
ACCENT_YELLOW = "#fdd835"

# Custom Premium CSS Styling for the Streamlit UI
CUSTOM_CSS = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');

    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App Header Banner styled with Navy Blue and Yellow */
    .traffic-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d1b60 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        border-bottom: 6px solid #fdd835;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(26, 35, 126, 0.15);
    }
    
    .traffic-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #fdd835 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .traffic-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.75rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Info Badge / Metric Card containers */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    .kpi-card {
        flex: 1;
        min-width: 220px;
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .kpi-card.info-card {
        border-left: 6px solid #1a237e;
    }

    .kpi-card.danger-card {
        border-left: 6px solid #d32f2f;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
    }

    .kpi-value.info-value {
        color: #1a237e;
    }

    .kpi-value.danger-value {
        color: #d32f2f;
    }

    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6c757d;
        margin-top: 0.75rem;
    }

    /* Style improvements for file uploader and sidebar */
    .stFileUploader {
        border: 2px dashed #1a237e !important;
        border-radius: 12px !important;
        background-color: rgba(26, 35, 126, 0.02) !important;
        padding: 1.5rem !important;
    }

    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
    }

    /* Section styling */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a237e;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
</style>
"""
