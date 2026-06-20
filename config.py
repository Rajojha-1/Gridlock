# Configuration settings for Bengaluru Traffic Violation Detection System

# Bounding Box Color Mapping (RGB format)
# Curated palette replacing generic primary colors with premium UI colors
COLOR_MAP = {
    'numberPlate': (239, 68, 68),          # Vibrant Coral Red
    'faceWithNoHelmet': (239, 68, 68),      # Vibrant Coral Red (Danger)
    'faceWithGoodHelmet': (16, 185, 129),    # Emerald Green (Safe)
    'faceWithBadHelmet': (245, 158, 11),     # Amber Orange (Defective)
    'rider': (99, 102, 241),               # Electric Indigo
    'triple_riding': (249, 115, 22),       # Neon Orange (Danger)
    'person_no_seatbelt': (239, 68, 68),    # Vibrant Coral Red (Danger)
    'person_seatbelt': (16, 185, 129),      # Emerald Green (Safe)
    'License_Plate': (99, 102, 241),       # Electric Indigo
}

# Vehicle classes (rendered with clean white/gray bounding boxes)
VEHICLE_CLASSES = {
    'Truck', 'bicycle', 'bus', 'car', 'lcv', 'three-wheeler', 'two-wheeler', 'vehicle'
}

# Default color for unrecognized classes (Light Slate Gray)
DEFAULT_BOX_COLOR = (148, 163, 184)

# Violation classes that must be highlighted
VIOLATION_CLASSES = {
    'faceWithNoHelmet',
    'faceWithBadHelmet',
    'triple_riding',
    'person_no_seatbelt',
    'no_helmet'
}

# Human-readable labels for violations
VIOLATION_LABELS = {
    'faceWithNoHelmet': 'No Helmet (Rider)',
    'faceWithBadHelmet': 'Unsafe / Damaged Helmet',
    'no_helmet': 'No Helmet (Multi-Rider)',
    'triple_riding': 'Triple Riding',
    'person_no_seatbelt': 'No Seatbelt'
}

# Challan amounts (in Rupees) for each violation class
VIOLATION_CHALLANS = {
    'faceWithNoHelmet': 1000,
    'faceWithBadHelmet': 1000,
    'no_helmet': 1000,
    'triple_riding': 1000,
    'person_no_seatbelt': 1000
}

# Bengaluru Traffic Police Theme Colors
PRIMARY_NAVY = "#6366f1"     # Neon Indigo
ACCENT_YELLOW = "#fbbf24"    # Bright Gold / Amber

# Custom Premium CSS Styling for the Streamlit UI (Dark Glassmorphic Theme)
CUSTOM_CSS = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global layout overrides for Streamlit */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 50%, #020617 100%);
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Target all Streamlit header background to match dark mode */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Style titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Style the sidebar labels and status text */
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* Header Banner styling - Premium Glassmorphic Card */
    .traffic-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(30, 41, 59, 0.4) 100%);
        padding: 2.2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    .traffic-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: 2.6rem;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    
    .traffic-header p {
        font-size: 1.15rem;
        color: #cbd5e1 !important;
        margin-top: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* KPI Metrics Cards Container */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    /* Glassmorphic KPI Cards */
    .kpi-card {
        flex: 1;
        min-width: 220px;
        background: rgba(30, 41, 59, 0.45);
        padding: 1.6rem;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        backdrop-filter: blur(12px);
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .kpi-card.info-card {
        border-left: 6px solid #6366f1;
    }

    .kpi-card.danger-card {
        border-left: 6px solid #ef4444;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }

    .kpi-value.info-value {
        color: #818cf8 !important;
        text-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    }

    .kpi-value.danger-value {
        color: #f87171 !important;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }

    .kpi-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8 !important;
        margin-top: 0.75rem;
    }

    /* Style file uploader container */
    .stFileUploader {
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(30, 41, 59, 0.25) !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
    }

    /* Section styling */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: #a5b4fc;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.6rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }

    /* Dossier Grid Layout */
    .dossier-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }

    /* Premium Glassmorphic Dossier Card */
    .dossier-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        backdrop-filter: blur(12px);
    }

    .dossier-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 25px rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.35);
    }

    .dossier-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.8rem;
    }

    .dossier-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        color: #ffffff;
        letter-spacing: 0.5px;
    }

    .dossier-body {
        flex-grow: 1;
    }

    .dossier-footer {
        margin-top: 1.5rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Pill Badges for Decision States */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }

    .badge-compliant {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-enforced {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
    }

    .badge-abstained {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }

    .badge-dismissed {
        background-color: rgba(107, 114, 128, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(107, 114, 128, 0.3);
    }

    .badge-approved {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    }

    .badge-rejected {
        background-color: rgba(120, 113, 108, 0.15);
        color: #a8a29e;
        border: 1px solid rgba(120, 113, 108, 0.3);
    }

    /* Severity Tags */
    .severity-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .severity-tag.critical {
        background-color: #ef4444;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }

    .severity-tag.high {
        background-color: #f97316;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(249, 115, 22, 0.4);
    }

    .severity-tag.medium {
        background-color: #f59e0b;
        color: #ffffff;
    }

    .severity-tag.low {
        background-color: #10b981;
        color: #ffffff;
    }

    /* Violation Pill list */
    .violation-pill-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .violation-pill {
        border-radius: 50px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        border: 1px solid transparent;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    .violation-pill.helmet-pill {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border-color: rgba(239, 68, 68, 0.2);
    }

    .violation-pill.triple-pill {
        background-color: rgba(249, 115, 22, 0.12);
        color: #fb923c;
        border-color: rgba(249, 115, 22, 0.2);
    }

    .violation-pill.seatbelt-pill {
        background-color: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border-color: rgba(168, 85, 247, 0.2);
    }

    /* Plate badge visualizer */
    .violation-plate-val {
        font-family: 'Outfit', 'Courier New', monospace;
        font-weight: 700;
        background-color: rgba(99, 102, 241, 0.12);
        padding: 3px 8px;
        border-radius: 6px;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Officer Review Panel */
    .review-panel {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        backdrop-filter: blur(12px);
    }

    /* Streamlit custom styling overrides for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 0.75rem 1.2rem !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom: 2px solid #818cf8 !important;
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #020617;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
"""
