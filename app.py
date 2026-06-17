import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import configuration and CSS styling
import config
from config import CUSTOM_CSS, VIOLATION_CLASSES, PRIMARY_NAVY, ACCENT_YELLOW

# Import modular helper utilities
from utils.model_loader import load_yolo_models, load_ocr_reader
from utils.inference import run_parallel_inference
from utils.ocr_processor import process_license_plates
from utils.annotator import draw_annotations
from utils.report_generator import generate_report, highlight_violations, convert_df_to_csv, convert_df_to_json
from utils.visualizer import plot_violation_breakdown, plot_confidence_distribution

# Configure Streamlit page layout
st.set_page_config(
    page_title="TrafficAI — Bengaluru Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR - Model Management & Control Panel
# ----------------------------------------------------
st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 10px; border-bottom: 2px solid {ACCENT_YELLOW}; margin-bottom: 20px;">
        <h2 style="color: {PRIMARY_NAVY}; font-family: 'Outfit', sans-serif; font-weight: 800; margin:0;">TrafficAI</h2>
        <span style="font-size: 0.85rem; color: #6c757d; font-weight: 600; letter-spacing: 1px;">BENGALURU DIVISION</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("⚙️ System Status")

# Initialize and check model status
yolo_models = load_yolo_models()
ocr_reader = load_ocr_reader()

model_loaded_count = sum(1 for m in yolo_models.values() if m is not None)
total_models = len(yolo_models)

# Display status of each YOLO model
st.sidebar.markdown("**YOLOv8 Models Status:**")
model_names = {
    'helmet': 'Helmet & Rider Model',
    'plate': 'License Plate Model',
    'triple': 'Triple Riding Model',
    'seatbelt': 'Seatbelt Model',
    'bengaluru': 'Bengaluru Vehicle Model'
}

all_models_present = True
for key, label in model_names.items():
    if yolo_models.get(key) is not None:
        st.sidebar.markdown(f"✅ {label}: `Loaded`")
    else:
        st.sidebar.markdown(f"❌ {label}: `Weights Missing`")
        all_models_present = False

# Auto-toggle simulation mode if weights are missing
if not all_models_present:
    st.sidebar.warning("⚠️ Some weights are missing from models/ folder. System will default to Simulation Mode.")
    default_sim_mode = True
else:
    default_sim_mode = False

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Controls")

# Controller to manually override simulation mode
sim_mode = st.sidebar.checkbox(
    "Enable Simulation Mode", 
    value=default_sim_mode,
    help="Runs pre-defined simulated detections for validation without model weight files."
)

# Confidence slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Detections below this confidence level will be filtered out."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**TrafficAI Bengaluru** uses 5 parallelized neural networks to enforce traffic rules. "
    "Upload a road image to detect violations like triple riding, riding without helmet, and driving without seatbelt."
)

# ----------------------------------------------------
# MAIN UI HEADER
# ----------------------------------------------------
st.markdown("""
    <div class="traffic-header">
        <h1>TrafficAI — Bengaluru Violation Detection System</h1>
        <p>AI-powered traffic enforcement for safer roads</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIMULATION DETECTIONS GENERATOR
# ----------------------------------------------------
def generate_simulated_detections(width, height, conf_threshold):
    """
    Generates a set of realistic mock detections based on image dimensions.
    Filters them dynamically using the user-defined confidence threshold.
    """
    mock_pool = [
        {
            'box': [width * 0.22, height * 0.35, width * 0.42, height * 0.90],
            'confidence': 0.88,
            'class_name': 'rider',
            'source_model': 'helmet'
        },
        {
            'box': [width * 0.29, height * 0.22, width * 0.38, height * 0.38],
            'confidence': 0.92,
            'class_name': 'faceWithNoHelmet',
            'source_model': 'helmet'
        },
        {
            'box': [width * 0.30, height * 0.72, width * 0.40, height * 0.80],
            'confidence': 0.95,
            'class_name': 'License_Plate',
            'source_model': 'plate',
            'text': 'KA-03-MY-5678'
        },
        {
            'box': [width * 0.15, height * 0.28, width * 0.52, height * 0.95],
            'confidence': 0.82,
            'class_name': 'triple_riding',
            'source_model': 'triple'
        },
        {
            'box': [width * 0.55, height * 0.30, width * 0.92, height * 0.85],
            'confidence': 0.91,
            'class_name': 'car',
            'source_model': 'bengaluru'
        },
        {
            'box': [width * 0.60, height * 0.45, width * 0.75, height * 0.68],
            'confidence': 0.74,
            'class_name': 'person_no_seatbelt',
            'source_model': 'seatbelt'
        }
    ]
    
    # Filter by user conf_threshold
    filtered_detections = [det for det in mock_pool if det['confidence'] >= conf_threshold]
    
    # Separate plates for OCR processing simulation
    plates = [det for det in filtered_detections if det['class_name'] in ['License_Plate', 'numberPlate']]
    
    return filtered_detections, plates

# ----------------------------------------------------
# PIPELINE ORCHESTRATION
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a road/traffic image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image to scan for violations."
)

if uploaded_file is not None:
    try:
        # Load and convert PIL image to NumPy (RGB)
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert("RGB"))
        h, w, _ = image_np.shape
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate file MD5 hash for session state caching
        import hashlib
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Determine if we should reload detections from scratch
        should_run_inference = (
            'cached_hash' not in st.session_state 
            or st.session_state['cached_hash'] != file_hash
            or 'sim_mode_state' not in st.session_state
            or st.session_state['sim_mode_state'] != sim_mode
        )
        
        if should_run_inference:
            st.session_state['cached_hash'] = file_hash
            st.session_state['sim_mode_state'] = sim_mode
            
            # Baseline threshold of 15% to capture all potential detections (e.g. low-conf triple riding)
            baseline_conf = 0.15
            
            if sim_mode:
                raw_detections, raw_plates = generate_simulated_detections(w, h, baseline_conf)
            else:
                with st.spinner("🚀 Running 5 YOLOv8 models in parallel (ThreadPoolExecutor)..."):
                    # Run parallel inference at baseline threshold
                    raw_detections = run_parallel_inference(yolo_models, image_np, baseline_conf)
                
                with st.spinner("🔍 Segmenting license plate regions and running EasyOCR..."):
                    # Deduplicate and OCR process plates
                    raw_detections, raw_plates = process_license_plates(image_np, raw_detections, ocr_reader)
            
            st.session_state['raw_detections'] = raw_detections
            st.session_state['raw_plates'] = raw_plates
            
        # Dynamically filter cached detections based on current slider threshold
        detections = [d for d in st.session_state['raw_detections'] if d['confidence'] >= conf_threshold]
        plates = [p for p in st.session_state['raw_plates'] if p['confidence'] >= conf_threshold]
        
        # Annotate image
        annotated_image = draw_annotations(image_np, detections)
        
        # Compile report DataFrame
        report_df = generate_report(detections, plates, timestamp)
        
        # Filter for actual violations to drive counts
        violations_only = report_df[report_df['Violation Type'].isin(VIOLATION_CLASSES)]
        total_violations = len(violations_only)
        total_detections_count = len(report_df)
        
        # ----------------------------------------------------
        # STATISTICS / METRICS CARDS
        # ----------------------------------------------------
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card info-card">
                    <div class="kpi-value info-value">{total_detections_count}</div>
                    <div class="kpi-label">Total Objects Tracked</div>
                </div>
                <div class="kpi-card danger-card">
                    <div class="kpi-value danger-value">{total_violations}</div>
                    <div class="kpi-label">Active Violations Detected</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # ----------------------------------------------------
        # SIDE-BY-SIDE IMAGE COMPARISON
        # ----------------------------------------------------
        st.markdown('<div class="section-title">🖼️ Incident Visualizer</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='text-align:center; font-weight:600; color:#495057;'>Original Traffic Feed</p>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("<p style='text-align:center; font-weight:600; color:#495057;'>AI Annotated Feed</p>", unsafe_allow_html=True)
            st.image(annotated_image, use_container_width=True)
            
        # ----------------------------------------------------
        # VIOLATIONS LOG / TABLE
        # ----------------------------------------------------
        st.markdown('<div class="section-title">📋 Traffic Violation Register</div>', unsafe_allow_html=True)
        
        if report_df.empty:
            st.success("🟢 No objects or violations detected at the set confidence threshold. Safe driving!")
        else:
            # Display styled report with highlighted violations
            styled_df = report_df.style.apply(highlight_violations, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # ----------------------------------------------------
            # DOWNLOAD EXPORT ACTIONS
            # ----------------------------------------------------
            col_d1, col_d2, col_d3 = st.columns(3)
            
            # Format filename timestamp
            ts_str = timestamp.replace(" ", "_").replace(":", "-")
            
            # 1. Download Annotated Image
            buf = io.BytesIO()
            Image.fromarray(annotated_image).save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            with col_d1:
                st.download_button(
                    label="📥 Download Annotated Image (PNG)",
                    data=img_bytes,
                    file_name=f"bengaluru_traffic_annotated_{ts_str}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            # 2. Download CSV report
            csv_bytes = convert_df_to_csv(report_df)
            with col_d2:
                st.download_button(
                    label="📊 Download Violation Report (CSV)",
                    data=csv_bytes,
                    file_name=f"bengaluru_traffic_violations_{ts_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            # 3. Download JSON report
            json_bytes = convert_df_to_json(report_df)
            with col_d3:
                st.download_button(
                    label="📄 Download Violation Report (JSON)",
                    data=json_bytes,
                    file_name=f"bengaluru_traffic_violations_{ts_str}.json",
                    mime="application/json",
                    use_container_width=True
                )

        # ----------------------------------------------------
        # ANALYTICS DASHBOARD
        # ----------------------------------------------------
        st.markdown('<div class="section-title">📊 Real-Time Enforcement Analytics</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            bar_fig = plot_violation_breakdown(report_df)
            st.plotly_chart(bar_fig, use_container_width=True)
            
        with col_c2:
            dist_fig = plot_confidence_distribution(report_df)
            st.plotly_chart(dist_fig, use_container_width=True)
            
        # ----------------------------------------------------
        # DEVELOPER DIAGNOSTICS EXPANDER
        # ----------------------------------------------------
        st.markdown('<div class="section-title">🛠️ Developer Diagnostics</div>', unsafe_allow_html=True)
        with st.expander("🔍 View Raw Model Detections (Threshold baseline: 15% confidence)"):
            st.markdown(
                "This log displays all candidate detections found by the 5 models. "
                "Move the **Confidence Threshold** slider in the sidebar to dynamically filter them in the main view."
            )
            raw_detections = st.session_state.get('raw_detections', [])
            if raw_detections:
                diag_rows = []
                for rd in raw_detections:
                    status = "✅ Active (Passed Slider)" if rd['confidence'] >= conf_threshold else "⚠️ Filtered (Below Slider)"
                    diag_rows.append({
                        'Class Name': rd['class_name'],
                        'Confidence %': round(rd['confidence'] * 100, 2),
                        'Source Model': rd['source_model'],
                        'Status': status,
                        'Bounding Box': [round(coord, 1) for coord in rd['box']]
                    })
                diag_df = pd.DataFrame(diag_rows)
                # Sort by confidence descending
                diag_df = diag_df.sort_values(by='Confidence %', ascending=False)
                st.dataframe(diag_df, use_container_width=True)
            else:
                st.write("No raw detections found above 15% confidence.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
        st.exception(e)
else:
    # Landing state instructions when no image is uploaded
    st.markdown("""
        <div style="background-color: #f8f9fa; border: 2px dashed #1a237e; border-radius: 12px; padding: 3rem; text-align: center; margin-top: 2rem;">
            <span style="font-size: 4rem;">🚦</span>
            <h3 style="color: #1a237e; font-family: 'Outfit', sans-serif; font-weight: 700; margin-top: 1rem;">Awaiting Traffic Feed Upload</h3>
            <p style="color: #6c757d; max-width: 600px; margin: 0.5rem auto 1.5rem auto; font-size: 1rem;">
                Upload a traffic junction image above. The AI enforcement module will automatically identify vehicles, scan for seatbelts/helmets, read license plates, and flag violations.
            </p>
            <p style="font-size: 0.85rem; color: #868e96; font-weight: 500;">
                💡 No model weights? Enable <strong>Simulation Mode</strong> in the sidebar to view full system functionality!
            </p>
        </div>
    """, unsafe_allow_html=True)
