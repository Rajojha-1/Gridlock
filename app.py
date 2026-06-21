import os
import io
import time
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Environment Check: True if running locally, False if deployed to Hugging Face Spaces
is_local = "SPACE_ID" not in os.environ and os.environ.get("SYSTEM") != "spaces"

def get_image_base64(img_np):
    """
    Converts a NumPy RGB image into a base64 JPEG string for inline HTML rendering.
    """
    if img_np is None or img_np.size == 0:
        return ""
    try:
        buffered = io.BytesIO()
        pil_img = Image.fromarray(img_np.astype('uint8'))
        pil_img.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return ""

# Import configuration and CSS styling
import config
from config import CUSTOM_CSS, VIOLATION_CLASSES, PRIMARY_NAVY, ACCENT_YELLOW, VIOLATION_LABELS, VIOLATION_CHALLANS

# Import modular helper utilities
from utils.model_loader import load_yolo_models, load_ocr_reader
from utils.inference import run_parallel_inference
from utils.ocr_processor import process_license_plates
from utils.dossier_engine import compile_dossiers
from utils.annotator import draw_annotations
from utils.report_generator import generate_report, highlight_violations, convert_df_to_csv, convert_df_to_json
from utils.visualizer import plot_violation_breakdown, plot_confidence_distribution
from utils.cropper import (
    crop_image_box, find_associated_plate_detection, 
    find_associated_vehicle_body, get_violation_union_box, 
    generate_plate_placeholder
)

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

# Check model weights existence (fast disk check, no model load)
model_paths = {
    'helmet': 'models/helmet.pt',
    'plate': 'models/plate.pt',
    'triple': 'models/triple.pt',
    'seatbelt': 'models/seatbelt.pt',
    'bengaluru': 'models/bengaluru.pt'
}
weights_exist = {name: os.path.exists(path) for name, path in model_paths.items()}
all_models_present = all(weights_exist.values())

# Display status of each YOLO model
st.sidebar.markdown("**YOLOv8 Models Status:**")
model_names = {
    'helmet': 'Helmet & Rider Model',
    'plate': 'License Plate Model',
    'triple': 'Triple Riding Model',
    'seatbelt': 'Seatbelt Model',
    'bengaluru': 'Bengaluru Vehicle Model'
}

for key, label in model_names.items():
    if weights_exist.get(key):
        st.sidebar.markdown(f"✅ {label}: `Ready`")
    else:
        st.sidebar.markdown(f"❌ {label}: `Weights Missing`")

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
    Generates a set of realistic mock detections featuring compound violations,
    varying confidence levels, and compliant cases.
    """
    mock_pool = [
        # Vehicle 1: Motorcycle KA-03-MY-5678 (Compound Violation: No Helmet + Triple Riding)
        {
            'box': [width * 0.15, height * 0.35, width * 0.45, height * 0.90],
            'confidence': 0.88,
            'class_name': 'rider',
            'source_model': 'helmet'
        },
        {
            'box': [width * 0.25, height * 0.20, width * 0.35, height * 0.35],
            'confidence': 0.92,
            'class_name': 'faceWithNoHelmet',
            'source_model': 'helmet'
        },
        {
            'box': [width * 0.28, height * 0.70, width * 0.38, height * 0.78],
            'confidence': 0.95,
            'class_name': 'License_Plate',
            'source_model': 'plate',
            'text': 'KA-03-MY-5678'
        },
        {
            'box': [width * 0.12, height * 0.30, width * 0.48, height * 0.92],
            'confidence': 0.65, # Borderline confidence -> Will trigger Abstained state
            'class_name': 'triple_riding',
            'source_model': 'triple'
        },
        
        # Vehicle 2: Car KA-51-EF-1234 (Single Violation: Seatbelt)
        {
            'box': [width * 0.50, height * 0.25, width * 0.90, height * 0.85],
            'confidence': 0.91,
            'class_name': 'car',
            'source_model': 'bengaluru'
        },
        {
            'box': [width * 0.55, height * 0.75, width * 0.68, height * 0.83],
            'confidence': 0.94,
            'class_name': 'License_Plate',
            'source_model': 'plate',
            'text': 'KA-51-EF-1234'
        },
        {
            'box': [width * 0.58, height * 0.40, width * 0.72, height * 0.65],
            'confidence': 0.89, # High confidence -> Will trigger Auto-Enforce
            'class_name': 'person_no_seatbelt',
            'source_model': 'seatbelt'
        },
        
        # Vehicle 3: Motorcycle KA-04-AB-9012 (Fully Compliant)
        {
            'box': [width * 0.02, height * 0.40, width * 0.20, height * 0.80],
            'confidence': 0.96,
            'class_name': 'two-wheeler',
            'source_model': 'bengaluru'
        },
        {
            'box': [width * 0.08, height * 0.72, width * 0.16, height * 0.78],
            'confidence': 0.93,
            'class_name': 'License_Plate',
            'source_model': 'plate',
            'text': 'KA-04-AB-9012'
        },
        {
            'box': [width * 0.08, height * 0.32, width * 0.14, height * 0.45],
            'confidence': 0.94,
            'class_name': 'faceWithGoodHelmet',
            'source_model': 'helmet'
        },
        
        # Vehicle 4: Unidentified Vehicle (Low-confidence Violation -> Auto-Dismissed)
        {
            'box': [width * 0.40, height * 0.50, width * 0.48, height * 0.62],
            'confidence': 0.38, # Below low threshold -> Will trigger Auto-Dismiss
            'class_name': 'no_helmet',
            'source_model': 'triple'
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
            or 'raw_detections' not in st.session_state
            or 'raw_plates' not in st.session_state
        )
        
        # Initialize manual states dictionary in session state if not present
        if 'dossier_manual_states' not in st.session_state:
            st.session_state['dossier_manual_states'] = {}

        if should_run_inference:
            # Baseline threshold of 15% to capture all potential detections (e.g. low-conf triple riding)
            baseline_conf = 0.15
            
            if sim_mode:
                raw_detections, raw_plates = generate_simulated_detections(w, h, baseline_conf)
            else:
                if 'models_loaded' not in st.session_state:
                    with st.spinner("Loading 5 YOLOv8 models"):
                        yolo_models = load_yolo_models()
                    st.session_state['models_loaded'] = True
                else:
                    yolo_models = load_yolo_models()
                
                with st.spinner("Running parallel inference"):
                    # Run parallel inference at baseline threshold
                    raw_detections = run_parallel_inference(yolo_models, image_np, baseline_conf)
                
                if 'ocr_loaded' not in st.session_state:
                    with st.spinner("Initializing EasyOCR"):
                        ocr_reader = load_ocr_reader()
                    st.session_state['ocr_loaded'] = True
                else:
                    ocr_reader = load_ocr_reader()
                
                with st.spinner("Segmenting license plate regions and running EasyOCR..."):
                    # Deduplicate and OCR process plates
                    raw_detections, raw_plates = process_license_plates(image_np, raw_detections, ocr_reader)
            
            # Save to session state only after successful run to avoid partial initialization bugs
            st.session_state['raw_detections'] = raw_detections
            st.session_state['raw_plates'] = raw_plates
            st.session_state['cached_hash'] = file_hash
            st.session_state['sim_mode_state'] = sim_mode
            st.session_state['dossiers'] = compile_dossiers(raw_detections, raw_plates)
            st.session_state['dossier_manual_states'] = {}
            
        # Dynamically filter cached detections based on current slider threshold
        detections = [d for d in st.session_state['raw_detections'] if d['confidence'] >= conf_threshold]
        plates = [p for p in st.session_state['raw_plates'] if p['confidence'] >= conf_threshold]
        
        # Annotate image
        annotated_image = draw_annotations(image_np, detections)
        
        # Compile report DataFrame
        report_df = generate_report(detections, plates, timestamp)
        # Compiles/filters dossiers on the fly
        all_dossiers = []
        for d in st.session_state.get('dossiers', []):
            # Filter violations in this dossier based on confidence slider
            filtered_violations = [v for v in d['violations'] if v['confidence'] >= conf_threshold]
            
            # Skip unidentified dossiers that have no active violations at the current confidence slider
            if not filtered_violations and d['plate_text'] == 'UNIDENTIFIED':
                continue
                
            # Create a copy so we don't modify the session state original directly
            d_copy = d.copy()
            d_copy['violations'] = filtered_violations
            
            # Recalculate fine and severity based on filtered violations
            fine_map = {
                'faceWithNoHelmet': 500,
                'no_helmet': 500,
                'faceWithBadHelmet': 500,
                'triple_riding': 1000,
                'person_no_seatbelt': 500
            }
            total_fine = sum(fine_map.get(v['class_name'], 500) for v in filtered_violations)
            d_copy['compounded_fine'] = total_fine
            
            # Calculate severity
            num_v = len(filtered_violations)
            has_triple = any(v['class_name'] == 'triple_riding' for v in filtered_violations)
            if num_v >= 3:
                d_copy['severity'] = 'CRITICAL'
            elif num_v == 2 or has_triple:
                d_copy['severity'] = 'HIGH'
            elif num_v == 1:
                d_copy['severity'] = 'MEDIUM'
            else:
                d_copy['severity'] = 'LOW'
                
            # Apply manual state override if present, otherwise recalculate calibrated state
            d_id = d['id']
            if 'dossier_manual_states' not in st.session_state:
                st.session_state['dossier_manual_states'] = {}
                
            if d_id in st.session_state['dossier_manual_states']:
                d_copy['state'] = st.session_state['dossier_manual_states'][d_id]
                d_copy['reason'] = f"Manual Triage: Audited and {d_copy['state'].lower()} by Traffic Officer."
            else:
                # Recalculate state based on current filtered violations
                if not filtered_violations:
                    d_copy['state'] = 'COMPLIANT'
                    d_copy['reason'] = 'Compliant: No violations detected.'
                else:
                    confidences = [v['confidence'] for v in filtered_violations]
                    max_conf = max(confidences) if confidences else 0.0
                    min_conf = min(confidences) if confidences else 0.0
                    
                    if min_conf >= 0.80:
                        d_copy['state'] = 'AUTO-ENFORCED'
                        d_copy['reason'] = f"Autonomous Challan: All active violations exceed the 80% threshold (Min confidence: {min_conf*100:.1f}%)."
                    elif max_conf >= 0.45:
                        d_copy['state'] = 'ABSTAINED'
                        d_copy['reason'] = f"Held for Audit: Violation confidence ({min_conf*100:.1f}%) is below the 80% auto-enforce threshold, but exceeds the 45% dismissal threshold (Max confidence: {max_conf*100:.1f}%)."
                    else:
                        d_copy['state'] = 'DISMISSED'
                        d_copy['reason'] = f"Dismissed: All active violations fall below the 45% confidence threshold (Max confidence: {max_conf*100:.1f}%)."
            
            all_dossiers.append(d_copy)

        # Calculate counts
        total_vehicles_tracked = len(all_dossiers)
        active_violations_dossiers = [d for d in all_dossiers if d['state'] in ['AUTO-ENFORCED', 'ABSTAINED', 'APPROVED']]
        total_violations_count = len(active_violations_dossiers)

        # ----------------------------------------------------
        # STATISTICS / METRICS CARDS
        # ----------------------------------------------------
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card info-card">
                    <div class="kpi-value info-value">{total_vehicles_tracked}</div>
                    <div class="kpi-label">Vehicles Tracked</div>
                </div>
                <div class="kpi-card danger-card">
                    <div class="kpi-value danger-value">{total_violations_count}</div>
                    <div class="kpi-label">Active Violation Cases</div>
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
            try:
                st.image(image, use_container_width=True)
            except TypeError:
                st.image(image, use_column_width=True)
            
        with col2:
            st.markdown("<p style='text-align:center; font-weight:600; color:#495057;'>AI Annotated Feed</p>", unsafe_allow_html=True)
            try:
                st.image(annotated_image, use_container_width=True)
            except TypeError:
                st.image(annotated_image, use_column_width=True)

        # ----------------------------------------------------
        # TABBED OPERATIONS PANEL
        # ----------------------------------------------------
        tab_dossiers, tab_review, tab_analytics = st.tabs([
            "🚨 Active Incident Dossiers",
            "👮 Officer Review Station",
            "📊 Portal Performance & Metrics"
        ])

        with tab_dossiers:
            st.markdown('<div class="section-title" style="margin-top:0;">🚨 Traffic Violation Dossier Grid</div>', unsafe_allow_html=True)
            st.markdown(
                "Unified case files for vehicles flagged with single or compound infractions. "
                "High-confidence detections are automatically enforced, while borderline cases are held for review."
            )
            
            # We want to display all dossiers that have violations (meaning state is not COMPLIANT)
            incident_dossiers = [d for d in all_dossiers if d['state'] != 'COMPLIANT']
            
            if not incident_dossiers:
                st.success("🟢 No incident dossiers compiled! All vehicles in this junction are compliant.")
            else:
                st.write("") # spacing
                
                # Setup columns for card grid layout
                d_cols = st.columns(3)
                for d_idx, d in enumerate(incident_dossiers):
                    col_target = d_cols[d_idx % 3]
                    
                    with col_target:
                        # Extract crops
                        veh_crop = None
                        if d['vehicle_box'] is not None:
                            veh_crop = crop_image_box(image_np, d['vehicle_box'])
                        elif d['violations']:
                            veh_crop = crop_image_box(image_np, d['violations'][0]['box'], pad_pct=0.25)
                            
                        plate_crop = None
                        if d['plate_box'] is not None:
                            plate_crop = crop_image_box(image_np, d['plate_box'], pad_pct=0.05)
                        else:
                            plate_crop = generate_plate_placeholder("NO PLATE DETECTED")
                            
                        # Encode to base64
                        veh_b64 = get_image_base64(veh_crop)
                        plate_b64 = get_image_base64(plate_crop)
                        
                        # Pills HTML
                        pills_html = ""
                        for v in d['violations']:
                            v_cls = v['class_name'].lower()
                            pill_class = 'helmet-pill' if 'helmet' in v_cls else 'triple-pill' if 'triple' in v_cls else 'seatbelt-pill'
                            pills_html += f'<span class="violation-pill {pill_class}">{v["label"]} ({v["confidence"]*100:.1f}%)</span>'
                            
                        # Badge mapping
                        badge_map = {
                            'AUTO-ENFORCED': 'badge-enforced',
                            'ABSTAINED': 'badge-abstained',
                            'APPROVED': 'badge-approved',
                            'REJECTED': 'badge-rejected',
                            'DISMISSED': 'badge-dismissed'
                        }
                        badge_style = badge_map.get(d['state'], 'badge-dismissed')
                        
                        st.markdown(f"""
                            <div class="dossier-card">
                                <div>
                                    <div class="dossier-header">
                                        <span class="dossier-title">{d['vehicle_type'].replace('-', ' ').upper()}</span>
                                        <span class="severity-tag {d['severity'].lower()}">{d['severity']}</span>
                                    </div>
                                    <div class="dossier-body">
                                        <div style="display: flex; gap: 10px; margin-bottom: 12px;">
                                            <div style="flex: 1.2; text-align: center;">
                                                <img src="{veh_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #e9ecef; aspect-ratio: 4/3; object-fit: cover;" />
                                                <span style="font-size: 0.65rem; color: #6c757d; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Vehicle Evidence</span>
                                            </div>
                                            <div style="flex: 1; text-align: center;">
                                                <img src="{plate_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #e9ecef; aspect-ratio: 4/3; object-fit: cover;" />
                                                <span style="font-size: 0.65rem; color: #6c757d; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">License Plate</span>
                                            </div>
                                        </div>
                                        <div style="margin-top: 10px; font-size:0.9rem;">
                                            <strong>Plate Number:</strong> <span class="violation-plate-val">{d['plate_text']}</span>
                                        </div>
                                        <div class="violation-pill-list" style="margin-top:8px; margin-bottom:8px;">
                                            {pills_html}
                                        </div>
                                        <div style="margin-top: 10px; font-size: 0.95rem;">
                                            <strong>Compounded Fine:</strong> <span style="color: #d32f2f; font-weight:700;">₹{d['compounded_fine']:,}</span>
                                        </div>
                                        <div style="margin-top: 8px; font-size: 0.8rem; color: #6c757d; line-height: 1.3;">
                                            <strong>Decision Reason:</strong> {d.get('reason', 'N/A')}
                                        </div>
                                    </div>
                                </div>
                                <div class="dossier-footer">
                                    <span class="status-badge {badge_style}">{d['state'].replace('-', ' ')}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

        with tab_review:
            st.markdown('<div class="section-title" style="margin-top:0;">👮 Interactive Officer Review Station</div>', unsafe_allow_html=True)
            st.markdown(
                "Review borderline or ambiguous detections flagged by the calibrated abstention engine. "
                "Manual approvals immediately update the incident dossier state and generate official challans."
            )
            
            abstained_cases = [d for d in all_dossiers if d['state'] == 'ABSTAINED']
            
            if not abstained_cases:
                st.success("🟢 **No pending cases!** All incident reports have been successfully triaged.")
            else:
                for d_idx, d in enumerate(abstained_cases):
                    d_id = d['id']
                    
                    st.markdown(f"""
                        <div class="review-panel">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f3f5; padding-bottom: 8px; margin-bottom: 15px;">
                                <h4 style="margin: 0; color: #1a237e; font-family:'Outfit',sans-serif; font-weight:700;">
                                    Case File: #{d_id} ({d['vehicle_type'].replace('-', ' ').title()})
                                </h4>
                                <span class="severity-tag {d['severity'].lower()}">Severity: {d['severity']}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_ev, col_det = st.columns([6, 4])
                    
                    veh_crop = None
                    if d['vehicle_box'] is not None:
                        veh_crop = crop_image_box(image_np, d['vehicle_box'])
                    elif d['violations']:
                        veh_crop = crop_image_box(image_np, d['violations'][0]['box'], pad_pct=0.25)
                        
                    plate_crop = None
                    if d['plate_box'] is not None:
                        plate_crop = crop_image_box(image_np, d['plate_box'], pad_pct=0.05)
                    else:
                        plate_crop = generate_plate_placeholder("NO PLATE DETECTED")
                        
                    with col_ev:
                        c1, c2 = st.columns(2)
                        with c1:
                            if veh_crop is not None:
                                st.image(veh_crop, caption="Visual Evidence", use_column_width=True)
                            else:
                                st.info("No vehicle crop available")
                        with c2:
                            if plate_crop is not None:
                                st.image(plate_crop, caption="Plate Bounding Crop", use_column_width=True)
                            else:
                                st.info("No plate crop available")
                                
                    with col_det:
                        st.markdown(f"**Plate Number:** <span class='violation-plate-val'>{d['plate_text']}</span>", unsafe_allow_html=True)
                        st.markdown("**Infractions under Auditing:**")
                        for v in d['violations']:
                            st.markdown(f"- ⚠️ **{v['label']}** (Confidence score: `{v['confidence']*100:.1f}%`)")
                            
                        st.markdown(f"**Compounded Challan Fine:** <span style='color:#d32f2f; font-weight:700; font-size:1.15rem;'>₹{d['compounded_fine']:,}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Audit Trigger Reason:** {d.get('reason', 'N/A')}")
                        st.write("") # spacing
                        
                        b_approve, b_dismiss = st.columns(2)
                        with b_approve:
                            if st.button("👮 Approve Challan", key=f"app_{d_id}_{d_idx}", use_container_width=True):
                                st.session_state['dossier_manual_states'][d_id] = 'APPROVED'
                                st.success(f"Challan approved for {d['plate_text']}!")
                                time.sleep(0.4)
                                st.rerun()
                        with b_dismiss:
                            if st.button("❌ Reject / Dismiss", key=f"dism_{d_id}_{d_idx}", use_container_width=True):
                                st.session_state['dossier_manual_states'][d_id] = 'REJECTED'
                                st.warning(f"Case {d_id} dismissed.")
                                time.sleep(0.4)
                                st.rerun()

        with tab_analytics:
            st.markdown('<div class="section-title" style="margin-top:0;">📊 Real-Time Enforcement Analytics</div>', unsafe_allow_html=True)
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                bar_fig = plot_violation_breakdown(report_df)
                st.plotly_chart(bar_fig, use_container_width=True)
                
            with col_c2:
                dist_fig = plot_confidence_distribution(report_df)
                st.plotly_chart(dist_fig, use_container_width=True)
                
            # Exporters section inside Analytics tab
            st.markdown('<div class="section-title">📥 Operations Export Portal</div>', unsafe_allow_html=True)
            col_d1, col_d2, col_d3 = st.columns(3)
            
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
                
            # Add Local Performance Metrics here
            if is_local:
                st.markdown('<div class="section-title">🛠️ Developer Diagnostics</div>', unsafe_allow_html=True)
                with st.expander("🔍 View Raw Model Detections (Threshold baseline: 15% confidence)"):
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
