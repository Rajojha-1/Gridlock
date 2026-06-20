import numpy as np
import cv2
import re

def calculate_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) of two bounding boxes.
    Each box is defined as [x1, y1, x2, y2].
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - intersection_area)

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area

def score_indian_plate(text):
    """
    Scores how closely a string matches the standard Indian license plate format.
    Format: AA DD AA DDDD or AA DD DDDD (where A = letter, D = digit).
    """
    text = text.upper().replace(" ", "").replace("-", "")
    if not text:
        return 0
    
    score = 0
    
    # 1. Length check: standard plates are between 7 and 10 characters
    if 7 <= len(text) <= 10:
        score += 2
        
    # 2. Check if starts with a valid Indian state code (2 letters)
    state_codes = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ", "HR", 
        "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", 
        "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB"
    }
    if text[:2] in state_codes:
        score += 5
    elif text[:2].isalpha():
        score += 2 # starts with 2 letters but not standard state code
        
    # 3. Check if characters 3 and 4 are digits (RTO code)
    if len(text) >= 4 and text[2:4].isdigit():
        score += 4
        
    # 4. Check if the end of the plate is 4 digits
    if len(text) >= 4 and text[-4:].isdigit():
        score += 4
    elif len(text) >= 3 and text[-3:].isdigit():
        score += 2
        
    # 5. Check regex patterns
    # Standard format: State(2) + RTO(2) + Series(1-2) + Number(4)
    if re.match(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$", text):
        score += 10
    # Central/Bharat series format: YY BH DDDD XX
    elif re.match(r"^\d{2}BH\d{4}[A-Z]{1,2}$", text):
        score += 10
    # Format: State(2) + RTO(2) + Number(4) (no letters in between)
    elif re.match(r"^[A-Z]{2}\d{2}\d{4}$", text):
        score += 5
        
    return score

def run_ensemble_ocr(cropped, ocr_reader):
    """
    Runs PaddleOCR on three differently preprocessed versions of the cropped plate image,
    scores each candidate text against standard Indian license plate formats,
    and returns the best candidate text.
    """
    if cropped is None or cropped.size == 0 or ocr_reader is None:
        return "N/A"
        
    candidates = []
    
    try:
        # Preprocess 1: Baseline Grayscale + 3x Upscaling (high resolution)
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        upscaled_baseline = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        upscaled_baseline_rgb = cv2.cvtColor(upscaled_baseline, cv2.COLOR_GRAY2RGB)
        candidates.append((upscaled_baseline_rgb, "baseline"))
        
        # Preprocess 2: CLAHE Contrast Enhancement + Bilateral Filter + 3x Upscaling
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        upscaled_contrast = cv2.resize(denoised, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        upscaled_contrast_rgb = cv2.cvtColor(upscaled_contrast, cv2.COLOR_GRAY2RGB)
        candidates.append((upscaled_contrast_rgb, "contrast"))
        
        # Preprocess 3: Adaptive Thresholding + 3x Upscaling
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        upscaled_thresh = cv2.resize(thresh, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        upscaled_thresh_rgb = cv2.cvtColor(upscaled_thresh, cv2.COLOR_GRAY2RGB)
        candidates.append((upscaled_thresh_rgb, "threshold"))
        
    except Exception as prep_err:
        print(f"Error during OCR preprocessing: {prep_err}")
        # Fallback to standard color crop
        candidates = [(cropped, "raw")]

    ocr_results_list = []
    
    for img_variant, name in candidates:
        try:
            # Predict using EasyOCR readtext API
            results = ocr_reader.readtext(img_variant)
            
            if results and len(results) > 0:
                text_parts = []
                confidences = []
                
                for res in results:
                    # EasyOCR returns list of (bbox, text, confidence)
                    bbox, txt, conf = res
                    if conf > 0.10:
                        text_parts.append(txt)
                        confidences.append(conf)
                
                text = " ".join(text_parts).strip().upper()
                # Keep only alphanumeric, spaces, and hyphens
                cleaned = "".join([c for c in text if c.isalnum() or c in [' ', '-']])
                cleaned = cleaned.strip()
                
                if cleaned:
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    ocr_results_list.append({
                        'text': cleaned,
                        'confidence': avg_conf,
                        'source': name
                    })
        except Exception as ocr_err:
            print(f"EasyOCR variant error ({name}): {ocr_err}")
            
    if not ocr_results_list:
        return "N/A"
        
    # Score each candidate text
    for item in ocr_results_list:
        item['score'] = score_indian_plate(item['text'])
        
    # Sort candidates by plate format score descending, and then by OCR confidence descending
    ocr_results_list.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
    
    # Return the highest scoring / most confident text
    best_candidate = ocr_results_list[0]['text']
    return best_candidate

def process_license_plates(image_np, detections, ocr_reader):
    """
    Extracts plate bounding boxes, filters duplicate boxes (IoU > 0.5),
    crops plate regions from the original image, runs PaddleOCR, and returns:
      1. Updated list of detections (with plate text assigned)
      2. List of deduplicated plate detections
    """
    plate_detections = []
    other_detections = []

    # Separate plate detections ('License_Plate' and 'numberPlate') from others
    for det in detections:
        if det['class_name'] in ['License_Plate', 'numberPlate']:
            plate_detections.append(det)
        else:
            other_detections.append(det)

    # Sort plates by confidence descending
    plate_detections = sorted(plate_detections, key=lambda x: x['confidence'], reverse=True)
    deduplicated_plates = []

    # Non-maximum suppression / deduplication based on IoU
    for p in plate_detections:
        overlap = False
        for kp in deduplicated_plates:
            if calculate_iou(p['box'], kp['box']) > 0.5:
                overlap = True
                break
        if not overlap:
            deduplicated_plates.append(p)

    h, w, _ = image_np.shape

    # Perform Ensemble PaddleOCR on each unique plate
    for p in deduplicated_plates:
        x1, y1, x2, y2 = map(int, p['box'])
        
        # Add a 15% margin padding to crop area for better OCR legibility
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped.size > 0 and ocr_reader is not None:
            p['text'] = run_ensemble_ocr(cropped, ocr_reader)
        else:
            p['text'] = "N/A"

    # Merge deduplicated plates back with other detections
    merged_detections = deduplicated_plates + other_detections
    return merged_detections, deduplicated_plates

def associate_violation_with_plate(violation_box, plates):
    """
    Associates a violation box with the nearest license plate using Euclidean distance.
    Returns the plate's extracted OCR text, or 'N/A' if no plates exist.
    """
    if not plates:
        return "N/A"

    # Compute violation center
    vx_center = (violation_box[0] + violation_box[2]) / 2.0
    vy_center = (violation_box[1] + violation_box[3]) / 2.0

    nearest_text = "N/A"
    min_distance = float('inf')

    for p in plates:
        px_center = (p['box'][0] + p['box'][2]) / 2.0
        py_center = (p['box'][1] + p['box'][3]) / 2.0
        
        # Calculate distance
        distance = np.sqrt((vx_center - px_center)**2 + (vy_center - py_center)**2)
        if distance < min_distance:
            min_distance = distance
            # Get text (default to 'N/A' if missing)
            nearest_text = p.get('text', 'N/A')

    return nearest_text
