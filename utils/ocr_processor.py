import numpy as np
import cv2

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

def process_license_plates(image_np, detections, ocr_reader):
    """
    Extracts plate bounding boxes, filters duplicate boxes (IoU > 0.5),
    crops plate regions from the original image, runs OCR reader, and returns:
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

    # Perform OCR on each unique plate
    for p in deduplicated_plates:
        x1, y1, x2, y2 = map(int, p['box'])
        
        # Add a 5% margin padding to crop area for better OCR legibility
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped.size > 0 and ocr_reader is not None:
            try:
                # Preprocess for OCR: convert RGB to Grayscale
                gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                # Resize to double the size (2x upscale) using cubic interpolation for higher resolution text
                upscaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                
                # EasyOCR returns list of: (bbox, text, conf)
                ocr_results = ocr_reader.readtext(upscaled)
                if ocr_results:
                    # Filter items with lower confidence (> 0.10) to preserve characters
                    text_parts = [res[1] for res in ocr_results if res[2] > 0.10]
                    text = " ".join(text_parts).strip().upper()
                    
                    # Retain only uppercase letters, digits, spaces, and hyphens
                    cleaned_text = "".join([c for c in text if c.isalnum() or c in [' ', '-']])
                    p['text'] = cleaned_text if cleaned_text.strip() else "N/A"
                else:
                    p['text'] = "N/A"
            except Exception as e:
                print(f"EasyOCR Error: {e}")
                p['text'] = "N/A"
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
