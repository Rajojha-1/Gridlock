import cv2
import numpy as np
from config import COLOR_MAP, VEHICLE_CLASSES, DEFAULT_BOX_COLOR

def draw_annotations(image_np, detections):
    """
    Draws colored bounding boxes and text labels onto the image.
    Works directly in RGB color space since PIL and Streamlit expect RGB.
    """
    annotated = image_np.copy()
    h, w, _ = annotated.shape

    # Calculate thickness and font scale proportional to image dimensions
    thickness = max(2, int(min(h, w) * 0.003))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(h, w) * 0.0006)
    text_thickness = max(1, int(thickness / 2))

    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        class_name = det['class_name']
        conf = det['confidence']

        # Determine bounding box color
        if class_name in COLOR_MAP:
            color = COLOR_MAP[class_name]
        elif class_name in VEHICLE_CLASSES:
            color = (255, 255, 255)  # White
        else:
            color = DEFAULT_BOX_COLOR

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label: e.g. "faceWithNoHelmet 89.2%"
        label = f"{class_name} {conf * 100:.1f}%"
        
        # If OCR text is available for this plate, append it
        if 'text' in det and det['text'] and det['text'] != 'N/A':
            label += f" [{det['text']}]"

        # Calculate text dimensions
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Define text background rectangle (make sure it stays inside image bounds)
        label_y1 = max(0, y1 - text_h - 10)
        label_y2 = y1
        label_x1 = x1
        label_x2 = min(w, x1 + text_w + 10)

        # Draw background label box
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # Calculate luminance of background box to choose contrasting text color (black vs white)
        # Luminance = 0.299 * R + 0.587 * G + 0.114 * B
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        # Draw text label slightly padded
        cv2.putText(
            annotated, 
            label, 
            (x1 + 5, y1 - 5), 
            font, 
            font_scale, 
            text_color, 
            text_thickness, 
            lineType=cv2.LINE_AA
        )

    return annotated
