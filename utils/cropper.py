import numpy as np
import cv2
from config import VEHICLE_CLASSES

BODY_CLASSES = VEHICLE_CLASSES.union({'rider'})

def crop_image_box(image_np, box, pad_pct=0.0):
    """
    Crops a bounding box from the image, with optional relative padding.
    box: list/tuple [x1, y1, x2, y2]
    """
    if image_np is None or box is None:
        return None

    h, w, _ = image_np.shape
    x1, y1, x2, y2 = map(int, box)

    if pad_pct > 0:
        pw = int((x2 - x1) * pad_pct)
        ph = int((y2 - y1) * pad_pct)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
    else:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

    if x2 > x1 and y2 > y1:
        return image_np[y1:y2, x1:x2]
    return None

def find_associated_plate_detection(violation_box, plates):
    """
    Finds the nearest plate detection using Euclidean distance of centers.
    """
    if not plates:
        return None

    vx_center = (violation_box[0] + violation_box[2]) / 2.0
    vy_center = (violation_box[1] + violation_box[3]) / 2.0

    nearest_plate = None
    min_distance = float('inf')

    for p in plates:
        px_center = (p['box'][0] + p['box'][2]) / 2.0
        py_center = (p['box'][1] + p['box'][3]) / 2.0

        distance = np.sqrt((vx_center - px_center) ** 2 + (vy_center - py_center) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_plate = p

    return nearest_plate

def get_box_overlap(box1, box2):
    """
    Calculates overlap area between box1 and box2.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def find_associated_vehicle_body(violation_box, detections):
    """
    Finds the vehicle or rider body associated with a violation.
    Prioritizes the body that has the maximum overlap with the violation.
    If no bodies overlap, finds the closest one by Euclidean distance of centers.
    """
    body_detections = [d for d in detections if d['class_name'] in BODY_CLASSES]
    if not body_detections:
        return None

    # Step 1: Check overlap
    overlapping_bodies = []
    for b in body_detections:
        overlap = get_box_overlap(violation_box, b['box'])
        if overlap > 0:
            overlapping_bodies.append((overlap, b))

    if overlapping_bodies:
        # Sort by overlap area descending and return the highest overlap
        overlapping_bodies.sort(key=lambda x: x[0], reverse=True)
        return overlapping_bodies[0][1]

    # Step 2: Fallback to center distance
    vx_center = (violation_box[0] + violation_box[2]) / 2.0
    vy_center = (violation_box[1] + violation_box[3]) / 2.0

    nearest_body = None
    min_distance = float('inf')

    for b in body_detections:
        bx_center = (b['box'][0] + b['box'][2]) / 2.0
        by_center = (b['box'][1] + b['box'][3]) / 2.0

        distance = np.sqrt((vx_center - bx_center) ** 2 + (vy_center - by_center) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_body = b

    # If the closest body is extremely far away, it might not be related
    # But let's return it as the best candidate
    return nearest_body

def get_violation_union_box(violation_box, body_box, image_shape):
    """
    Unions the violation box and the body box to capture the person with their vehicle.
    Clips coordinates to image boundaries.
    """
    h, w, _ = image_shape
    vx1, vy1, vx2, vy2 = violation_box

    if body_box is None:
        # Fallback: pad the violation box by 50%
        pw = (vx2 - vx1) * 0.5
        ph = (vy2 - vy1) * 0.5
        x1 = max(0, vx1 - pw)
        y1 = max(0, vy1 - ph)
        x2 = min(w, vx2 + pw)
        y2 = min(h, vy2 + ph)
        return [x1, y1, x2, y2]

    bx1, by1, bx2, by2 = body_box['box']
    ux1 = max(0, min(vx1, bx1))
    uy1 = max(0, min(vy1, by1))
    ux2 = min(w, max(vx2, bx2))
    uy2 = min(h, max(vy2, by2))

    return [ux1, uy1, ux2, uy2]

def generate_plate_placeholder(text="PLATE NOT DETECTED", width=240, height=120):
    """
    Generates a clean gray placeholder image.
    """
    # Create a light gray background
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    # Draw double border for aesthetics
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (220, 220, 220), -1)
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (200, 200, 200), 1)

    # Put Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (width - tw) // 2
    ty = (height + th) // 2

    # Draw centered text
    cv2.putText(img, text, (tx, ty), font, font_scale, (120, 120, 120), thickness, cv2.LINE_AA)
    return img
