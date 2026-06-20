import numpy as np
import uuid

def compile_dossiers(detections, plates):
    """
    Compiles raw object detections and deduplicated plates into unified vehicle dossiers.
    Groups multiple violations per vehicle, computes compound fines, and runs a
    calibrated abstention model (Auto-Enforced, Abstained/Human Review, or Dismissed).
    """
    all_dossiers = []
    
    # 1. Initialize dossiers for all detected plates
    plate_dossiers = {}
    for p in plates:
        plate_text = p.get('text', 'UNKNOWN')
        if not plate_text or plate_text == 'N/A':
            plate_text = f"UNKNOWN-{uuid.uuid4().hex[:6].upper()}"
            
        plate_dossiers[plate_text] = {
            'id': plate_text,
            'type': 'identified',
            'plate_text': plate_text,
            'plate_box': p['box'],
            'plate_confidence': p['confidence'],
            'vehicle_type': 'two-wheeler',  # default, will refine
            'violations': [],
            'vehicle_box': None,
            'state': 'COMPLIANT',  # COMPLIANT, AUTO-ENFORCED, ABSTAINED, DISMISSED, APPROVED, REJECTED
            'compounded_fine': 0,
            'severity': 'LOW',  # LOW, MEDIUM, HIGH, CRITICAL
            'reason': 'Compliant: No violations detected.'
        }
        
    # 2. Find vehicles and associate vehicle type with plates
    vehicles = [d for d in detections if d['class_name'] in [
        'Truck', 'bicycle', 'bus', 'car', 'lcv', 'three-wheeler', 'two-wheeler', 'vehicle', 'rider'
    ]]
    
    # Refine plate dossiers with closest vehicle bounding box
    for plate_text, dos in plate_dossiers.items():
        p_box = dos['plate_box']
        p_center = np.array([(p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0])
        
        best_veh = None
        min_dist = float('inf')
        
        for v in vehicles:
            v_box = v['box']
            v_center = np.array([(v_box[0] + v_box[2]) / 2.0, (v_box[1] + v_box[3]) / 2.0])
            dist = np.linalg.norm(p_center - v_center)
            
            if dist < min_dist:
                min_dist = dist
                best_veh = v
                
        if best_veh and min_dist < 400.0:  # threshold distance
            dos['vehicle_type'] = best_veh['class_name']
            dos['vehicle_box'] = best_veh['box']
            if dos['vehicle_type'] in ['rider', 'two-wheeler', 'vehicle']:
                dos['vehicle_type'] = 'two-wheeler'
            elif dos['vehicle_type'] in ['car', 'bus', 'Truck', 'lcv']:
                dos['vehicle_type'] = 'four-wheeler'
            elif dos['vehicle_type'] == 'three-wheeler':
                dos['vehicle_type'] = 'three-wheeler'
                
    # 3. Separate and associate violations
    violations = []
    violation_classes = {
        'faceWithNoHelmet': 'No Helmet (Rider)',
        'no_helmet': 'No Helmet (Rider)',
        'faceWithBadHelmet': 'Defective Helmet',
        'triple_riding': 'Triple Riding',
        'person_no_seatbelt': 'No Seatbelt (Occupant)'
    }
    
    for d in detections:
        if d['class_name'] in violation_classes:
            violations.append(d)
            
    # Track which violations are associated with identified plates
    associated_violations = set()
    
    for idx, v in enumerate(violations):
        v_box = v['box']
        v_center = np.array([(v_box[0] + v_box[2]) / 2.0, (v_box[1] + v_box[3]) / 2.0])
        
        best_plate = None
        min_dist = float('inf')
        
        for plate_text, dos in plate_dossiers.items():
            p_box = dos['plate_box']
            p_center = np.array([(p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0])
            dist = np.linalg.norm(v_center - p_center)
            if dist < min_dist:
                min_dist = dist
                best_plate = plate_text
                
        # If there is a close plate, associate it
        if best_plate and min_dist < 500.0:
            v_info = {
                'class_name': v['class_name'],
                'label': violation_classes[v['class_name']],
                'confidence': v['confidence'],
                'box': v['box']
            }
            plate_dossiers[best_plate]['violations'].append(v_info)
            associated_violations.add(idx)

    # 4. Group remaining orphan violations into Unidentified Vehicle Dossiers
    orphan_violations = [v for idx, v in enumerate(violations) if idx not in associated_violations]
    
    unidentified_dossiers = []
    while orphan_violations:
        # Take the first orphan violation and cluster nearby violations
        v_current = orphan_violations.pop(0)
        v_c_box = v_current['box']
        v_c_center = np.array([(v_c_box[0] + v_c_box[2]) / 2.0, (v_c_box[1] + v_c_box[3]) / 2.0])
        
        cluster = [v_current]
        
        # Find others in proximity (e.g. within 250 pixels center-to-center)
        i = 0
        while i < len(orphan_violations):
            v_o = orphan_violations[i]
            v_o_box = v_o['box']
            v_o_center = np.array([(v_o_box[0] + v_o_box[2]) / 2.0, (v_o_box[1] + v_o_box[3]) / 2.0])
            if np.linalg.norm(v_c_center - v_o_center) < 250.0:
                cluster.append(orphan_violations.pop(i))
            else:
                i += 1
                
        # Create unidentified dossier
        uid = f"UNIDENTIFIED-{uuid.uuid4().hex[:6].upper()}"
        
        # Infer vehicle type based on violations
        v_types = [vc['class_name'] for vc in cluster]
        inferred_type = 'two-wheeler'
        if 'person_no_seatbelt' in v_types:
            inferred_type = 'four-wheeler'
            
        dos_violations = [{
            'class_name': c['class_name'],
            'label': violation_classes[c['class_name']],
            'confidence': c['confidence'],
            'box': c['box']
        } for c in cluster]
        
        unidentified_dossiers.append({
            'id': uid,
            'type': 'unidentified',
            'plate_text': 'UNIDENTIFIED',
            'plate_box': None,
            'plate_confidence': 0.0,
            'vehicle_type': inferred_type,
            'violations': dos_violations,
            'vehicle_box': v_c_box,  # use one of the boxes
            'state': 'COMPLIANT',
            'compounded_fine': 0,
            'severity': 'LOW',
            'reason': 'Compliant: No violations detected.'
        })
        
    all_dossiers = list(plate_dossiers.values()) + unidentified_dossiers
    
    # 5. Calculate fine, severity, and calibrated decision state for each dossier
    fine_map = {
        'faceWithNoHelmet': 500,
        'no_helmet': 500,
        'faceWithBadHelmet': 500,
        'triple_riding': 1000,
        'person_no_seatbelt': 500
    }
    
    for dos in all_dossiers:
        if not dos['violations']:
            dos['state'] = 'COMPLIANT'
            dos['compounded_fine'] = 0
            dos['severity'] = 'LOW'
            dos['reason'] = 'Compliant: No violations detected.'
            continue
            
        # Calculate compounded fine
        total_fine = 0
        for v in dos['violations']:
            total_fine += fine_map.get(v['class_name'], 500)
        dos['compounded_fine'] = total_fine
        
        # Calculate severity based on number of violations and types
        num_v = len(dos['violations'])
        has_triple = any(v['class_name'] == 'triple_riding' for v in dos['violations'])
        
        if num_v >= 3:
            dos['severity'] = 'CRITICAL'
        elif num_v == 2 or has_triple:
            dos['severity'] = 'HIGH'
        else:
            dos['severity'] = 'MEDIUM'
            
        # Calibrated Abstention logic
        # High threshold = 0.80 (Auto-Enforce)
        # Low threshold = 0.45 (Human review)
        # If any violation falls below 0.80, it requires human verification (Abstained)
        # If all violations fall below 0.45, it is automatically dismissed
        confidences = [v['confidence'] for v in dos['violations']]
        max_conf = max(confidences) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0
        
        if min_conf >= 0.80:
            dos['state'] = 'AUTO-ENFORCED'
            dos['reason'] = f"Autonomous Challan: All active violations exceed the 80% threshold (Min confidence: {min_conf*100:.1f}%)."
        elif max_conf >= 0.45:
            dos['state'] = 'ABSTAINED'
            dos['reason'] = f"Held for Audit: Violation confidence ({min_conf*100:.1f}%) is below the 80% auto-enforce threshold, but exceeds the 45% dismissal threshold (Max confidence: {max_conf*100:.1f}%)."
        else:
            dos['state'] = 'DISMISSED'
            dos['reason'] = f"Dismissed: All active violations fall below the 45% confidence threshold (Max confidence: {max_conf*100:.1f}%)."
            
    return all_dossiers
