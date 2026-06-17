import pandas as pd
from config import VIOLATION_CLASSES
from utils.ocr_processor import associate_violation_with_plate

def generate_report(detections, plates, timestamp):
    """
    Creates a Pandas DataFrame containing traffic detections.
    Excludes raw plates from the row listings since they are associated as columns,
    preventing duplicate plate-only entries.
    """
    rows = []
    
    for det in detections:
        class_name = det['class_name']
        
        # Skip raw plates themselves in the table rows to keep the log clean,
        # since plate text is already displayed in its own column for other detections.
        if class_name in ['License_Plate', 'numberPlate']:
            continue
            
        conf_pct = round(det['confidence'] * 100, 2)
        plate_text = associate_violation_with_plate(det['box'], plates)
        
        rows.append({
            'Violation Type': class_name,
            'Confidence %': conf_pct,
            'Plate Number': plate_text,
            'Timestamp': timestamp
        })
        
    # If no detections exist, return an empty DataFrame with target columns
    if not rows:
        return pd.DataFrame(columns=['Violation Type', 'Confidence %', 'Plate Number', 'Timestamp'])
        
    df = pd.DataFrame(rows)
    # Sort detections so violations appear at the top
    df['is_violation'] = df['Violation Type'].apply(lambda x: x in VIOLATION_CLASSES)
    df = df.sort_values(by=['is_violation', 'Confidence %'], ascending=[False, False])
    df = df.drop(columns=['is_violation'])
    return df

def highlight_violations(row):
    """
    Pandas styling function to highlight rows representing traffic violations in red.
    """
    is_violation = row['Violation Type'] in VIOLATION_CLASSES
    if is_violation:
        # Subtle light-red background with red border accent
        return ['background-color: rgba(211, 47, 47, 0.15); color: #d32f2f; font-weight: bold;'] * len(row)
    else:
        return [''] * len(row)

def convert_df_to_csv(df):
    """
    Converts DataFrame to CSV bytes for Streamlit download.
    """
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_json(df):
    """
    Converts DataFrame to JSON bytes for Streamlit download.
    """
    return df.to_json(orient='records', indent=4).encode('utf-8')
