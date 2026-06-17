import concurrent.futures
import streamlit as st

# Safe import for Hugging Face ZeroGPU spaces decorator
try:
    import spaces
except ImportError:
    # Fallback dummy decorator for local running
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                return lambda f: f
            return func

@spaces.GPU
def run_parallel_inference(models, image_np, conf_threshold=0.5):
    """
    Runs all 5 YOLOv8 models in parallel using ThreadPoolExecutor.
    Gathers all results and merges them into a single list of detections.
    Each detection has: box [x1, y1, x2, y2], confidence, class_name, and source_model.
    """
    raw_results = {}
    
    # Worker function for single model inference
    def run_single_inference(model_name, model):
        if model is None:
            return []
        try:
            # Run YOLOv8 prediction
            # verbose=False suppresses logging in terminal
            preds = model.predict(source=image_np, conf=conf_threshold, verbose=False)
            if len(preds) > 0:
                return preds[0]
        except Exception as e:
            # Handle runtime device or tensor errors
            print(f"Error running inference for {model_name}: {e}")
        return None

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {
            executor.submit(run_single_inference, name, model): name
            for name, model in models.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                raw_results[name] = future.result()
            except Exception as e:
                raw_results[name] = None

    # Merge detections into a single list
    merged_detections = []
    
    for name, result in raw_results.items():
        if result is None or not hasattr(result, 'boxes') or result.boxes is None:
            continue
            
        model_names = result.names
        for box in result.boxes:
            try:
                # Convert coords to list of floats [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                class_name = model_names[cls_id]
                
                merged_detections.append({
                    'box': xyxy,
                    'confidence': conf,
                    'class_name': class_name,
                    'source_model': name
                })
            except Exception as box_err:
                print(f"Error parsing box from model {name}: {box_err}")
                
    return merged_detections
