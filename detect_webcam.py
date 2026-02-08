"""
Face Mask Detection - DNN Face Detector Version
Uses deep learning based face detector instead of Haar Cascade.
This will detect faces better, but mask classification still depends on training data.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'face_mask_detector.h5'

# DNN face detector model files
# These need to be downloaded (script will try to use them if available)
DNN_PROTO = 'deploy.prototxt'
DNN_MODEL = 'res10_300x300_ssd_iter_140000.caffemodel'

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.3

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CATEGORIES = ['with_mask', 'without_mask']
COLORS = {
    'with_mask': (0, 255, 0),
    'without_mask': (0, 0, 255)
}

SCREENSHOT_DIR = 'screenshots'
DEBUG_MODE = True

# Face detection confidence threshold
FACE_DETECTION_CONFIDENCE = 0.5


# ============================================================================
# DOWNLOAD DNN FACE DETECTOR FILES
# ============================================================================

def download_face_detector_files():
    """
    Download DNN face detector model files if not present.
    Uses OpenCV's DNN face detector (Caffe model).
    """

    print("\nChecking for DNN face detector files...")

    # URLs for the model files
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    files_downloaded = False

    # Check and download prototxt file
    if not os.path.exists(DNN_PROTO):
        print(f"Downloading {DNN_PROTO}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(proto_url, DNN_PROTO)
            print(f"[PASS] Downloaded {DNN_PROTO}")
            files_downloaded = True
        except Exception as e:
            print(f"[FAIL] Could not download {DNN_PROTO}: {e}")
            return False

    # Check and download model file
    if not os.path.exists(DNN_MODEL):
        print(f"Downloading {DNN_MODEL} (this may take a minute, ~10MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, DNN_MODEL)
            print(f"[PASS] Downloaded {DNN_MODEL}")
            files_downloaded = True
        except Exception as e:
            print(f"[FAIL] Could not download {DNN_MODEL}: {e}")
            return False

    if not files_downloaded:
        print("[PASS] DNN face detector files already present")

    return True


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models():
    """
    Load DNN face detector and mask classification model.

    DNN Face Detector:
    - Uses deep learning (Caffe framework)
    - More accurate than Haar Cascade
    - Better with different angles and lighting
    - Slower but still real-time capable

    Returns:
        face_net: DNN face detector
        model: Mask classification model
    """

    print("Loading models...")

    # Download DNN files if needed
    if not download_face_detector_files():
        print("[FAIL] Could not get DNN face detector files")
        return None, None

    # Load DNN face detector
    # readNetFromCaffe loads a Caffe model
    # This model was trained on face detection
    try:
        face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("[PASS] DNN face detector loaded")
    except Exception as e:
        print(f"[FAIL] Error loading DNN face detector: {e}")
        return None, None

    # Load mask classification model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("[PASS] Mask classification model loaded")

    return face_net, model


# ============================================================================
# DETECT FACES USING DNN
# ============================================================================

def detect_faces_dnn(frame, face_net):
    """
    Detect faces using DNN instead of Haar Cascade.

    Process:
    1. Prepare blob from image (preprocessing for DNN)
    2. Pass through network
    3. Get detections with confidence scores
    4. Filter by confidence threshold

    Args:
        frame: Input image
        face_net: DNN face detector

    Returns:
        faces: List of (x, y, w, h) tuples
    """

    h, w = frame.shape[:2]

    # Create blob from image
    # blobFromImage does:
    # - Resize to 300x300 (what the model expects)
    # - Scale pixel values by 1.0
    # - Subtract mean values (104, 117, 123) for normalization
    # - Swap R and B channels (BGR to RGB)
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 117.0, 123.0)
    )

    # Pass blob through network
    face_net.setInput(blob)

    # Get detections
    # Output shape: [1, 1, N, 7]
    # Each detection has 7 values:
    # [0-1]: batch and class (always 0)
    # [2]: confidence score
    # [3-6]: bounding box coordinates (normalized 0-1)
    detections = face_net.forward()

    faces = []

    # Process each detection
    for i in range(detections.shape[2]):
        # Get confidence
        confidence = detections[0, 0, i, 2]

        # Filter by confidence threshold
        if confidence > FACE_DETECTION_CONFIDENCE:
            # Get bounding box coordinates
            # These are normalized (0-1), so we multiply by image dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Convert to (x, y, w, h) format
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            faces.append((x, y, w, h))

    return faces


# ============================================================================
# CLASSIFY FACES
# ============================================================================

def detect_and_classify_faces(frame, face_net, model):
    """
    Detect faces using DNN and classify them.
    """

    # Detect faces using DNN
    faces = detect_faces_dnn(frame, face_net)

    detections = []

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y + h, x:x + w]

        if face_img.size == 0:
            continue

        # Resize to model input size
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Preprocess for MobileNetV2
        face_preprocessed = preprocess_input(face_rgb * 255)

        # Add batch dimension
        face_batch = np.expand_dims(face_preprocessed, axis=0)

        # Predict
        predictions = model.predict(face_batch, verbose=0)

        # Get probabilities
        prob_with_mask = predictions[0][0]
        prob_without_mask = predictions[0][1]

        # Get predicted class
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        label = CATEGORIES[class_idx]

        detections.append({
            'bbox': (x, y, w, h),
            'label': label,
            'confidence': confidence,
            'prob_with_mask': prob_with_mask,
            'prob_without_mask': prob_without_mask
        })

    return detections


# ============================================================================
# DRAW DETECTIONS
# ============================================================================

def draw_detections(frame, detections):
    """Draw bounding boxes and labels."""

    stats = {
        'total': len(detections),
        'with_mask': 0,
        'without_mask': 0
    }

    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        prob_with_mask = detection['prob_with_mask']
        prob_without_mask = detection['prob_without_mask']

        stats[label] += 1

        color = COLORS[label]

        if confidence < CONFIDENCE_THRESHOLD:
            color = tuple(int(c * 0.5) for c in color)

        thickness = 3 if confidence >= CONFIDENCE_THRESHOLD else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if DEBUG_MODE:
            text = f"{label}: {confidence * 100:.1f}%"
            text2 = f"W:{prob_with_mask * 100:.0f}% N:{prob_without_mask * 100:.0f}%"
        else:
            text = f"{label}: {confidence * 100:.1f}%"
            text2 = None

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1
        )

        cv2.putText(
            frame, text, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

        if text2 and DEBUG_MODE:
            cv2.putText(
                frame, text2, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    return frame, stats


# ============================================================================
# DRAW INFO PANEL
# ============================================================================

def draw_info_panel(frame, fps, stats):
    """Draw information panel."""

    overlay = frame.copy()
    panel_height = 160
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)

    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    stats_text = f"Faces: {stats['total']} | With Mask: {stats['with_mask']} | Without: {stats['without_mask']}"
    cv2.putText(frame, stats_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Face Detector: DNN (Deep Learning)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f"Mask Classifier: Trained on 10 images", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    controls_text = "'q'=quit | 's'=screenshot | 'd'=toggle debug"
    cv2.putText(frame, controls_text, (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================================
# SAVE SCREENSHOT
# ============================================================================

def save_screenshot(frame):
    """Save current frame as screenshot."""

    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_DIR}/screenshot_dnn_{timestamp}.jpg"

    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

    return filename


# ============================================================================
# WEBCAM DETECTION
# ============================================================================

def webcam_detection(face_net, model):
    """Perform real-time detection using DNN face detector."""

    global DEBUG_MODE

    print("\nInitializing webcam...")
    print("=" * 60)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[FAIL] Could not open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera opened: {actual_width} x {actual_height}")
    print("\nUsing DNN Face Detector (better face detection)")
    print("BUT: Mask classifier still limited by small dataset")
    print("=" * 60)

    fps = 0
    frame_count = 0
    start_time = time.time()

    print("\nStarting detection...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        detections = detect_and_classify_faces(frame, face_net, model)

        frame, stats = draw_detections(frame, detections)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time

        frame = draw_info_panel(frame, fps, stats)

        cv2.imshow('Face Mask Detection - DNN Version', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nStopping...")
            break
        elif key == ord('s'):
            save_screenshot(frame)
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")

    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print("\n" + "=" * 60)
    print("Session complete!")
    print(f"Average FPS: {avg_fps:.2f}")
    print("=" * 60)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FACE MASK DETECTION - DNN FACE DETECTOR")
    print("=" * 60)

    print("\nIMPORTANT:")
    print("DNN face detector will find faces BETTER than Haar Cascade.")
    print("However, mask classification accuracy still depends on training data.")
    print("With only 10 training images, predictions will remain limited.")
    print("=" * 60)

    try:
        face_net, model = load_models()
        if face_net is None or model is None:
            exit(1)
    except Exception as e:
        print(f"[FAIL] Error loading models: {e}")
        exit(1)

    try:
        webcam_detection(face_net, model)
        print("\n[PASS] Detection complete!")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback

        traceback.print_exc()