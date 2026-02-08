"""
Face Mask Detection - Static Image Detection
This script loads the trained model and detects face masks in static images.
It first detects faces, then classifies each face as with_mask or without_mask.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and cascade paths
MODEL_PATH = 'face_mask_detector.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Detection parameters
IMG_SIZE = 224  # Input size for our CNN
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display prediction

# Class labels
CATEGORIES = ['with_mask', 'without_mask']

# Colors for bounding boxes (BGR format)
# Green for with_mask, Red for without_mask
COLORS = {
    'with_mask': (0, 255, 0),
    'without_mask': (0, 0, 255)
}


# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL AND FACE DETECTOR
# ============================================================================

def load_detector_and_model():
    """
    Load the face detection cascade and trained mask detection model.

    Face Detection:
    - Uses Haar Cascade classifier (classical computer vision)
    - Pre-trained model that comes with OpenCV
    - Fast and works well for frontal faces

    Mask Detection:
    - Our custom trained MobileNetV2 model
    - Classifies detected faces as with_mask or without_mask

    Returns:
        face_cascade: OpenCV cascade classifier for face detection
        model: Loaded Keras model for mask classification
    """

    print("Loading face detector and mask detection model...")

    # Load Haar Cascade for face detection
    # This is a pre-trained classifier that detects faces
    # It works by looking for patterns of light and dark regions
    # that are characteristic of human faces
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # Verify cascade loaded successfully
    if face_cascade.empty():
        raise IOError(f"Failed to load face cascade from {FACE_CASCADE_PATH}")

    # Load our trained mask detection model
    # This is the model we trained in the previous script
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

    model = load_model(MODEL_PATH)

    print(f"[PASS] Face detector loaded")
    print(f"[PASS] Mask detection model loaded from {MODEL_PATH}")

    return face_cascade, model


# ============================================================================
# STEP 2: DETECT FACES IN IMAGE
# ============================================================================

def detect_faces(image, face_cascade):
    """
    Detect faces in an image using Haar Cascade classifier.

    Process:
    1. Convert image to grayscale (Haar Cascade works on grayscale)
    2. Apply the cascade classifier to find faces
    3. Returns bounding box coordinates for each face

    Args:
        image: BGR image (numpy array)
        face_cascade: Loaded Haar Cascade classifier

    Returns:
        faces: List of (x, y, w, h) tuples for each detected face
               x, y: top-left corner coordinates
               w, h: width and height of the bounding box
    """

    # Convert to grayscale
    # Haar Cascade operates on grayscale images for efficiency
    # Color information is not needed for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # detectMultiScale parameters:
    # - scaleFactor=1.1: How much to reduce image size at each scale
    # - minNeighbors=5: How many neighbors each candidate needs to be retained
    # - minSize=(30, 30): Minimum face size to detect
    # Higher minNeighbors = fewer false positives but may miss some faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


# ============================================================================
# STEP 3: CLASSIFY EACH DETECTED FACE
# ============================================================================

def classify_face(face_img, model):
    """
    Classify a face region as with_mask or without_mask.

    Process:
    1. Preprocess the face image (resize, normalize)
    2. Pass through our trained model
    3. Get prediction probabilities
    4. Return predicted class and confidence

    Args:
        face_img: Cropped face region (BGR image)
        model: Trained Keras model

    Returns:
        label: Predicted class ('with_mask' or 'without_mask')
        confidence: Confidence score (0-100%)
    """

    # Resize face to model input size (224x224)
    # Our model expects this specific size
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to RGB
    # OpenCV loads images in BGR, but our model was trained on RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Preprocess for MobileNetV2
    # This applies the same preprocessing used during training
    # Converts pixel values from [0, 255] to [-1, 1] range
    face_preprocessed = preprocess_input(face_rgb * 255)

    # Add batch dimension
    # Model expects input shape: (batch_size, height, width, channels)
    # We have: (224, 224, 3)
    # We need: (1, 224, 224, 3)
    # expand_dims adds a dimension at position 0
    face_batch = np.expand_dims(face_preprocessed, axis=0)

    # Make prediction
    # Returns probability distribution over classes
    # Example: [[0.95, 0.05]] means 95% with_mask, 5% without_mask
    predictions = model.predict(face_batch, verbose=0)

    # Get the predicted class
    # argmax returns the index of the maximum value
    # 0 = with_mask, 1 = without_mask
    class_idx = np.argmax(predictions[0])

    # Get the confidence score
    # This is the probability of the predicted class
    confidence = predictions[0][class_idx] * 100

    # Get class label
    label = CATEGORIES[class_idx]

    return label, confidence


# ============================================================================
# STEP 4: PROCESS IMAGE AND DRAW RESULTS
# ============================================================================

def detect_and_draw(image_path, face_cascade, model, output_path=None):
    """
    Detect faces and masks in an image, then draw bounding boxes and labels.

    Complete pipeline:
    1. Load image
    2. Detect all faces
    3. For each face:
       - Extract face region
       - Classify as with_mask or without_mask
       - Draw bounding box and label
    4. Display and optionally save result

    Args:
        image_path: Path to input image
        face_cascade: Face detector
        model: Mask classification model
        output_path: Optional path to save result image
    """

    print(f"\nProcessing image: {image_path}")
    print("=" * 60)

    # Load the image
    # cv2.imread reads image from disk as numpy array in BGR format
    image = cv2.imread(image_path)

    if image is None:
        print(f"[FAIL] Could not load image from {image_path}")
        return

    # Get image dimensions
    h, w = image.shape[:2]
    print(f"Image size: {w} x {h}")

    # Detect faces
    faces = detect_faces(image, face_cascade)
    print(f"Detected {len(faces)} face(s)")

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        print(f"\nFace {i + 1}:")
        print(f"  Location: ({x}, {y}), Size: {w} x {h}")

        # Extract face region
        # image[y:y+h, x:x+w] extracts the rectangular region
        # This is numpy array slicing: [rows, columns]
        face_img = image[y:y + h, x:x + w]

        # Classify the face
        label, confidence = classify_face(face_img, model)
        print(f"  Prediction: {label}")
        print(f"  Confidence: {confidence:.2f}%")

        # Only draw if confidence is above threshold
        if confidence >= CONFIDENCE_THRESHOLD * 100:
            # Get color based on prediction
            color = COLORS[label]

            # Draw bounding box around face
            # cv2.rectangle draws a rectangle on the image
            # (x, y): top-left corner
            # (x+w, y+h): bottom-right corner
            # color: BGR tuple
            # 2: line thickness
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Prepare label text
            text = f"{label}: {confidence:.2f}%"

            # Calculate text size for background rectangle
            # getTextSize returns ((width, height), baseline)
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Draw filled rectangle for text background
            # This makes text more readable
            cv2.rectangle(
                image,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1  # -1 means filled rectangle
            )

            # Draw text on top of the rectangle
            # putText parameters:
            # - image: where to draw
            # - text: what to write
            # - (x, y - 5): position (bottom-left corner of text)
            # - font: font type
            # - 0.6: font scale
            # - (255, 255, 255): white color for text
            # - 2: thickness
            cv2.putText(
                image,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # Save result if output path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"\n[PASS] Result saved to {output_path}")

    # Display the result
    # cv2.imshow creates a window and displays the image
    cv2.imshow('Face Mask Detection', image)
    print("\nPress any key to close the window...")

    # waitKey(0) waits indefinitely for a key press
    # This keeps the window open until user presses a key
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution flow:
    1. Load face detector and mask classification model
    2. Get image path from user
    3. Process image and display results
    """

    print("=" * 60)
    print("FACE MASK DETECTION - STATIC IMAGE")
    print("=" * 60)

    # Load models
    try:
        face_cascade, model = load_detector_and_model()
    except Exception as e:
        print(f"[FAIL] Error loading models: {e}")
        exit(1)

    # Get image path from user
    print("\n" + "=" * 60)
    print("Enter the path to an image file")
    print("Or press Enter to use a test image from dataset")
    print("=" * 60)

    image_path = input("\nImage path: ").strip()

    # If no path provided, use a sample from dataset
    if not image_path:
        # Try to find a sample image from the dataset
        test_paths = [
            'dataset/with_mask/1.jpg',
            'dataset/without_mask/1.jpg'
        ]

        for path in test_paths:
            if os.path.exists(path):
                image_path = path
                print(f"Using test image: {image_path}")
                break

        if not image_path:
            print("[FAIL] No test images found. Please provide an image path.")
            exit(1)

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"[FAIL] Image not found: {image_path}")
        exit(1)

    # Process the image
    # Generate output filename
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')

    try:
        detect_and_draw(image_path, face_cascade, model, output_path)
        print("\n[PASS] Detection complete!")
    except Exception as e:
        print(f"\n[FAIL] Error during detection: {e}")
        import traceback

        traceback.print_exc()