"""
Data Collection Script - Capture Training Images
This script helps you collect training images using your webcam.
Capture images with and without a mask to expand your dataset.
"""

import cv2
import os
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory structure
DATASET_DIR = 'dataset'
WITH_MASK_DIR = os.path.join(DATASET_DIR, 'with_mask')
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, 'without_mask')

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Capture settings
COUNTDOWN_SECONDS = 3  # Countdown before capturing
MIN_INTERVAL_SECONDS = 1  # Minimum time between captures

# Image quality
IMAGE_QUALITY = 95  # JPEG quality (0-100)


# ============================================================================
# SETUP DIRECTORIES
# ============================================================================

def setup_directories():
    """
    Create dataset directories if they do not exist.
    Preserves existing images.
    """

    print("Setting up directories...")

    # Create main dataset directory
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"[CREATED] {DATASET_DIR}")

    # Create with_mask directory
    if not os.path.exists(WITH_MASK_DIR):
        os.makedirs(WITH_MASK_DIR)
        print(f"[CREATED] {WITH_MASK_DIR}")

    # Create without_mask directory
    if not os.path.exists(WITHOUT_MASK_DIR):
        os.makedirs(WITHOUT_MASK_DIR)
        print(f"[CREATED] {WITHOUT_MASK_DIR}")

    # Count existing images
    with_mask_count = len([f for f in os.listdir(WITH_MASK_DIR)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    without_mask_count = len([f for f in os.listdir(WITHOUT_MASK_DIR)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\nCurrent dataset:")
    print(f"  with_mask: {with_mask_count} images")
    print(f"  without_mask: {without_mask_count} images")
    print(f"  Total: {with_mask_count + without_mask_count} images")

    return with_mask_count, without_mask_count


# ============================================================================
# GET NEXT FILENAME
# ============================================================================

def get_next_filename(directory, prefix="img"):
    """
    Get the next available filename in a directory.

    Finds the highest numbered file and returns the next number.
    Format: prefix_001.jpg, prefix_002.jpg, etc.

    Args:
        directory: Directory to check
        prefix: Prefix for filename

    Returns:
        Full path for next image
    """

    # Get all existing image files
    existing_files = [f for f in os.listdir(directory)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Find highest number
    max_num = 0
    for filename in existing_files:
        try:
            # Extract number from filename
            # Handles formats like: img_001.jpg, 5.jpg, photo_123.png
            parts = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            parts = parts.split('_')
            num = int(parts[-1])
            max_num = max(max_num, num)
        except:
            continue

    # Return next number
    next_num = max_num + 1
    filename = f"{prefix}_{next_num:03d}.jpg"

    return os.path.join(directory, filename)


# ============================================================================
# SAVE IMAGE
# ============================================================================

def save_image(frame, directory, prefix="img"):
    """
    Save frame to directory with next available filename.

    Args:
        frame: Image to save
        directory: Where to save
        prefix: Filename prefix

    Returns:
        filepath: Path where image was saved
    """

    filepath = get_next_filename(directory, prefix)

    # Save with high quality
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])

    return filepath


# ============================================================================
# DRAW UI OVERLAY
# ============================================================================

def draw_ui(frame, mode, countdown, total_captured):
    """
    Draw user interface overlay on frame.

    Shows:
    - Current mode (with_mask or without_mask)
    - Countdown timer
    - Number of images captured
    - Instructions

    Args:
        frame: Frame to draw on
        mode: Current capture mode
        countdown: Countdown timer value (or None)
        total_captured: Number of images captured in this session

    Returns:
        frame: Frame with UI overlay
    """

    h, w = frame.shape[:2]

    # Create semi-transparent overlay for top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw mode indicator
    mode_text = f"MODE: {mode.upper()}"
    mode_color = (0, 255, 0) if mode == 'with_mask' else (0, 0, 255)
    cv2.putText(frame, mode_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)

    # Draw capture count
    count_text = f"Captured this session: {total_captured}"
    cv2.putText(frame, count_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw instructions
    instructions = "SPACE=Capture | M=Switch Mode | Q=Quit"
    cv2.putText(frame, instructions, (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw countdown if active
    if countdown is not None and countdown > 0:
        # Draw large countdown in center
        countdown_text = str(int(countdown))

        # Get text size to center it
        (text_width, text_height), _ = cv2.getTextSize(
            countdown_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            10
        )

        x = (w - text_width) // 2
        y = (h + text_height) // 2

        # Draw countdown with outline for visibility
        cv2.putText(frame, countdown_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
        cv2.putText(frame, countdown_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

    # Draw capture guide frame
    # This helps user position their face
    margin = 100
    cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (0, 255, 0), 2)
    guide_text = "Position face within frame"
    cv2.putText(frame, guide_text, (margin, margin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


# ============================================================================
# DATA COLLECTION SESSION
# ============================================================================

def collect_data():
    """
    Main data collection loop.

    Process:
    1. Open webcam
    2. Show live preview
    3. User positions face and presses SPACE
    4. Countdown starts
    5. Image captured and saved
    6. Repeat or switch mode
    """

    print("\n" + "=" * 60)
    print("Starting data collection session...")
    print("=" * 60)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[FAIL] Could not open camera {CAMERA_INDEX}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera opened: {actual_width} x {actual_height}")

    # Initial mode
    mode = 'with_mask'

    # Capture tracking
    total_captured = 0
    last_capture_time = 0
    countdown_start = None

    print("\nInstructions:")
    print("  1. Position your face within the green frame")
    print("  2. Press SPACE to start countdown and capture")
    print("  3. Press M to switch between with_mask and without_mask modes")
    print("  4. Press Q to quit")
    print("\nTips for good training data:")
    print("  - Capture from different angles (front, slight left/right)")
    print("  - Different lighting (bright, dim, natural light)")
    print("  - Different distances from camera")
    print("  - Different facial expressions")
    print("  - Move around the room between captures")
    print("=" * 60)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[FAIL] Failed to read frame")
            break

        # Mirror the frame for more intuitive use
        frame = cv2.flip(frame, 1)

        # Handle countdown
        countdown = None
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            countdown = COUNTDOWN_SECONDS - elapsed

            # Capture when countdown reaches 0
            if countdown <= 0:
                # Determine save directory based on mode
                save_dir = WITH_MASK_DIR if mode == 'with_mask' else WITHOUT_MASK_DIR

                # Save image
                filepath = save_image(frame, save_dir)
                total_captured += 1

                # Get just the filename for display
                filename = os.path.basename(filepath)
                print(f"[CAPTURED] {mode}: {filename} (Total: {total_captured})")

                # Reset countdown
                countdown_start = None
                last_capture_time = time.time()

        # Draw UI overlay
        frame = draw_ui(frame, mode, countdown, total_captured)

        # Display frame
        cv2.imshow('Data Collection - Press SPACE to capture', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        # Quit on 'q'
        if key == ord('q'):
            print("\nStopping data collection...")
            break

        # Start countdown on SPACE
        elif key == ord(' '):
            # Check minimum interval between captures
            current_time = time.time()
            if current_time - last_capture_time >= MIN_INTERVAL_SECONDS:
                countdown_start = time.time()
                print(f"Countdown started for {mode}...")
            else:
                print("Please wait before next capture")

        # Switch mode on 'm'
        elif key == ord('m'):
            mode = 'without_mask' if mode == 'with_mask' else 'with_mask'
            print(f"\nMode switched to: {mode}")
            print(f"Put {'ON' if mode == 'with_mask' else 'OFF'} your mask now")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Images captured this session: {total_captured}")

    # Count total images now
    with_mask_count = len([f for f in os.listdir(WITH_MASK_DIR)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    without_mask_count = len([f for f in os.listdir(WITHOUT_MASK_DIR)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\nTotal dataset now:")
    print(f"  with_mask: {with_mask_count} images")
    print(f"  without_mask: {without_mask_count} images")
    print(f"  Total: {with_mask_count + without_mask_count} images")
    print("=" * 60)

    # Recommendations
    print("\nRecommendations:")
    total = with_mask_count + without_mask_count

    if total < 50:
        print(f"  Current: {total} images - Need at least 50 more")
        print("  Keep collecting!")
    elif total < 100:
        print(f"  Current: {total} images - Getting better!")
        print("  Collect 50-100 more for good accuracy")
    elif total < 200:
        print(f"  Current: {total} images - Good start!")
        print("  This should give basic accuracy")
    else:
        print(f"  Current: {total} images - Excellent!")
        print("  This should give good accuracy")

    print("\nNext steps:")
    print("  1. Run: python 1_data_preparation.py")
    print("  2. Run: python 2_train_model.py")
    print("  3. Test with: python 5_detect_webcam.py")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DATA COLLECTION TOOL")
    print("Capture training images for face mask detection")
    print("=" * 60)

    # Setup directories
    with_mask_count, without_mask_count = setup_directories()

    # Instructions
    print("\n" + "=" * 60)
    print("IMPORTANT TIPS FOR QUALITY TRAINING DATA:")
    print("=" * 60)
    print("\n1. VARIETY IS KEY:")
    print("   - Different angles (front, left, right)")
    print("   - Different distances from camera")
    print("   - Different lighting conditions")
    print("   - Different locations in your room")
    print("   - Different facial expressions")

    print("\n2. FOR WITH_MASK MODE:")
    print("   - Use different masks if possible")
    print("   - Position mask properly on face")
    print("   - Try different mask colors")

    print("\n3. FOR WITHOUT_MASK MODE:")
    print("   - Make sure no mask is visible")
    print("   - Cover same variations as with_mask")

    print("\n4. HOW MANY IMAGES:")
    print("   - Minimum: 50 per class (100 total)")
    print("   - Recommended: 100 per class (200 total)")
    print("   - Better: 200+ per class (400+ total)")

    print("\n5. BALANCE IS IMPORTANT:")
    print("   - Try to have similar numbers in both classes")
    print(f"   - Currently: {with_mask_count} with_mask, {without_mask_count} without_mask")

    print("\n" + "=" * 60)

    # Ask user if ready
    input("\nPress Enter to start data collection (or Ctrl+C to cancel)...")

    # Start collection
    try:
        collect_data()
    except KeyboardInterrupt:
        print("\n\nData collection cancelled by user")
    except Exception as e:
        print(f"\n[FAIL] Error during collection: {e}")
        import traceback

        traceback.print_exc()