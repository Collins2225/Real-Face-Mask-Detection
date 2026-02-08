"""
Quick Verification Script
Run this BEFORE the main scripts to check everything is set up correctly.
"""

import os
import sys

print("=" * 70)
print("FACE MASK DETECTION - PRE-FLIGHT VERIFICATION")
print("=" * 70)

# ============================================================================
# TEST 1: CHECK PYTHON VERSION
# ============================================================================
print("\n[TEST 1] Checking Python version...")
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major >= 3 and python_version.minor >= 8:
    print("[PASS] Python version is compatible (3.8 or higher)")
else:
    print("[FAIL] Python version too old. Please use Python 3.8 or higher")

# ============================================================================
# TEST 2: CHECK REQUIRED PACKAGES
# ============================================================================
print("\n[TEST 2] Checking required packages...")

required_packages = {
    'tensorflow': 'TensorFlow',
    'cv2': 'OpenCV (cv2)',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-learn',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn'
}

missing_packages = []
installed_packages = []

for package_name, display_name in required_packages.items():
    try:
        if package_name == 'cv2':
            import cv2

            version = cv2.__version__
        elif package_name == 'sklearn':
            import sklearn

            version = sklearn.__version__
        else:
            exec(f"import {package_name}")
            exec(f"version = {package_name}.__version__")

        print(f"  [PASS] {display_name}: {version}")
        installed_packages.append(display_name)
    except ImportError:
        print(f"  [FAIL] {display_name}: NOT INSTALLED")
        missing_packages.append(display_name)

if missing_packages:
    print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
else:
    print("\n[PASS] All required packages are installed!")

# ============================================================================
# TEST 3: CHECK DATASET FOLDER STRUCTURE
# ============================================================================
print("\n[TEST 3] Checking dataset folder structure...")

# Check if dataset folder exists
if os.path.exists('dataset'):
    print("  [PASS] 'dataset' folder found")

    # List all items in dataset folder
    dataset_contents = os.listdir('dataset')
    print(f"  Found folders/files: {dataset_contents}")

    # Check for mask folders (check both naming conventions)
    possible_with_mask = ['with_mask', 'WithMask', 'with mask', 'With_Mask']
    possible_without_mask = ['without_mask', 'WithoutMask', 'without mask', 'Without_Mask']

    with_mask_folder = None
    without_mask_folder = None

    # Find the actual folder names
    for item in dataset_contents:
        if item in possible_with_mask:
            with_mask_folder = item
        if item in possible_without_mask:
            without_mask_folder = item

    # Check with_mask folder
    if with_mask_folder:
        with_mask_path = os.path.join('dataset', with_mask_folder)
        with_mask_images = [f for f in os.listdir(with_mask_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  [PASS] '{with_mask_folder}' folder found")
        print(f"     - Contains {len(with_mask_images)} images: {with_mask_images}")

        if len(with_mask_images) < 5:
            print(f"     [WARNING] Only {len(with_mask_images)} images found. Need at least 5.")
    else:
        print(f"  [FAIL] 'with_mask' folder NOT FOUND")
        print(f"     Looking for one of: {possible_with_mask}")

    # Check without_mask folder
    if without_mask_folder:
        without_mask_path = os.path.join('dataset', without_mask_folder)
        without_mask_images = [f for f in os.listdir(without_mask_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  [PASS] '{without_mask_folder}' folder found")
        print(f"     - Contains {len(without_mask_images)} images: {without_mask_images}")

        if len(without_mask_images) < 5:
            print(f"     [WARNING] Only {len(without_mask_images)} images found. Need at least 5.")
    else:
        print(f"  [FAIL] 'without_mask' folder NOT FOUND")
        print(f"     Looking for one of: {possible_without_mask}")

    # Check if folder names match script expectations
    if with_mask_folder and without_mask_folder:
        if with_mask_folder != 'with_mask' or without_mask_folder != 'without_mask':
            print(f"\n  [WARNING] FOLDER NAME MISMATCH:")
            print(f"     Your folders: '{with_mask_folder}' and '{without_mask_folder}'")
            print(f"     Script expects: 'with_mask' and 'without_mask'")
            print(f"\n  FIX OPTIONS:")
            print(f"     Option 1 (Recommended): Rename folders to 'with_mask' and 'without_mask'")
            print(f"     Option 2: Update CATEGORIES in 1_data_preparation.py to:")
            print(f"               CATEGORIES = ['{with_mask_folder}', '{without_mask_folder}']")
        else:
            print(f"\n  [PASS] Folder names match script expectations!")
else:
    print("  [FAIL] 'dataset' folder NOT FOUND")
    print("     Please create: dataset/with_mask/ and dataset/without_mask/")

# ============================================================================
# TEST 4: TEST IMAGE LOADING
# ============================================================================
print("\n[TEST 4] Testing image loading with OpenCV...")

if 'cv2' in sys.modules and with_mask_folder:
    import cv2
    import numpy as np

    # Try to load one image from with_mask folder
    with_mask_path = os.path.join('dataset', with_mask_folder)
    test_images = [f for f in os.listdir(with_mask_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if test_images:
        test_image_path = os.path.join(with_mask_path, test_images[0])
        print(f"  Testing with: {test_image_path}")

        # Try to load the image
        img = cv2.imread(test_image_path)

        if img is not None:
            print(f"  [PASS] Image loaded successfully!")
            print(f"     - Original shape: {img.shape}")
            print(f"     - Data type: {img.dtype}")

            # Try preprocessing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = img_resized / 255.0

            print(f"     - After preprocessing: {img_normalized.shape}")
            print(f"     - Pixel range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
            print(f"  [PASS] Preprocessing works correctly!")
        else:
            print(f"  [FAIL] Failed to load image. Check if file is corrupted.")
    else:
        print(f"  [WARNING] No images found to test")
else:
    print("  [WARNING] Skipping (OpenCV not installed or no dataset)")

# ============================================================================
# TEST 5: CHECK DISK SPACE
# ============================================================================
print("\n[TEST 5] Checking available disk space...")

try:
    import shutil

    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    print(f"  Free disk space: {free_gb:.2f} GB")

    if free_gb > 1.0:
        print(f"  [PASS] Sufficient disk space available")
    else:
        print(f"  [WARNING] Low disk space (less than 1 GB free)")
except:
    print("  [WARNING] Could not check disk space")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_good = True

# Summary checks
if python_version.major >= 3 and python_version.minor >= 8:
    print("[PASS] Python version OK")
else:
    print("[FAIL] Python version needs upgrade")
    all_good = False

if not missing_packages:
    print("[PASS] All packages installed")
else:
    print(f"[FAIL] Missing packages: {', '.join(missing_packages)}")
    all_good = False

if os.path.exists('dataset') and with_mask_folder and without_mask_folder:
    print("[PASS] Dataset structure OK")
    if with_mask_folder != 'with_mask' or without_mask_folder != 'without_mask':
        print("[WARNING] Folder names need adjustment (see details above)")
        all_good = False
else:
    print("[FAIL] Dataset structure incorrect")
    all_good = False

print("=" * 70)

if all_good:
    print("\nALL CHECKS PASSED! You're ready to run:")
    print("   1. python 1_data_preparation.py")
    print("   2. python 2_train_model.py")
else:
    print("\nPLEASE FIX THE ISSUES ABOVE BEFORE PROCEEDING")
    print("\nCommon fixes:")
    print("  - Install packages: pip install -r requirements.txt")
    print("  - Rename folders: 'WithMask' -> 'with_mask', 'WithoutMask' -> 'without_mask'")
    print("  - Or update CATEGORIES variable in 1_data_preparation.py")

print("=" * 70)