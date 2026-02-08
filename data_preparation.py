"""
Face Mask Detection - Data Preparation Script
This script prepares the dataset for training our CNN model.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define paths - adjust these based on where you store your dataset
DATA_DIR = 'dataset'  # Main dataset directory
CATEGORIES = ['with_mask', 'without_mask']  # Our two classes

# Image preprocessing parameters
IMG_SIZE = 224  # Input size for our CNN (224x224 is common for transfer learning)
BATCH_SIZE = 8  # Reduced batch size for small dataset
TEST_SIZE = 0.2  # 20% of data for testing (2 images with 10 total)
VAL_SIZE = 0.25  # 25% of remaining data for validation (2 images from remaining 8)


# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================

def load_dataset(data_dir, categories, img_size):
    """
    Load images from directories and prepare them for training.

    Args:
        data_dir: Path to the main dataset folder
        categories: List of class names (subdirectories)
        img_size: Target size to resize all images to

    Returns:
        data: List of preprocessed images
        labels: List of corresponding labels (0 or 1)
    """
    data = []
    labels = []

    # Loop through each category (with_mask, without_mask)
    for category_idx, category in enumerate(categories):
        # Build the full path: dataset/with_mask or dataset/without_mask
        category_path = os.path.join(data_dir, category)

        print(f"\nLoading images from: {category_path}")

        # Check if directory exists
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} not found!")
            continue

        # Get all image files in this category
        image_files = os.listdir(category_path)
        print(f"Found {len(image_files)} images in {category}")

        # Process each image
        for img_file in image_files:
            try:
                # Build full image path
                img_path = os.path.join(category_path, img_file)

                # Read the image using OpenCV
                # cv2.imread returns a numpy array in BGR format
                image = cv2.imread(img_path)

                # Skip if image couldn't be loaded
                if image is None:
                    continue

                # Convert from BGR (OpenCV default) to RGB (standard format)
                # This is important because our model will expect RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize image to our target size (224x224)
                # This ensures all images have the same dimensions for the CNN
                image = cv2.resize(image, (img_size, img_size))

                # Normalize pixel values to range [0, 1]
                # Original values are [0, 255], dividing by 255 gives [0, 1]
                # This helps the neural network train better
                image = image / 255.0

                # Add the processed image and its label to our lists
                data.append(image)
                labels.append(category_idx)  # 0 for with_mask, 1 for without_mask

            except Exception as e:
                # If any error occurs, skip this image and continue
                print(f"Error processing {img_file}: {e}")
                continue

    # Convert lists to numpy arrays
    # Neural networks work with numpy arrays
    data = np.array(data, dtype='float32')
    labels = np.array(labels)

    print(f"\n{'=' * 60}")
    print(f"Dataset loaded successfully!")
    print(f"Total images: {len(data)}")
    print(f"Image shape: {data[0].shape}")
    print(f"Data type: {data.dtype}")
    print(f"{'=' * 60}\n")

    return data, labels


# ============================================================================
# STEP 2: SPLIT DATASET INTO TRAIN, VALIDATION, AND TEST SETS
# ============================================================================

def split_dataset(data, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into training, validation, and test sets.

    Why we need three sets:
    - Training set: Used to train the model (learn patterns)
    - Validation set: Used during training to tune hyperparameters
    - Test set: Used after training to evaluate final performance

    Args:
        data: Array of images
        labels: Array of labels
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    # First split: separate out test set
    # train_test_split shuffles data and splits it randomly
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Ensures balanced class distribution in splits
    )

    # Second split: divide remaining data into train and validation
    # We adjust val_size because we're working with the remaining data
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# STEP 3: DATA AUGMENTATION
# ============================================================================

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators with augmentation for training.

    Data augmentation artificially increases dataset size by creating
    modified versions of images (rotated, flipped, zoomed, etc.)
    This helps the model generalize better and prevents overfitting.

    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Number of images per batch

    Returns:
        train_generator, val_generator
    """

    # Training data augmentation
    # For small datasets, we apply MORE aggressive transformations
    # This creates more variety and helps prevent overfitting
    train_datagen = ImageDataGenerator(
        rotation_range=30,  # Randomly rotate images up to 30 degrees
        width_shift_range=0.3,  # Randomly shift images horizontally (30%)
        height_shift_range=0.3,  # Randomly shift images vertically (30%)
        horizontal_flip=True,  # Randomly flip images horizontally
        zoom_range=0.3,  # Randomly zoom in/out
        shear_range=0.2,  # Randomly apply shear transformation
        brightness_range=[0.7, 1.3],  # Randomly adjust brightness
        fill_mode='nearest'  # Strategy for filling new pixels
    )

    # Validation data - NO augmentation
    # We want to evaluate on original images to get true performance
    val_datagen = ImageDataGenerator()

    # Fit the generator on training data
    # This computes any statistics needed for transformations
    train_datagen.fit(X_train)

    # Create generators that will yield batches of augmented images
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True  # Shuffle data each epoch
    )

    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle validation data
    )

    return train_generator, val_generator


# ============================================================================
# STEP 4: VISUALIZE SAMPLE IMAGES
# ============================================================================

def visualize_samples(data, labels, categories, num_samples=8):
    """
    Display sample images from the dataset.
    This helps us verify the data is loaded correctly.

    Args:
        data: Array of images
        labels: Array of labels
        categories: List of category names
        num_samples: Number of images to display
    """
    plt.figure(figsize=(12, 6))

    # Randomly select indices to display
    indices = np.random.choice(len(data), num_samples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(2, 4, i + 1)

        # Display the image
        plt.imshow(data[idx])

        # Set title with the class name
        plt.title(f"{categories[labels[idx]]}")

        # Remove axis ticks for cleaner look
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("Sample images saved as 'sample_images.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution flow:
    1. Load the dataset
    2. Split into train/val/test
    3. Create data generators
    4. Visualize samples
    5. Save processed data
    """

    print("=" * 60)
    print("FACE MASK DETECTION - DATA PREPARATION")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    data, labels = load_dataset(DATA_DIR, CATEGORIES, IMG_SIZE)

    # Step 2: Split dataset
    print("\n[2/5] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        data, labels, TEST_SIZE, VAL_SIZE
    )

    # Step 3: Create data generators
    print("\n[3/5] Creating data generators...")
    train_gen, val_gen = create_data_generators(
        X_train, y_train, X_val, y_val, BATCH_SIZE
    )

    # Step 4: Visualize samples
    print("\n[4/5] Visualizing sample images...")
    visualize_samples(data, labels, CATEGORIES)

    # Step 5: Save processed data for later use
    print("\n[5/5] Saving processed data...")
    np.savez_compressed(
        'processed_data.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print("Data saved to 'processed_data.npz'")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run this script to prepare your data")
    print("2. Move on to building and training the CNN model")
    print("=" * 60)