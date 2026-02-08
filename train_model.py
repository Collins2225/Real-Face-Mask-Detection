"""
Face Mask Detection - Model Training with Transfer Learning
This script builds and trains a CNN using MobileNetV2 (pre-trained on ImageNet)
Transfer learning allows us to work effectively with small datasets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model parameters
IMG_SIZE = 224  # MobileNetV2 expects 224x224 images
NUM_CLASSES = 2  # with_mask and without_mask
LEARNING_RATE = 0.0001  # Small learning rate for fine-tuning
EPOCHS = 50  # Maximum number of training iterations
BATCH_SIZE = 8  # Small batch size for small dataset (was 32, reduced for stability)

# Paths
MODEL_SAVE_PATH = 'face_mask_detector.h5'
DATA_FILE = 'processed_data.npz'


# ============================================================================
# STEP 1: BUILD THE MODEL USING TRANSFER LEARNING
# ============================================================================

def build_model(img_size=224, num_classes=2):
    """
    Build a face mask detection model using Transfer Learning.

    Transfer Learning Strategy:
    1. Load MobileNetV2 pre-trained on ImageNet (1.4M images, 1000 classes)
    2. Freeze the base model (don't retrain these layers initially)
    3. Add custom layers on top for our specific task (mask detection)
    4. Train only the new layers we added

    Why MobileNetV2?
    - Lightweight and fast (good for real-time detection)
    - Pre-trained on ImageNet (already knows visual features)
    - Works well with small datasets through transfer learning
    - Optimized for mobile/edge devices

    Args:
        img_size: Input image size (224x224)
        num_classes: Number of output classes (2: with/without mask)

    Returns:
        model: Compiled Keras model ready for training
    """

    print("Building model with Transfer Learning...")
    print("=" * 60)

    # Load the pre-trained MobileNetV2 model
    # include_top=False: Remove the original classification layers
    # weights='imagenet': Use weights from ImageNet training
    # input_shape: Specify our input dimensions (224, 224, 3 for RGB)
    base_model = MobileNetV2(
        include_top=False,  # Don't include the ImageNet classifier
        weights='imagenet',  # Use pre-trained weights
        input_shape=(img_size, img_size, 3)
    )

    # Freeze the base model
    # This means we won't update these weights during initial training
    # The model already knows how to detect edges, textures, patterns, etc.
    base_model.trainable = False

    print(f"Base Model: MobileNetV2")
    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Base model trainable: {base_model.trainable}")

    # Build our custom classification head
    # This is what we'll actually train for mask detection

    # Start with the base model
    x = base_model.output

    # GlobalAveragePooling2D: Converts feature maps to a single vector
    # Instead of flattening (which creates many parameters), we average
    # Input: (7, 7, 1280) -> Output: (1280,)
    # This reduces parameters and prevents overfitting
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Dense layer: Fully connected layer with 128 neurons
    # ReLU activation: Introduces non-linearity (outputs max(0, x))
    # This layer learns high-level combinations of features
    x = Dense(128, activation='relu', name='dense_1')(x)

    # Dropout: Randomly sets 50% of inputs to 0 during training
    # This is a regularization technique that prevents overfitting
    # Forces the network to learn redundant representations
    # Only active during training, not during inference
    x = Dropout(0.5, name='dropout')(x)

    # Output layer: Final classification layer
    # 2 neurons for 2 classes (with_mask, without_mask)
    # Softmax activation: Converts outputs to probabilities that sum to 1
    # Example output: [0.85, 0.15] means 85% confident it's "with_mask"
    predictions = Dense(num_classes, activation='softmax', name='output')(x)

    # Create the final model
    # Inputs: Images from base model
    # Outputs: Class probabilities from our custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

    print(f"\nCustom layers added:")
    print(f"  - GlobalAveragePooling2D")
    print(f"  - Dense(128, relu)")
    print(f"  - Dropout(0.5)")
    print(f"  - Dense({num_classes}, softmax)")

    return model


# ============================================================================
# STEP 2: COMPILE THE MODEL
# ============================================================================

def compile_model(model, learning_rate=0.0001):
    """
    Compile the model with optimizer, loss function, and metrics.

    Compilation configures the learning process:
    - Optimizer: Algorithm that updates weights (how we learn)
    - Loss: Function we're trying to minimize (what we optimize)
    - Metrics: What we track to evaluate performance

    Args:
        model: Keras model to compile
        learning_rate: Step size for gradient descent

    Returns:
        Compiled model
    """

    # Adam Optimizer: Adaptive learning rate optimization algorithm
    # Combines the best of RMSprop and Momentum
    # Adapts learning rate for each parameter automatically
    # Good default choice for most problems
    optimizer = Adam(learning_rate=learning_rate)

    # Sparse Categorical Crossentropy Loss:
    # - "Categorical": For classification problems
    # - "Crossentropy": Measures difference between predicted and true probabilities
    # - "Sparse": Labels are integers (0, 1) not one-hot encoded ([1,0], [0,1])
    # This is the standard loss for multi-class classification
    loss = 'sparse_categorical_crossentropy'

    # Metrics to track:
    # - Accuracy: Percentage of correct predictions
    # Simple but effective metric for balanced datasets
    metrics = ['accuracy']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    print("\nModel compiled successfully!")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Loss: {loss}")
    print(f"Metrics: {metrics}")

    return model


# ============================================================================
# STEP 3: SETUP CALLBACKS FOR TRAINING
# ============================================================================

def setup_callbacks(model_path):
    """
    Setup callbacks to monitor and control training.

    Callbacks are functions called at certain points during training:
    - After each epoch
    - After each batch
    - When certain conditions are met

    They help us:
    - Save the best model
    - Stop training early if not improving
    - Adjust learning rate dynamically

    Args:
        model_path: Where to save the best model

    Returns:
        List of callback objects
    """

    # ModelCheckpoint: Saves the model after every epoch
    # monitor='val_accuracy': Track validation accuracy
    # save_best_only=True: Only save when validation accuracy improves
    # mode='max': Higher is better for accuracy
    # This ensures we keep the best performing model
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # EarlyStopping: Stop training if model stops improving
    # monitor='val_loss': Watch validation loss
    # patience=10: Wait 10 epochs before stopping
    # restore_best_weights=True: Restore weights from best epoch
    # Prevents overfitting and saves time
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: Reduce learning rate when progress plateaus
    # monitor='val_loss': Watch validation loss
    # factor=0.5: Multiply learning rate by 0.5 when triggered
    # patience=5: Wait 5 epochs before reducing
    # min_lr=1e-7: Don't go below this learning rate
    # Helps fine-tune the model when learning slows down
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    print("\nCallbacks configured:")
    print("  - ModelCheckpoint: Save best model")
    print("  - EarlyStopping: Stop if no improvement for 10 epochs")
    print("  - ReduceLROnPlateau: Reduce LR if plateau for 5 epochs")

    return [checkpoint, early_stop, reduce_lr]


# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, callbacks, epochs=50, batch_size=8):
    """
    Train the model on our dataset.

    Training Process:
    1. Forward pass: Pass images through network, get predictions
    2. Calculate loss: Compare predictions to true labels
    3. Backward pass: Calculate gradients (how to adjust weights)
    4. Update weights: Use optimizer to update based on gradients
    5. Repeat for each batch and epoch

    Args:
        model: Compiled model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        callbacks: List of callback functions
        epochs: Number of complete passes through the dataset
        batch_size: Number of samples per gradient update

    Returns:
        history: Training history (loss, accuracy per epoch)
    """

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(X_train) // batch_size}")
    print(f"Max epochs: {epochs}")
    print("=" * 60 + "\n")

    # Preprocess images for MobileNetV2
    # MobileNetV2 expects inputs in range [-1, 1]
    # Our images are currently in [0, 1], so we need to scale them
    X_train_processed = preprocess_input(X_train * 255)  # Convert back to [0,255] then to [-1,1]
    X_val_processed = preprocess_input(X_val * 255)

    # Train the model
    # fit() performs the actual training
    # Returns a History object containing training metrics
    history = model.fit(
        X_train_processed, y_train,  # Training data
        validation_data=(X_val_processed, y_val),  # Validation data
        epochs=epochs,  # Maximum number of epochs
        batch_size=batch_size,  # Samples per batch
        callbacks=callbacks,  # Our configured callbacks
        verbose=1  # Show progress bar
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return history


# ============================================================================
# STEP 5: VISUALIZE TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.

    This helps us understand:
    - Is the model learning? (both losses decreasing)
    - Is it overfitting? (training good, validation bad)
    - When did it perform best?

    Args:
        history: Training history object from model.fit()
    """

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()


# ============================================================================
# STEP 6: EVALUATE ON TEST SET
# ============================================================================

def evaluate_model(model, X_test, y_test, categories=['with_mask', 'without_mask']):
    """
    Evaluate the trained model on unseen test data.

    This gives us an honest assessment of performance:
    - The model has never seen this data during training
    - Shows how well it generalizes to new images

    Args:
        model: Trained model
        X_test, y_test: Test data and labels
        categories: Class names for reporting
    """

    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)

    # Preprocess test data the same way as training data
    X_test_processed = preprocess_input(X_test * 255)

    # Get model predictions
    # predict() returns probability distributions for each image
    # Example: [[0.92, 0.08], [0.15, 0.85]] for 2 images
    predictions = model.predict(X_test_processed, verbose=0)

    # Convert probabilities to class labels
    # argmax() returns the index of the highest probability
    # [0.92, 0.08] -> 0 (with_mask), [0.15, 0.85] -> 1 (without_mask)
    y_pred = np.argmax(predictions, axis=1)

    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test, verbose=0)

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Classification Report: Shows precision, recall, F1-score per class
    # Precision: Of all predicted positive, how many were correct?
    # Recall: Of all actual positive, how many did we find?
    # F1-score: Harmonic mean of precision and recall
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Confusion Matrix: Shows where the model makes mistakes
    # Rows: True labels, Columns: Predicted labels
    # Diagonal: Correct predictions
    # Off-diagonal: Misclassifications
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.show()

    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main training pipeline:
    1. Load preprocessed data
    2. Build model with transfer learning
    3. Compile model
    4. Setup callbacks
    5. Train model
    6. Visualize results
    7. Evaluate on test set
    """

    print("=" * 60)
    print("FACE MASK DETECTION - MODEL TRAINING")
    print("Transfer Learning with MobileNetV2")
    print("=" * 60)

    # Check if processed data exists
    if not os.path.exists(DATA_FILE):
        print(f"\nError: {DATA_FILE} not found!")
        print("Please run '1_data_preparation.py' first to prepare the data.")
        exit()

    # Load preprocessed data
    print(f"\nLoading data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    print("Data loaded successfully!")

    # Build model
    print("\n[1/6] Building model...")
    model = build_model(IMG_SIZE, NUM_CLASSES)

    # Show model summary
    # This displays all layers, output shapes, and parameters
    print("\nModel Architecture:")
    model.summary()

    # Compile model
    print("\n[2/6] Compiling model...")
    model = compile_model(model, LEARNING_RATE)

    # Setup callbacks
    print("\n[3/6] Setting up callbacks...")
    callbacks = setup_callbacks(MODEL_SAVE_PATH)

    # Train model
    print("\n[4/6] Training model...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        callbacks, EPOCHS, BATCH_SIZE
    )

    # Plot training history
    print("\n[5/6] Plotting training history...")
    plot_training_history(history)

    # Evaluate on test set
    print("\n[6/6] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("1. Review training plots to check for overfitting")
    print("2. Check test accuracy and confusion matrix")
    print("3. Move to Phase 3: Real-time detection")
    print("=" * 60)