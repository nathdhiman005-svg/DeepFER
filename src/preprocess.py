import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import random
import tensorflow as tf

BATCH_SIZE = 32
ORIGINAL_IMAGE_SIZE = (48, 48)
TARGET_IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def get_dataset_paths():
    """
    Returns the paths to the training and testing dataset directories.
    Assumes the dataset is located in the 'dataset' folder in the root directory.
    """
    base_dir = Path("dataset")
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    return train_dir, test_dir

def analyze_dataset(train_dir, test_dir):
    """
    Traverses the dataset to count images, verify dimensions, and detect corrupted files.
    """
    directories_to_check = [train_dir, test_dir]
    
    total_images = 0
    train_images_count = 0
    test_images_count = 0
    emotion_counts = Counter()
    
    # Using a set to keep track of unique image shapes (height, width, channels)
    unique_shapes = set()
    corrupted_images = []
    
    # Dictionary to store all valid image paths per emotion to pick a random sample later
    valid_image_paths_per_emotion = {}
    
    # Image statistics tracking
    global_min = float('inf')
    global_max = float('-inf')
    total_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    pixel_histogram = np.zeros(256, dtype=np.float64)
    data_type = None

    for directory in directories_to_check:
        if not directory.exists():
            continue
            
        for emotion_dir in directory.iterdir():
            # Skip files like .DS_Store or similar
            if not emotion_dir.is_dir():
                continue
                
            emotion_name = emotion_dir.name
            
            if emotion_name not in valid_image_paths_per_emotion:
                valid_image_paths_per_emotion[emotion_name] = []
            
            for image_path in emotion_dir.glob("*"):
                # Check if it's a file (ignoring any subdirectories)
                if image_path.is_file():
                    total_images += 1
                    
                    if directory == train_dir:
                        train_images_count += 1
                    elif directory == test_dir:
                        test_images_count += 1
                        
                    emotion_counts[emotion_name] += 1
                    
                    # Try to read the image using OpenCV to check for corruption and dimensions
                    # cv2.IMREAD_UNCHANGED ensures we read the actual number of channels
                    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        corrupted_images.append(str(image_path))
                    else:
                        shape = img.shape
                        unique_shapes.add(shape)
                        valid_image_paths_per_emotion[emotion_name].append(str(image_path))
                        
                        # Update image statistics
                        if data_type is None:
                            data_type = str(img.dtype)
                            
                        global_min = min(global_min, float(img.min()))
                        global_max = max(global_max, float(img.max()))
                        
                        img_float = img.astype(np.float64)
                        total_pixels += img.size
                        sum_pixels += np.sum(img_float)
                        sum_sq_pixels += np.sum(img_float ** 2)
                        
                        hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
                        pixel_histogram += hist
                        
    # Randomly select one sample image path per emotion
    sample_images = {}
    for emotion, paths in valid_image_paths_per_emotion.items():
        if paths:
            sample_images[emotion] = random.choice(paths)
            
    mean_pixel = sum_pixels / total_pixels if total_pixels > 0 else 0
    variance = (sum_sq_pixels / total_pixels) - (mean_pixel ** 2) if total_pixels > 0 else 0
    std_pixel = np.sqrt(variance) if variance > 0 else 0

    return {
        'total_images': total_images,
        'train_images_count': train_images_count,
        'test_images_count': test_images_count,
        'emotion_counts': emotion_counts,
        'unique_shapes': unique_shapes,
        'corrupted_images': corrupted_images,
        'sample_images': sample_images,
        'data_type': data_type,
        'global_min': int(global_min) if global_min != float('inf') else 0,
        'global_max': int(global_max) if global_max != float('-inf') else 0,
        'mean_pixel': mean_pixel,
        'std_pixel': std_pixel,
        'pixel_histogram': pixel_histogram
    }

def print_summary(stats):
    """
    Prints a clean summary of the dataset in the terminal, matching the expected output.
    """
    print("============================")
    print("DATASET SUMMARY")
    print("============================")
    print(f"\nTraining Images: {stats['train_images_count']}")
    print(f"\nTest Images: {stats['test_images_count']}")
    print(f"\nTotal Images: {stats['total_images']}")
    print("\nEmotion Distribution:\n")
    
    # Sort emotions alphabetically for clean output (Angry, Disgust, Fear, etc.)
    for emotion, count in sorted(stats['emotion_counts'].items()):
        print(f"{emotion.capitalize()}: {count}")
        
    # Extract dimensions and channels from unique shapes
    if len(stats['unique_shapes']) == 1:
        shape = list(stats['unique_shapes'])[0]
        # shape might be (Height, Width) for grayscale or (Height, Width, Channels)
        if len(shape) == 2:
            height, width = shape
            channels = 1
        else:
            height, width, channels = shape
            
        print(f"\nImage Size: {width}x{height}")
        print(f"\nChannels: {channels}")
    else:
        print("\nImage Size: Multiple sizes detected")
        print("\nChannels: Multiple channel formats detected")
        
    print(f"\nCorrupted Images: {len(stats['corrupted_images'])}\n")
    print("============================")

def print_image_statistics(stats):
    """
    Prints the calculated image statistics including data type, min, max, mean, and std deviation.
    """
    print("============================")
    print("IMAGE STATISTICS")
    print("============================")
    print(f"\nImage Data Type: {stats.get('data_type', 'Unknown')}")
    print(f"\nMinimum Pixel: {stats.get('global_min', 0)}")
    print(f"\nMaximum Pixel: {stats.get('global_max', 0)}")
    print(f"\nMean Pixel: {stats.get('mean_pixel', 0):.2f}")
    print(f"\nStandard Deviation: {stats.get('std_pixel', 0):.2f}")
    print("\n============================")

def plot_pixel_intensity_histogram(pixel_histogram):
    """
    Generates, saves, and displays a pixel intensity histogram.
    The histogram is saved to 'outputs/plots/pixel_intensity_histogram.png'.
    """
    if pixel_histogram is None:
        return
        
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pixel_intensity_histogram.png"
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(256), pixel_histogram, color='gray', width=1.0)
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save the histogram automatically
    plt.savefig(output_path)
    print(f"\nHistogram saved to: {output_path}")
    
    # Display the histogram
    plt.show()

def visualize_samples(sample_images):
    """
    Displays one random sample image from each emotion class using Matplotlib.
    """
    if not sample_images:
        print("No sample images to display.")
        return
        
    num_emotions = len(sample_images)
    # Create a figure to hold the subplots side by side
    fig, axes = plt.subplots(1, num_emotions, figsize=(15, 3))
    
    # Handle the edge case where there is only one emotion
    if num_emotions == 1:
        axes = [axes]
        
    for ax, (emotion, image_path) in zip(axes, sorted(sample_images.items())):
        # Read the image to display it
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # If the image is single-channel grayscale, specify the colormap
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            # OpenCV loads color images in BGR format, Matplotlib expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
        ax.set_title(emotion.capitalize())
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_emotion_distribution(emotion_counts):
    """
    Generates and displays a bar chart showing the number of images in each emotion class.
    """
    if not emotion_counts:
        return
        
    emotions = sorted(emotion_counts.keys())
    counts = [emotion_counts[em] for em in emotions]
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(emotions, counts, color='skyblue')
    
    # Add title and labels
    plt.title('Emotion Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    
    # Rotate x-axis labels if needed for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def load_tf_train_datasets(train_dir):
    """
    Loads and splits the training dataset into training and validation subsets.
    Uses an 80/20 split with a fixed seed for reproducibility.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode='grayscale',
        image_size=ORIGINAL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode='grayscale',
        image_size=ORIGINAL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=42
    )
    return train_ds, val_ds

def load_tf_test_dataset(test_dir):
    """
    Loads the testing dataset using TensorFlow's image_dataset_from_directory.
    Images are loaded as grayscale, without shuffling.
    """
    return tf.keras.utils.image_dataset_from_directory(
        test_dir,
        color_mode='grayscale',
        image_size=ORIGINAL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def preprocess_image(image, label):
    """
    Applies preprocessing operations to an image and returns it with its label.
    """
    image = tf.image.resize(image, TARGET_IMAGE_SIZE)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def preprocess_single_image(image):
    """
    Preprocesses a single image for inference.

    Steps:
    1. Convert RGB to Grayscale.
    2. Resize to FER2013 resolution (48x48).
    3. Convert grayscale back to RGB.
    4. Resize to MobileNetV2 input size (224x224).
    5. Apply MobileNetV2 preprocessing.
    """

    # ------------------------------------------------------
    # Convert RGB → Grayscale
    # ------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    # ------------------------------------------------------
    # Resize to FER2013 size
    # ------------------------------------------------------

    image = cv2.resize(
        image,
        ORIGINAL_IMAGE_SIZE,
    )

    # ------------------------------------------------------
    # Convert Grayscale → RGB
    # ------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB,
    )

    # ------------------------------------------------------
    # Resize to MobileNetV2 size
    # ------------------------------------------------------

    image = cv2.resize(
        image,
        TARGET_IMAGE_SIZE,
    )

    # ------------------------------------------------------
    # Convert to float32
    # ------------------------------------------------------

    image = image.astype(np.float32)

    # ------------------------------------------------------
    # MobileNetV2 preprocessing
    # ------------------------------------------------------

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    # ------------------------------------------------------
    # Add batch dimension
    # ------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0,
    )

    return image
    
def preprocess_dataset(ds):
    """
    Preprocesses the dataset for MobileNetV2:
    1. Resizes images from ORIGINAL_IMAGE_SIZE to TARGET_IMAGE_SIZE.
    2. Converts grayscale (1 channel) to RGB (3 channels) using TF operations.
    3. Applies MobileNetV2 specific preprocessing (scaling to [-1, 1]).
    4. Prefetches for optimal performance using AUTOTUNE.
    """
    # Apply the combined preprocessing function
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # Prefetch dataset elements
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def print_preprocessing_summary(train_ds, val_ds, test_ds):
    """
    Prints a summary of the fully preprocessed datasets.
    """
    print("\n==========================")
    print("PREPROCESSING SUMMARY")
    print("==========================")
    print(f"\nTraining Dataset: {len(train_ds)}")
    print(f"\nValidation Dataset: {len(val_ds)}")
    print(f"\nTesting Dataset: {len(test_ds)}")
    
    for images, labels in train_ds.take(1):
        print(f"\nImage Shape: {images.shape}")
        print(f"\nLabel Shape: {labels.shape}")
        print(f"\nBatch Size: {BATCH_SIZE}")
        break
        
    print("\n==========================\n")

def display_processed_sample(train_ds, class_names):
    """
    Displays one processed image and its label to verify the preprocessing pipeline.
    """
    for images, labels in train_ds.take(1):
        img = images[0].numpy()
        label_idx = labels[0].numpy()
        class_name = class_names[label_idx]
        
        # MobileNetV2 preprocess_input scales to [-1, 1], so we unscale to [0, 1] for displaying
        img_display = (img + 1.0) / 2.0
        img_display = np.clip(img_display, 0.0, 1.0)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(img_display)
        plt.title(f"Processed: {class_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        break


def main():
    """
    Main execution block that coordinates the dataset analysis and visualization.
    """
    # 1. Get paths to the training and testing directories
    train_dir, test_dir = get_dataset_paths()
    
    # 2. Analyze the dataset to get counts, shapes, and detect corruptions
    stats = analyze_dataset(train_dir, test_dir)
    
    # 3. Print the formatted summary to the terminal
    print_summary(stats)
    
    # 4. Print the newly added image statistics
    print_image_statistics(stats)
    
    # 5. Display a random sample image from each class
    visualize_samples(stats['sample_images'])
    
    # 6. Display a bar chart of the emotion distribution
    plot_emotion_distribution(stats['emotion_counts'])
    
    # 7. Generate, save, and display the pixel intensity histogram
    plot_pixel_intensity_histogram(stats.get('pixel_histogram'))

if __name__ == "__main__":
    main()
