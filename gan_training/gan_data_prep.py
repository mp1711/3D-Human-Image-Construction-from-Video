import cv2
import os

def create_pix2pix_dataset(segmented_dir, extracted_dir, output_dir, size=(256, 256)):
    """
    Combine segmented and extracted images side by side for Pix2Pix GAN training.
    
    Args:
        segmented_dir (str): Path to the folder containing segmented images.
        extracted_dir (str): Path to the folder containing extracted images.
        output_dir (str): Path to save concatenated images.
        size (tuple): Size to resize images (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    segmented_files = os.listdir(segmented_dir)

    for filename in segmented_files:
        segmented_path = os.path.join(segmented_dir, filename)
        extracted_path = os.path.join(extracted_dir, filename)

        # Ensure both files exist
        if os.path.exists(segmented_path) and os.path.exists(extracted_path):
            segmented_img = cv2.imread(segmented_path)
            extracted_img = cv2.imread(extracted_path)

            # Resize both images
            resized_segmented = cv2.resize(segmented_img, size)
            resized_extracted = cv2.resize(extracted_img, size)

            # Concatenate images horizontally
            combined_img = cv2.hconcat([resized_segmented, resized_extracted])

            # Save combined image
            output_path = os.path.join(output_dir, os.path.basename(segmented_dir) + '-' + filename)
            cv2.imwrite(output_path, combined_img)

# Paths to input directories and output directory
segmented_dirs = [
    './assets/segmented_images/female-1-casual',
    './assets/segmented_images/female-3-casual',
    './assets/segmented_images/male-2-sport'
]

extracted_dirs = [
    './assets/extracted_images/female-1-casual',
    './assets/extracted_images/female-3-casual',
    './assets/extracted_images/male-2-sport'
]

output_dir = './assets/gan_train/train'

# Create Pix2Pix dataset
for segmented_dir, extracted_dir in zip(segmented_dirs, extracted_dirs):
    create_pix2pix_dataset(segmented_dir, extracted_dir, output_dir)
