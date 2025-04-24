import os
import random
import shutil

# Paths to train and test directories
train_dir = './assets/gan_train/train'
test_dir = './assets/gan_train/test'

# Create the test directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

# Load all image filenames in the train directory
image_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]

# Shuffle the image filenames
random.seed(42)  # Set seed for reproducibility
random.shuffle(image_files)

# Move a portion of images from train to test (e.g., 10% of the total)
num_images_to_move = int(len(image_files) * 0.1)
images_to_move = image_files[:num_images_to_move]

# Move images to the test directory
for file in images_to_move:
    shutil.move(os.path.join(train_dir, file), os.path.join(test_dir, file))

# Output the number of images moved
print(f"Moved {len(images_to_move)} images to the test directory.")
