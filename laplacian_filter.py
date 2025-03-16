import cv2
import numpy as np
import os

# Paths
INPUT_FOLDER = './assets/extracted_images'
OUTPUT_FOLDER = './assets/laplacian_filtered_images'

BLUR_THRESHOLD = 100.0  

def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return variance < threshold  # Return True if blurry

def deblur_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(gray_image, -1, sharpening_kernel)

def process_images(input_folder, output_folder):
    for person_dir in os.listdir(input_folder):
        person_input_path = os.path.join(input_folder, person_dir)
        person_output_path = os.path.join(output_folder, person_dir)
        os.makedirs(person_output_path, exist_ok=True)

        if os.path.isdir(person_input_path):
            for file in os.listdir(person_input_path):
                if file.endswith('.jpg'):
                    image_path = os.path.join(person_input_path, file)
                    image = cv2.imread(image_path)

                    if image is not None:
                        if is_blurry(image):
                            # Convert to grayscale before sharpening
                            processed_image = deblur_image(image)
                        else:
                            # Convert sharp images to grayscale as well
                            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        output_path = os.path.join(person_output_path, file)
                        cv2.imwrite(output_path, processed_image)

            print(f"Processed images for {person_dir}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Blur detection and deblurring complete.")
