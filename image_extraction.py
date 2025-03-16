import cv2
import os

# Paths
INPUT_FOLDER = './assets/people_snapshot_public'
OUTPUT_FOLDER = './assets/extracted_images'

FIXED_FPS = 50

def extract_frames(video_path, output_path, fps=FIXED_FPS):
    os.makedirs(output_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_step = max(1, original_fps // fps)

    count = 0
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_step == 0:
            image_path = os.path.join(output_path, f'frame_{count:04d}.jpg')
            cv2.imwrite(image_path, frame)
            count += 1
        
        frame_index += 1
    
    cap.release()
    print(f"Extracted {count} frames from {video_path}")

def process_videos(input_folder, output_folder):
    for person_dir in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_dir)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.endswith('.mp4'):
                    video_path = os.path.join(person_path, file)
                    output_path = os.path.join(output_folder, person_dir)
                    extract_frames(video_path, output_path)

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    process_videos(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Frame extraction complete.")
