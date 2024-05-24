import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import csv
import os
import pandas as pd

# MTCNN and InceptionResnetV1 model loading
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load necessary data for recognition (fill with actual data)
from torchvision import datasets
from torch.utils.data import DataLoader

# Veri setini yükleyin
dataset = datasets.ImageFolder('photos')  # 'photos' dizini etiketli alt dizinlere sahip olmalı
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0])
embedding_list = []  # List of pre-computed face embeddings
name_list = []       # List of corresponding names
for img, idx in loader:
    img = img.convert('RGB')
    faces, probs = mtcnn(img, return_prob=True)
    if faces is not None:
        for face, prob in zip(faces, probs):
            if prob > 0.90:
                emb = resnet(face.unsqueeze(0))
                embedding_list.append(emb.detach())
                name_list.append(idx_to_class[idx])

# Dictionary to keep track of recognized individuals
recognition_history = {}
consistent_frames_required = 15
csv_filename = "attendance_list.csv"

def update_recognition_history(name):
    if name not in recognition_history:
        recognition_history[name] = 1  # Initialize with a count
    else:
        recognition_history[name] += 1  # Increment on recognition

    # Check if all recent recognitions are the same and meet the required frame count
    if recognition_history[name] == consistent_frames_required:
        if name == "Unknown":
            recognition_history.clear()
        else:
            print(f"{name} is entering the room.")
            write_to_csv(name)
            recognition_history.clear()  # Reset history after recognition

def write_to_csv(name):
    # Check if the name is already in the CSV file
    file_exists = os.path.isfile(csv_filename)
    if file_exists:
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == name:
                    return  # Name already exists, so do nothing

    # Write the name to the CSV file if it doesn't already exist
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Attendance"])
        writer.writerow([name, "+"])

def recognize_person(face):
    emb = resnet(face.unsqueeze(0))
    min_dist = float('inf')
    min_dist_idx = None
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.nn.functional.pairwise_distance(emb, emb_db).item()
        if dist < min_dist:
            min_dist = dist
            min_dist_idx = idx
    if min_dist < 1.5:  # Threshold value
        return name_list[min_dist_idx]
    return "Unknown"

def show_csv():
    if os.path.isfile(csv_filename):
        df = pd.read_csv(csv_filename)
        print(df)
    else:
        print("CSV file not found.")

def clear_csv():
    with open(csv_filename, mode='w') as file:
        file.truncate()
    print("CSV file cleared.")

def process_video(source=0):
    # Kamera kaynağını başlat
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Yüzleri tespit et
        boxes, probs = mtcnn.detect(frame_rgb)

        # Yüzler için döngü
        if boxes is not None:
            highest_prob_idx = np.argmax(probs)  # En yüksek olasılığa sahip yüzü bul
            highest_prob_box = boxes[highest_prob_idx]
            highest_prob = probs[highest_prob_idx]

            x1, y1, x2, y2 = [int(b) for b in highest_prob_box]
            box_area = (x2 - x1) * (y2 - y1)
            print(f"Bounding box area: {box_area}")

            if box_area < 2000:
                recognition_history.clear()  # Reset history if face too small
                continue  # Skip the smaller faces

            face = frame_rgb[y1:y2, x1:x2]
            if face.size > 0:
                try:
                    face = cv2.resize(face, (240, 240))
                    face = torch.tensor(face).permute(2, 0, 1).float().div(255)
                    name = recognize_person(face)
                    update_recognition_history(name)  # Update recognition history
                    # Draw rectangle and text on the RGB frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except cv2.error as e:
                    print(f"Error resizing face: {e}")

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    process_video(0)
    """while True:
        choice = input("Choose Source:\n1. Webcam\n2. Video File\n3. Show CSV\n4. Clear CSV\n5. Exit\n")

        if choice == '1':
            process_video(0)
        elif choice == '2':
            video_path = input("Enter the path of the video file: ")
            process_video(video_path)
        elif choice == '3':
            show_csv()
        elif choice == '4':
            clear_csv()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")"""