import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import csv
import os
import pandas as pd
import sys
from datetime import datetime
import time

# Attendance_List klasörünün var olup olmadığını kontrol et, yoksa oluştur
attendance_dir = "Attendance_List"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# CSV dosyasının adı ve başlıkları
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = os.path.join(attendance_dir, f"participant_list_{current_time}.csv")
fields = ['Time', 'Participant', 'Status']

# id-names.csv dosyasını oku ve 'id' ve 'name' sütunlarını al
id_names = pd.read_csv('id-names.csv')
id_names = id_names[['id', 'name']]

# 'id' ve 'name' sütunlarından oluşan bir DataFrame'den sözlük oluştur
users = id_names.set_index('id')['name'].to_dict()

# MTCNN and InceptionResnetV1 model loading
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load necessary data for recognition (fill with actual data)
from torchvision import datasets
from torch.utils.data import DataLoader

# Load dataset
dataset = datasets.ImageFolder('faces')  # 'faces' dizini etiketli alt dizinlere sahip olmalı
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
                class_id = idx_to_class[idx]
                # Use name from users if available
                name_list.append(users.get(class_id, class_id))

# Dictionary to keep track of recognized individuals
recognition_history = {}
consistent_frames_required = 15

def update_recognition_history(name,id):
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
            write_to_csv(name,id)
            recognition_history.clear()  # Reset history after recognition

def write_to_csv(name,id):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Check if the name is already in the CSV file
    file_exists = os.path.isfile(csv_filename)
    if file_exists:
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 1 and row[1] == name:
                    return  # Name already exists, so do nothing

    # Write the name to the CSV file if it doesn't already exist
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)  # Write the header
        writer.writerow([current_time, name,id])

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

    total_time = 0
    frame_count = 0

    while True:
        start_time = time.time()  # Start time of frame processing

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
                    id = name
                    name = users[int(name)]
                    if name in users.values():
                        update_recognition_history(name,id)  # Update recognition history
                    # Draw rectangle and text on the RGB frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except cv2.error as e:
                    print(f"Error resizing face: {e}")

        # Display the resulting frame
        cv2.imshow('Video', frame)

        end_time = time.time()  # End time of frame processing
        frame_processing_time = end_time - start_time
        total_time += frame_processing_time
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_processing_time = total_time / frame_count
        print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")
    else:
        print("No frames processed.")

if __name__ == "__main__":
    try:
        video = sys.argv[1]
    except IndexError:
        video = 0
    process_video(video)
