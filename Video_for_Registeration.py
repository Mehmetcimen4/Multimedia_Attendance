import cv2
import pandas as pd
import os
from datetime import datetime
import sys

if __name__ == "__main__":
    # Komut satırı argümanlarını al
    id = sys.argv[3]
    name = sys.argv[2]
    video = sys.argv[1]
    
    # Kullanıcıdan alınan bilgileri kullanarak işlem yap
    print("Student ID:", id)
    print("Student Name:", name)
    print("Video: ", video)

if os.path.exists('id-names.csv'):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
else:
    id_names = pd.DataFrame(columns=['id', 'name'])
    id_names.to_csv('id-names.csv', index=False)

# 'faces' klasörü yoksa oluştur
if not os.path.exists('faces'):
    os.makedirs('faces')


# ID'ye göre kullanıcı adını kontrol et
if id in id_names['id'].values:
    name = id_names[id_names['id'] == id]['name'].item()
    print(f'Welcome Back {name}!!')
else:
    os.makedirs(f'faces/{id}')
    id_names = id_names._append({'id': id, 'name': name}, ignore_index=True)
    id_names.to_csv('id-names.csv', index=False)


# Haarcascade yüz tanıma modeli yükleniyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video dosyasını yükleme
video_path = video  # Video dosyasının yolunu belirtin
video_capture = cv2.VideoCapture(video_path)

# Klasör oluşturma
output_folder = f'faces/{id}'
os.makedirs(output_folder, exist_ok=True)

face_count = 0
frame_count = 0
capture_interval = 10  # Capture every 10 frames

while True:
    # Kameradan bir kare alınıyor
    ret, frame = video_capture.read()
    frame_count += 1

    if not ret:
        break

    # Kareyi gri tona dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit etme
    if frame_count % capture_interval == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Her bir yüz için işlemler
        for (x, y, w, h) in faces:
            # Yüzü kare içine alma
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Algılanan yüzü gösterme
            face_region = gray[y:y+h, x:x+w]  # Gri tonlamaya dönüştürüyoruz

            # Yüzü dosyaya kaydetme
            face_file_name = os.path.join(output_folder, f"face_{face_count}.jpg")
            cv2.imwrite(face_file_name, face_region)
            face_count += 1

    # Ekranı gösterme
    cv2.imshow('Video', frame)

    # Çıkış için 'q' tuşuna basma kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kapatma
video_capture.release()
cv2.destroyAllWindows()
