import cv2
import os
from datetime import datetime
# Haarcascade yüz tanıma modeli yükleniyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video dosyasını yükleme
video_path = "C:\\Users\\4242o\\Desktop\\Face-Recognition-with-OpenCV-main\\Videos\\First_one.mp4"  # Video dosyasının yolunu belirtin
video_capture = cv2.VideoCapture(video_path)

# Klasör oluşturma
output_folder = f"detected_faces\\faces_{datetime.now().strftime('%m_%d_%H%M')}"
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
