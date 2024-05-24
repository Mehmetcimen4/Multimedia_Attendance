import os
import numpy as np
import pandas as pd
import cv2 as cv
from datetime import datetime
import sys

if __name__ == "__main__":
    # Komut satırı argümanlarını al
    id = sys.argv[2]
    name = sys.argv[1]

    # Kullanıcıdan alınan bilgileri kullanarak işlem yap
    print("Student ID:", id)
    print("Student Name:", name)

# Veritabanı dosyası varsa yükle, yoksa boş bir DataFrame oluştur ve kaydet
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


# Kamera yakalama başlat
camera = cv.VideoCapture(0)
face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')

photos_taken = 0  # Alınan fotoğraf sayısını takip etmek için değişken
total_photos_needed = 30  # Toplamda alınacak fotoğraf sayısı

while photos_taken < total_photos_needed and cv.waitKey(1) & 0xFF != ord('q'):  # 'q' tuşuna basılmadığı ve istenen fotoğraf sayısına ulaşmadığı sürece döngüyü devam ettir
    _, img = camera.read()  # Kameradan bir kare oku

    faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 1:  # Birden fazla yüz tespit edilirse
        # Uyarı mesajını ekrana yaz
        cv.putText(img, 'WARNING: Multiple faces detected!', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:  # Tek bir yüz tespit edildiyse
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Yüzün etrafına kırmızı dikdörtgen çiz

            face_region = img[y:y + h, x:x + w]
            if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
                face_img = cv.resize(face_region, (220, 220))
                img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
                cv.imwrite(f'faces/{id}/{img_name}', face_img)
                photos_taken += 1
                print(f'{photos_taken}/{total_photos_needed} -> Photos taken!')  # Alınan fotoğraf sayısını yazdır

        # Kameranın görüntüsüne alınan fotoğraf sayısını yazdır
        cv.putText(img, f'Photos taken: {photos_taken}/{total_photos_needed}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.imshow('Face', img)  # İşlenmiş kareyi ekranda göster

# Kamera kaynağını serbest bırak ve OpenCV pencerelerini kapat
camera.release()
cv.destroyAllWindows()
