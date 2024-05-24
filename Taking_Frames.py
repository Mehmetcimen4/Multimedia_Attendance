import os
import numpy as np
import cv2 as cv
from datetime import datetime

face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')

# Yüz tanıma fonksiyonu
def face_recognition(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Kareyi gri tonlamalıya dönüştür
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Yüzün etrafına kırmızı dikdörtgen çiz

        face_region = grey[y:y + h, x:x + w]
        if np.average(face_region) > 50:
            face_img = cv.resize(face_region, (220, 220))
            img_name = f'face_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpeg'
            cv.imwrite(os.path.join('Detected_Faces', img_name), face_img)  # Yüzü dosyaya kaydet

# Resimlerin bulunduğu dizini tarayarak yüz tespiti yap
for img_name in os.listdir('Photos\\Osman'):
    img_path = os.path.join('Photos\\Osman', img_name)
    img = cv.imread(img_path)
    face_recognition(img)
