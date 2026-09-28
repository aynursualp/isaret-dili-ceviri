import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

print("model yükleniyor")
model = load_model('isaret_dili_modeli.h5')
actions = ['Merhaba', 'Evet', 'Hayir', 'Tesekkurler', 'Lutfen', 
           'Nasilsin', 'Iyiyim', 'Yardim', 'Ben', 'Sen', 
           'Dur', 'Tamam', 'Bekle', 'Sevmek', 'Gorusuruz']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

sequence = []
threshold = 0.85 #sadece %85 ve üzeri eminsen ekrana yazdır

print("kamera başlatılıyor..")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: 
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
       
        #o anki karenin 63 koordinatını al
        keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

        #hafızaya ekle
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            #veriyi (1, 30, 63) formatında besle
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            #en yüksek olasılıklı tahmini bul
            best_guess_index = np.argmax(res)
            confidence = res[best_guess_index]

            if confidence > threshold:
                predicted_word = actions[best_guess_index]

                cv2.putText(image, f"{predicted_word} (%{int(confidence*100)})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Isaret Dili Yapay Zeka Cevirmeni', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

