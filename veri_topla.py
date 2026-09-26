import cv2
import mediapipe as mp
import numpy as np
import os

# El takip modülü
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) 

actions = [
    'Merhaba', 'Evet', 'Hayir', 'Tesekkurler', 'Lutfen', 
    'Nasilsin', 'Iyiyim', 'Yardim', 'Ben', 'Sen',
    'Dur', 'Tamam', 'Bekle', 'Sevmek', 'Gorusuruz' 
]

DATA_PATH = os.path.join('Veri_Seti') 
SEQUENCE_LENGTH = 30 

for word in actions:
    os.makedirs(os.path.join(DATA_PATH, word), exist_ok=True)

current_index = 0
action = actions[current_index]

# Kamerayı aç
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frames_collected = []
is_recording = False

print("Kamera açılıyor...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    existing_files = os.listdir(os.path.join(DATA_PATH, action))
    sequence_count = len(existing_files)

    #el tespiti
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if is_recording:
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            frames_collected.append(keypoints)

            cv2.putText(image, f"KAYIT: {len(frames_collected)}/{SEQUENCE_LENGTH}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if len(frames_collected) == SEQUENCE_LENGTH:
                npy_path = os.path.join(DATA_PATH, action, f"{sequence_count}.npy")
                np.save(npy_path, frames_collected)
                print(f"Kayıt tamamlandı: {action} -> {npy_path}")
                is_recording = False
                frames_collected = []

    if not is_recording:
        #arayüz bilgileri
        cv2.putText(image, f"Secili Kelime: {action}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Bu kelimedeki toplam kayıt: {sequence_count}", (15,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "[R] Kaydet | [N] Sonraki kelime | [B] Onceki | [Q] Cikis", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Isaret Dili Veri Toplama', image)

    #klavye kontrolleri
    key = cv2.waitKey(10) & 0xFF

    if key == ord('r') and not is_recording:
        is_recording = True
        frames_collected = []
    elif key == ord('n') and not is_recording:
        current_index = (current_index + 1) % len(actions)
        action = actions[current_index]
    elif key== ord('b') and not is_recording:
        current_index = (current_index - 1) % len(actions)
        action = actions[current_index]
    elif key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()