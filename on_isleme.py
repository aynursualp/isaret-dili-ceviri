import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = os.path.join('Veri_Seti')
actions = np.array([
    'Merhaba', 'Evet', 'Hayir', 'Tesekkurler', 'Lutfen',
    'Nasilsin', 'Iyiyim', 'Yardim', 'Ben', 'Sen', 
    'Dur', 'Tamam', 'Bekle', 'Sevmek', 'Gorusuruz'
])

#sözlük oluşturma
label_map = {label:num for num, label in enumerate(actions)}

#klasörleri okuma ve birleştirme
sequences, labels = [], []
print("Veriler okunuyor, bu birkaç saniye sürebilir.")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    #tüm npy dosyalarını listele
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]

    for file in files:
        res = np.load(os.path.join(action_path, file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("-" * 30)
print(f"Toplam veri sayısı: {X.shape[0]}")
print(f"X (Özellikler) Boyutu: {X.shape} -> (Kayıt, Kare, Koordinat)")
print(f"y (Etiketler) Boyutu: {y.shape}")
print("-" * 30)

#veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitime Ayrılan Veri: {X_train.shape[0]}")
print(f"Teste Ayrılan Veri: {X_test.shape[0]}")

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("Ön işleme tamamlandı.")