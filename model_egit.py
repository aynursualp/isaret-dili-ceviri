import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

actions = np.array([
    'Merhaba','Evet', 'Hayir', 'Tesekkurler', 'Lutfen',
    'Nasilsin', 'Iyiyim', 'Yardim', 'Ben', 'Sen',
    'Dur', 'Tamam', 'Bekle', 'Sevmek', 'Gorusuruz'
])

#lstm mimarisi kurma
model = Sequential()

#ilk katman 30 kare (zaman adımı) ve 63 koordinat
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(Dropout(0.2)) #nöronların %20 sini kapatarak overfittingi engellemek için

model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))

#zaman serisi analizini bitirip sonuca bağlama
model.add(LSTM(64, return_sequences=False, activation='relu'))

#karar katmanları (ann deki standart gizli katmanlar)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

#çıkış katmanı, 15 kelime için 15 nöron, softmax aktivasyonu ile olasılık dağılımı
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Model mimarisi hazırlandı, eğitime başlanıyor..")

#model test verisinde 15 tur boyunca iyileşme göstermezse eğitimi kes ve en iyi ağırlıkları geri yükle
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=150,
                    validation_data=(X_test, y_test),
                    callbacks=[es])

model.save('isaret_dili_modeli.h5')
print("\nEğitim tamamlandı!")