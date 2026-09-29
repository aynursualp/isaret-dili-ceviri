# 🤟 Türk İşaret Dili (TİD) Gerçek Zamanlı Yapay Zeka Çevirmeni

Bu proje, kamera üzerinden alınan gerçek zamanlı video akışındaki el hareketlerini analiz ederek **Türk İşaret Dili'ndeki (TİD) 15 farklı kelimeyi** yapay zeka ile anında metne çeviren uçtan uca bir derin öğrenme (Deep Learning) sistemidir.

Model, el iskelet koordinatlarını çıkarmak için **MediaPipe**, zaman serisi (zaman içindeki ardışık hareket) analizi için ise **TensorFlow/Keras** tabanlı çok katmanlı **LSTM (Long Short-Term Memory)** sinir ağlarını kullanmaktadır.

## 🚀 Özellikler
* **Gerçek Zamanlı Çeviri:** Canlı kamera akışı üzerinden düşük gecikme ile tahminleme.
* **3D İskelet Çıkarımı:** MediaPipe ile her bir el için uzaydaki 21 farklı eklem noktasının (x, y, z koordinatları) anlık matris dönüşümü.
* **Zaman Serisi Analizi:** Hareketlerin başlangıç ve bitiş evrelerini anlamlandırabilmek için her tahminde son 30 karenin (frame) ardışık bir dizi olarak işlenmesi.
* **Otomatik Optimizasyon:** Eğitim sırasında ezberlemeyi (Overfitting) önlemek için Dropout katmanları ve model doygunluğa ulaştığında eğitimi kesen `EarlyStopping` mekanizması.

## 🛠️ Kullanılan Teknolojiler

| Kategori | Teknoloji / Kütüphane |
| :--- | :--- |
| **Dil** | Python |
| **Bilgisayarlı Görü** | OpenCV, MediaPipe |
| **Derin Öğrenme** | TensorFlow, Keras (LSTM) |
| **Veri İşleme** | NumPy, Scikit-Learn |

## 📂 Proje Yapısı ve Boru Hattı (Pipeline)
Proje, modüler bir yapıda 3 temel aşamaya bölünmüştür:

1. **`on_isleme.py`:** Toplanan ham video verilerini okur, el koordinatlarını çıkarır, verileri One-Hot Encoding ile etiketleyip modeli besleyecek `(462, 30, 63)` boyutundaki 3B NumPy tensörlerine dönüştürür.
2. **`model_egit.py`:** Hazırlanan dizileri alarak 3 adet LSTM ve sonrasındaki Dense karar katmanlarından oluşan sinir ağını eğitir. `val_categorical_accuracy` metriklerini izler ve en optimum ağırlıkları `.h5` formatında dışa aktarır.
3. **`gercek_zamanli_ceviri.py`:** Eğitilmiş modeli ve donanım kamerasını ayağa kaldırır. Canlı akıştan 30 karelik kayan pencereler (sliding window) oluşturarak modelden anlık olasılık tahminleri alır ve belirlenen eşik değerini (%85) aşan sonuçları ekrana basar.

## ⚠️ Kurulum ve Bağımlılık Notları
TensorFlow, JAX ve MediaPipe kütüphanelerinin çekirdek mimari uyuşmazlıklarını (Dependency Hell) önlemek için spesifik sürümlerin kullanılması gerekmektedir. Özellikle `StringDType` ve `_ARRAY_API` hataları almamak adına NumPy sürümü 2.0'ın altında tutulmuştur.

Çalışma ortamını izole bir `venv` içinde kurduktan sonra aşağıdaki komutlarla stabil bağımlılıkları yükleyebilirsiniz:

```bash
pip install "numpy<2"
pip install jax==0.4.23 jaxlib==0.4.23
pip install tensorflow opencv-python mediapipe scikit-learn
````
🧠 Sınıflandırılan Kelimeler
Model şu an için aşağıdaki 15 kelimelik konsept üzerinde eğitilmiş olup, veri seti genişletmeye açık bir mimaride tasarlanmıştır:
Merhaba, Evet, Hayır, Teşekkürler, Lütfen, Nasılsın, İyiyim, Yardım, Ben, Sen, Dur, Tamam, Bekle, Sevmek, Görüşürüz
