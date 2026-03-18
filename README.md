# EKG Sinyal Sınıflandırma ve TimeGAN ile Veri Artırma Projesi

Bu proje, **PTB-XL** veri kümesi kullanılarak 12 kanallı EKG sinyallerinin 5 ana tanı sınıfına (NORM, MI, STTC, CD, HYP) otomatik olarak sınıflandırılmasını hedefleyen kapsamlı bir derin öğrenme ve makine öğrenmesi pipeline'ıdır. Projenin en ayırt edici özelliği, veri dengesizliği (class imbalance) sorununu çözmek için **TimeGAN** (Time-series Generative Adversarial Networks) mimarisini kullanması ve sonuçların **Explainable AI (XAI)** yöntemleriyle açıklanabilir kılınmasıdır.

## 🚀 Öne Çıkan Özellikler

- **Hibrit Ensemble Mimarisi:** Derin Öğrenme (ResNet34, TCN, LSTM) ve Geleneksel Makine Öğrenmesi (XGBoost, LightGBM, Random Forest) modellerinin ağırlıklı ortalama ve stacking yöntemleriyle birleştirilmesi.
- **TimeGAN Entegrasyonu:** Gerçekçi sentetik sinyaller üreterek azınlık sınıfların (MI, HYP vb.) başarısını artıran veri artırma süreci.
- **İleri Seviye Sinyal İşleme:** Butterworth bant geçiren filtreleme ve Neurokit2 ile klinik öznitelik çıkarımı (HRV, QRS, P/T dalga genlikleri).
- **Açıklanabilir Yapay Zeka (XAI):** Modellerin karar mekanizmalarını görselleştirmek için SHAP, LIME ve Grad-CAM kullanımı.

## 📂 Proje Yapısı

```text
├── data/               # Ham ve işlenmiş veriler (PTB-XL veritabanı)
├── models/             # Eğitilmiş model ağırlıkları (.h5, .pkl, .weights.h5)
├── src/                # Pipeline çekirdek kodları
│   ├── data_loader.py  # PTB-XL meta veri yükleme ve sınıflandırma
│   ├── preprocessing.py# Sinyal filtreleme, segmentasyon ve ölçekleme
│   └── feature_extraction.py # Zaman/Frekans domeni ve morfolojik öznitelikler
├── notebooks/          # Adım adım uygulama ve analiz dosyaları
├── results/            # XAI çıktıları ve performans grafiklerinden örnekler
└── TimeGAN-pytorch/    # Sentetik veri üretimi için kullanılan GAN altyapısı
```

## 🛠️ Kurulum

Proje Python 3.12+ ortamında geliştirilmiştir. Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
# Ek olarak Neurokit2 ve keras-tcn gerekebilir
pip install neurokit2 keras-tcn
```

## 📊 İş Akışı (Pipeline)

1.  **Veri Hazırlığı:** PTB-XL veri kümesi yüklenir, eksik veriler (yaş, kilo vb.) median/mean ile doldurulur.
2.  **Sinyal İşleme:** 12 kanallı sinyaller 0.5-40 Hz arasında filtrelenir. Sinyaller 2.5 saniyelik segmentlere bölünür.
3.  **Veri Artırma (GAN):** Eğitim setindeki azınlık sınıflar için TimeGAN ile 1000 zaman adımlı sentetik veriler üretilir.
4.  **Öznitelik Çıkarımı:**
    - **İstatistiksel:** Ortalama, varyans, çarpıklık, basıklık vb.
    - **Frekans:** Welch PSD değerleri ve spektral entropi.
    - **Klinik:** Neurokit2 ile R-tepeleri tespiti, HRV parametreleri ve morfolojik dalga ölçümleri.
5.  **Modelleme:**
    - **ML Ensemble:** XGBoost, LightGBM, RF, SGD ve LinearSVC modelleri 'Stacking' yöntemiyle birleştirilir.
    - **DL Ensemble:** ResNet34, TCN ve LSTM modelleri 'Soft Voting' ile oylanır.
    - **Hibrit Final:** ML ve DL ensemble çıktıları ağırlıklı ortalama ile harmanlanır.
6.  **Açıklanabilirlik:** Grad-CAM ile sinyal üzerindeki odak noktaları, SHAP/LIME ile öznitelik önemleri analiz edilir.

## 📈 Performans Sonuçları

En iyi performans gösteren hibrit ensemble modelinin test seti sonuçları:
- **F1-Skor (Macro):** %98.05
- **Balanced Accuracy:** %97.56
- **ROC AUC:** %99.97

## 🔬 XAI Örnekleri

- **Grad-CAM:** Derin öğrenme modelinin QRS kompleksi üzerindeki aktivasyonunu doğrular.
- **SHAP:** Sınıflandırma kararında hangi HRV parametrelerinin daha etkili olduğunu global olarak gösterir.
- **LIME:** Belirli bir hasta tahmini için yerel öznitelik etkilerini açıklar.

## 🔗 Referanslar

- [PTB-XL Veriseti](https://physionet.org/content/ptb-xl/1.0.3/)
- [TimeGAN Pytorch Implementation](https://github.com/zwzhang123/TimeGAN-pytorch)
