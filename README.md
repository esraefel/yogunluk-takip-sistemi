# 📊 Gerçek Zamanlı Alan Kullanım ve Yoğunluk Takip Sistemi

Bu proje, mağaza veya belirli alanlardaki insan yoğunluğunu, giriş ve çıkış sayılarını gerçek zamanlı olarak takip etmek amacıyla geliştirilmiş bir **Yönetim Bilişim Sistemleri (MIS)** projesidir.

## 🚀 Özellikler
- **YOLOv8 ve ByteTrack** ile yüksek doğruluklu insan takibi.
- **Streamlit** tabanlı modern, sade ve kurumsal dashboard arayüzü.
- **SQLite Veritabanı** entegrasyonu ile analiz verilerinin kalıcı olarak saklanması.
- Veri analizi için **JSON ve CSV** formatlarında otomatik günlük rapor üretimi.
- Sistemin kararlılığını izlemek için detaylı **Python Logging** altyapısı.

## 🛠️ Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin ve proje klasöründe bir terminal açın.
2. Gerekli kütüphaneleri yükleyin:
```bash
# pip install streamlit opencv-python pandas plotly numpy ultralytics. 
3. Analiz etmek istediğiniz video dosyasını videos klasörünün içerisine aktarın ve adını test_video.mp4 yapın.
4. Sistemi başlatın:

# streamlit run app.py
 