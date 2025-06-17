# Chess Player Helper

🎓 *Bitirme Projesi - Görüntü İşleme Tekniği İle Satranç Oyun Hamlesi Tespiti ve Önermesi*

Bu proje, gerçek zamanlı görüntü işleme, nesne tanıma ve satranç kuralları doğrultusunda bir oyuncuya en iyi hamleyi öneren bir *satranç asistanı* sistemidir. Bilgisayar kamerasından alınan görüntüdeki satranç tahtasını analiz eder, taşların yerlerini belirler, mevcut durumu FEN formatına çevirir ve Stockfish motoru üzerinden en iyi hamleyi sunar.

## 🔧 Teknolojiler

- Python 3.9+
- OpenCV
- YOLOv8 (Ultralytics)
- PyQt5
- Chess (python-chess)
- Stockfish Online API
- NumPy

## 🧠 Proje Özeti

Chess Player Helper, fiziksel bir satranç tahtasının görüntüsünü işleyerek:
- YOLOv8 ile satranç taşlarını ve köşe noktalarını tespit eder,
- Perspektif dönüşümü ve ızgara hesaplamaları ile taşları karelere yerleştirir,
- Her hamleden sonra tahtanın FEN (Forsyth-Edwards Notation) karşılığını üretir,
- Tahtadaki durumu analiz ederek en iyi hamleyi önerir,
- Tüm süreci kullanıcıya görsel olarak sunan bir PyQt5 arayüzü sağlar.

## 🖼 Arayüz Özellikleri

- SVG destekli satranç tahtası gösterimi
- En iyi hamlenin okunabilir şekilde gösterimi
- Önceki hamleye geri dönme özelliği
- Taş konumlarının metin formatında gösterimi

## 🚀 Nasıl Çalıştırılır?

### 1. Gerekli Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
Not: requirements.txt dosyası içinde ultralytics, opencv-python, pyqt5, python-chess, requests ve numpy paketleri yer almalıdır.

2. Model Dosyalarını Yerleştirin.

models/best.pt → Satranç taşı tanıma modeli (YOLOv8)

models/best_corners.pt → Tahta köşe noktası tanıma modeli (YOLOv8)

3. Uygulamayı Başlatın

Kamera üzerinden gelen görüntüde c tuşuna basarak köşe tespiti yapılabilir. Ardından sistem otomatik olarak analiz yapar ve hamle önerir. q tuşuyla çıkış yapılır.

📁 Dosya Yapısı

```bash
.
├── main.py                 # Ana işlem dosyası (kamera ve işlem döngüsü)
├── chessGUI.py            # PyQt5 arayüz bileşeni
├── models/                # YOLOv8 model dosyaları (best.pt, best_corners.pt)
├── image/                 # (Opsiyonel) Test görüntüleri
```
🧠 Fonksiyonel Özellikler
📷 Görüntüden satranç tahtası algılama

♟ Taşların sınıflandırılması ve konumlandırılması

🔁 Oyun geçmişinin FEN formatıyla tutulması

🔍 En iyi hamlenin önerilmesi (Stockfish API ile)

❌ Hatalı FEN veya sahte hamlelerde doğrulama kontrolü

💡 Kullanım Senaryoları

Satranç oynarken en iyi hamleyi hızlıca görmek

Fiziksel tahta üzerinden dijital analiz yapmak

Yapay zekâ destekli eğitim ortamı sunmak

📷 Görseller (Ekran Görüntüleri)
(Buraya GUI'den alınacak bir ekran görüntüsü ekleyebilirsiniz.)



📌 Geliştirici: Aslı Cennet ERCAN
🎓 Üniversite: Marmara Üniversitesi 

📌 Geliştirici: Emine YİĞİT
🎓 Üniversite: Marmara Üniversitesi 

📌 Geliştirici: Yusuf DOĞAN
🎓 Üniversite: Marmara Üniversitesi 