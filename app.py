import cv2
import os

# Neden yazıldı?: Okuyacağımız video dosyasının bilgisayardaki yolunu belirtiyoruz.
VIDEO_PATH = os.path.join("videos", "test_video.mp4")

def main():
    # Neden yazıldı?: Belirttiğimiz video dosyasını mevcut mu diye kontrol ediyoruz.
    if not os.path.exists(VIDEO_PATH):
        print(f"Hata: '{VIDEO_PATH}' konumunda video dosyası bulunamadı!")
        print("Lütfen 'videos' klasörünün içine 'test_video.mp4' dosyasını ekleyin.")
        return

    # Neden yazıldı?: Video dosyasından kare kare görüntü akışını başlatmak için.
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Neden yazıldı?: Videonun saniyede kaç kare (FPS) oynatıldığını alıyoruz. 
    # Bu sayede videoyu kendi hızında oynatabileceğiz.
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    print(f"Video başarıyla yüklendi ({fps} FPS). İzlemek için bekleniyor...")

    # Neden yazıldı?: Video bitene kadar tüm kareleri döngüyle tek tek okuyoruz.
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Neden yazıldı?: Eğer ret False dönerse video bitmiş demektir, döngüden çıkıyoruz.
        if not ret:
            print("Video akışı başarıyla tamamlandı.")
            break

        # --- BURASI GELECEKTEKİ ADIMLARIMIZIN YERİ ---
        # TODO: Aşama 2 - YOLO ile insan tespiti (Detection) burada yapılacak.
        # TODO: Aşama 3 - Giriş/Çıkış takibi (Tracking) burada yapılacak.
        # ---------------------------------------------

        # Neden yazıldı?: Üzerinde analiz yapacağımız o anki video karesini ekranda gösteriyoruz.
        cv2.imshow("Magaza / Alan Yogunluk Takip Sistemi", frame)

        # Neden yazıldı?: Videoyu gerçek hızında oynatmak ve 'q' tuşuna basılırsa kapatmak için.
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Kullanıcı tarafından video kapatıldı.")
            break

    # Neden yazıldı?: Video bittiğinde belleği temizlemek ve pencereyi kapatmak için.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()