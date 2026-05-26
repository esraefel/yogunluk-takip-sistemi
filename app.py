import cv2
import os
from ultralytics import YOLO  # Neden yazıldı?: YOLO modelini projemize dahil etmek için

VIDEO_PATH = os.path.join("videos", "test_video.mp4")

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Hata: '{VIDEO_PATH}' konumunda video dosyası bulunamadı!")
        returnimport cv2
import os
from ultralytics import YOLO

VIDEO_PATH = os.path.join("videos", "test_video.mp4")

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Hata: '{VIDEO_PATH}' konumunda video dosyası bulunamadı!")
        return

    model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    print("YOLO ve Tracking Motoru aktif edildi...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True)

        for r in results:
            boxes = r.boxes
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    cls = int(box.cls[0])
                    
                    # Sadece insanları (0) takip et
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Neden yazıldı?: Dikdörtgenin rengini her insan için ID'sine göre dinamik yapıyoruz
                        # Böylece herkesin kutu rengi farklı görünecek (Takip görsel kalitesi için)
                        color = (int((track_id * 40) % 255), 255, int((track_id * 80) % 255))

                        # İnsanın etrafına kutu çiz
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Neden yazıldı?: Kutunun üzerine "ID: 1", "ID: 2" şeklinde benzersiz kimliği yazdırıyoruz.
                        label = f"ID: {track_id}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Magaza / Alan Yogunluk Takip Sistemi - Tracking", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    print("YOLO Modeli yüklendi. İnsan tespiti başlıyor...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Neden yazıldı?: Alınan anlık kareyi (frame) YOLO yapay zeka modeline gönderiyoruz.
        # stream=True yaparak videonun belleği şişirmesini engelliyoruz.
        results = model(frame, stream=True)

        # Neden yazıldı?: Modelin bulduğu sonuçları (kutuları, sınıfları) tek tek dönüyoruz.
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Neden yazıldı?: Bulunan nesnenin sınıf numarasını alıyoruz (Örn: COCO veri setinde İnsan = 0)
                cls = int(box.cls[0])
                
                # Eğer tespit edilen nesne bir İNSAN (0) ise ekrana çizdireceğiz
                if cls == 0:
                    # Neden yazıldı?: İnsanın etrafındaki kutunun koordinatlarını alıyoruz (X1, Y1, X2, Y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Neden yazıldı?: Güven skorunu alıyoruz (Yapay zeka % kaç ihtimalle bunun insan olduğunu düşünüyor?)
                    conf = float(box.conf[0])

                    # Neden yazıldı?: İnsanın etrafına yeşil bir dikdörtgen çiziyoruz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Neden yazıldı?: Kutunun üzerine güven skorunu yazdırıyoruz (Örn: Person 0.85)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sonucu ekranda göster
        cv2.imshow("Magaza / Alan Yogunluk Takip Sistemi - YOLO Detection", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()