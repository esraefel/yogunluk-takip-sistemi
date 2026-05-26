import cv2
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3  # Neden yazıldı?: SQLite veritabanı işlemleri için
import json     # Neden yazıldı?: JSON raporu üretmek için
import logging  # Neden yazıldı?: Sistemin ayak izlerini loglamak için
from datetime import datetime
from ultralytics import YOLO

# --- AŞAMA 8: LOGGING & ERROR HANDLING CONFIGURATION ---
# Neden yazıldı?: Sistemdeki tüm olayları tarih ve saatle 'sistem.log' dosyasına kaydeder.
logging.basicConfig(
    filename="sistem.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

st.set_page_config(page_title="Alan Yoğunluk Dashboard", layout="wide")
st.title("📊 Gerçek Zamanlı Alan Kullanım ve Yoğunluk Takibi")
st.sidebar.header("Sistem Ayarları")

VIDEO_PATH = os.path.join("videos", "test_video.mp4")

if "in_count" not in st.session_state:
    st.session_state.in_count = 0
if "out_count" not in st.session_state:
    st.session_state.out_count = 0

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📹 Canlı Analiz Akışı")
    FRAME_WINDOW = st.image([]) 

with col2:
    st.subheader("📈 Anlık İstatistikler")
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()

# --- AŞAMA 6: DATABASE (VERİTABANI) KURULUMU ---
# Neden yazıldı?: Veritabanı dosyasını oluşturur ve yoksa 'raporlar' tablosunu hazırlar.
def veritabanini_hazirla():
    try:
        conn = sqlite3.connect("yogunluk.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raporlar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tarih_saat TEXT,
                giris_sayisi INTEGER,
                cikis_sayisi INTEGER
            )
        """)
        conn.commit()
        conn.close()
        logging.info("Veritabanı bağlantısı başarıyla kuruldu ve tablolar doğrulandı.")
    except Exception as e:
        logging.error(f"Veritabanı hazırlanırken HATA oluştu: {str(e)}")

veritabanini_hazirla()

start_system = st.sidebar.button("Sistemi Başlat")

if start_system and os.path.exists(VIDEO_PATH):
    logging.info("Kullanıcı sistemi başlattı. Görüntü işleme döngüsüne giriliyor.")
    
    try:
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(VIDEO_PATH)
        POLYGON_POINTS = np.array([[380, 480], [640, 480], [640, 280], [450, 280]], np.int32)
        
        track_history = {}  
        already_counted = set() 
        frame_count = 0 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("Video dosyası bitti veya akış durdu.")
                break

            frame_count += 1 
            if frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            cv2.polylines(frame, [POLYGON_POINTS], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, "GIRIS / CIKIS", (390, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False)

            for r in results:
                if r.boxes.id is not None:
                    track_ids = r.boxes.id.int().tolist()
                    boxes = r.boxes

                    for box, track_id in zip(boxes, track_ids):
                        if int(box.cls[0]) == 0: 
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            is_inside = cv2.pointPolygonTest(POLYGON_POINTS, (cx, cy), False)

                            if track_id not in track_history:
                                track_history[track_id] = "inside" if is_inside >= 0 else "outside"
                                if is_inside >= 0 and track_id not in already_counted:
                                    st.session_state.in_count += 1
                                    already_counted.add(track_id)
                                    logging.info(f"Yeni Giriş Algılandı! ID: {track_id} - Toplam Giriş: {st.session_state.in_count}")

                            prev_state = track_history[track_id]

                            if prev_state == "outside" and is_inside >= 0 and track_id not in already_counted:
                                st.session_state.out_count += 1
                                already_counted.add(track_id)
                                logging.info(f"Yeni Çıkış Algılandı! ID: {track_id} - Toplam Çıkış: {st.session_state.out_count}")

                            track_history[track_id] = "inside" if is_inside >= 0 else "outside"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            # Modern Arayüz Kartları
            with stats_placeholder.container():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 8px;">
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 14px; font-weight: 600; color: #666; text-transform: uppercase;">Giriş</div>
                        <div style="font-size: 36px; font-weight: bold; color: #1f77b4; margin-top: 5px;">{st.session_state.in_count}</div>
                    </div>
                    <div style="text-align: center; flex: 1; border-left: 1px solid #e6e9ef;">
                        <div style="font-size: 14px; font-weight: 600; color: #666; text-transform: uppercase;">Çıkış</div>
                        <div style="font-size: 36px; font-weight: bold; color: #ff7f0e; margin-top: 5px;">{st.session_state.out_count}</div>
                    </div>
                </div>
                <div style="margin-bottom: 20px;"></div>
                """, unsafe_allow_html=True)

            # Dinamik Kırpışmasız Grafik Güncellemesi
            df_data = pd.DataFrame({
                "Aktivite": ["Giriş", "Çıkış"],
                "Kişi Sayısı": [st.session_state.in_count, st.session_state.out_count]
            })
            fig = px.bar(df_data, x="Aktivite", y="Kişi Sayısı", color="Aktivite", range_y=[0, max(15, st.session_state.in_count + 5)])
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"density_chart_{frame_count}")

        cap.release()

        # --- VİDEO BİTTİĞİNDE ÇALIŞACAK KURUMSAL ADIMLAR ---
        su_an = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # AŞAMA 6: VERİLERİ VERİTABANINA KAYDETME
        conn = sqlite3.connect("yogunluk.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO raporlar (tarih_saat, giris_sayisi, cikis_sayisi) VALUES (?, ?, ?)", 
                       (su_an, st.session_state.in_count, st.session_state.out_count))
        conn.commit()
        conn.close()
        logging.info("Günlük sayım verileri başarıyla SQLite veritabanına kaydedildi.")

        # AŞAMA 7: JSON VE CSV RAPORU ÜRETME
        rapor_veri = {
            "tarih_saat": su_an,
            "toplam_giris": st.session_state.in_count,
            "toplam_cikis": st.session_state.out_count
        }
        
        # JSON Çıktısı
        with open("gunluk_rapor.json", "w", encoding="utf-8") as f:
            json.dump(rapor_veri, f, ensure_ascii=False, indent=4)
            
        # CSV Çıktısı
        df_csv = pd.DataFrame([rapor_veri])
        df_csv.to_csv("gunluk_rapor.csv", index=False, encoding="utf-8")
        
        logging.info("gunluk_rapor.json ve gunluk_rapor.csv başarıyla oluşturuldu.")
        st.success("🎉 Video analizi tamamlandı! Veritabanı güncellendi, CSV/JSON raporları üretildi.")

    except Exception as ana_hata:
        logging.error(f"Sistem çalışırken kritik hata oluştu: {str(ana_hata)}")
        st.error("Sistem çalışırken bir hata oluştu. Lütfen log dosyasını kontrol edin.")