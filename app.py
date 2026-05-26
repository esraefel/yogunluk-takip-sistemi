import cv2
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from ultralytics import YOLO

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

start_system = st.sidebar.button("Sistemi Başlat")

if start_system and os.path.exists(VIDEO_PATH):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Sağ taraftaki kapı koordinatları
    POLYGON_POINTS = np.array([[380, 480], [640, 480], [640, 280], [450, 280]], np.int32)
    
    track_history = {}  
    already_counted = set() 
    frame_count = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1 
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        # İsteğin üzerine: Yazı sadece "GIRIS / CIKIS" olarak güncellendi
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

                        # --- DOĞRULANMIŞ GİRİŞ / ÇIKIŞ MANTIĞI ---
                        if track_id not in track_history:
                            # Kişinin ilk belirdiği konumu hafızaya al
                            track_history[track_id] = "inside" if is_inside >= 0 else "outside"
                            
                            # Kural 1: Eğer kişi İLK DEFA kapı alanının İÇİNDE belirmişse, dışarıdan gelmiştir -> GİRİŞ
                            if is_inside >= 0 and track_id not in already_counted:
                                st.session_state.in_count += 1
                                already_counted.add(track_id)

                        prev_state = track_history[track_id]

                        # Kural 2: Eğer kişi daha önce dışarıdaydı (mağazadaydı) ve ŞİMDİ kapı alanının içine girdiyse -> ÇIKIŞ
                        if prev_state == "outside" and is_inside >= 0 and track_id not in already_counted:
                            st.session_state.out_count += 1
                            already_counted.add(track_id)

                        # Durumu güncelle
                        track_history[track_id] = "inside" if is_inside >= 0 else "outside"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        # --- YENİ MODERN VE SADE TASARIM ---
        # Emojiler kaldırıldı, renkler netleştirildi, "Bölge" kelimesi silindi.
        with stats_placeholder.container():
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
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

        df_data = pd.DataFrame({
            "Aktivite": ["Giriş", "Çıkış"],
            "Kişi Sayısı": [st.session_state.in_count, st.session_state.out_count]
        })
        fig = px.bar(df_data, x="Aktivite", y="Kişi Sayısı", color="Aktivite", range_y=[0, max(15, st.session_state.in_count + 5)])
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"density_chart_{frame_count}")

    cap.release()