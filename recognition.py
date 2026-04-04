import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from insightface.app import FaceAnalysis
from db import (
    insert_person, save_embedding, get_all_embeddings,
    log_visit, get_visit_count,
    get_total_persons, get_total_visits, get_top_person
)

# ---------- SETTINGS ----------
SIMILARITY_THRESHOLD = 0.45
DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

PROCESS_EVERY = 5
DISPLAY_TIME = 2  # seconds

frame_count = 0
last_seen = {}

print("Loading Model...")
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0, det_size=(192, 192))
print("Model Loaded")

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera error")
    exit()

print("✅ Camera Started")
print("Press S = Save | Q = Quit")

# ---------- LOAD DATABASE ----------
known_embeddings = get_all_embeddings()

prev_time = 0
unknown_count = 0

# ---------- SAFE DISPLAY ----------
def safe_imshow(winname, frame):
    try:
        cv2.imshow(winname, frame)
    except cv2.error:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.pause(0.001)

def safe_waitkey(delay=1):
    try:
        return cv2.waitKey(delay) & 0xFF
    except cv2.error:
        return -1

# ---------- EMOTION MAPPING ----------
def confidence_to_emotion(conf):
    if conf > 80:
        return "Happy"
    elif conf > 50:
        return "Neutral"
    else:
        return "Sad"

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # ---------- FPS ----------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # ---------- PROCESS FACES ----------
    if frame_count % PROCESS_EVERY == 0:
        faces = app.get(frame)   # detect ALL faces

        for face in faces:
            embedding = face.embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            x1, y1, x2, y2 = face.bbox.astype(int)

            best_score = 0
            person_id = None

            for db_id, emb_list in known_embeddings.items():
                for db_emb in emb_list:
                    db_emb = db_emb / np.linalg.norm(db_emb)
                    score = np.dot(embedding, db_emb)
                    if score > best_score:
                        best_score = score
                        person_id = db_id

            if best_score > SIMILARITY_THRESHOLD:
                confidence = round(best_score * 100, 2)
                visits = get_visit_count(person_id)
                last_seen[person_id] = {
                    "confidence": confidence,
                    "visits": visits,
                    "time": current_time,
                    "bbox": (x1, y1, x2, y2),
                    "embedding": embedding
                }
            else:
                unknown_count += 1

    # ---------- DRAW RESULTS ----------
    for pid, data in list(last_seen.items()):
        if current_time - data["time"] < DISPLAY_TIME:
            x1, y1, x2, y2 = data["bbox"]

            # Oval face highlight
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = (x2 - x1) // 2
            ay = (y2 - y1) // 2
            cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (0, 0, 255), 2)

            label = f"ID:{pid}"
            emotion = confidence_to_emotion(data["confidence"])
            visits_text = f"Visits: {data['visits']}"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, visits_text, (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            del last_seen[pid]

    # ---------- ANALYTICS PANEL ----------
    total_persons = get_total_persons()
    total_visits = get_total_visits()
    top_person, top_visits = get_top_person()

    # Status bar with headline + timestamp
    status_color = (0, 255, 0) if len(last_seen) > 0 else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (640, 30), status_color, -1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"VisionGuard AI | {timestamp}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    # Footer analytics
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Persons: {total_persons}", (150, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Visits: {total_visits}", (300, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Unknown: {unknown_count}", (450, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if top_person:
        cv2.putText(frame, f"Top ID:{top_person} ({top_visits} visits)",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 200), 2)

    # ---------- SHOW ----------
    safe_imshow("VisionGuard AI", frame)

    key = safe_waitkey(1)

    # ---------- SAVE ----------
    if key == ord('s') and len(last_seen) > 0:
        for pid, data in last_seen.items():   # save ALL detected faces
            x1, y1, x2, y2 = data["bbox"]
            face_img = frame[y1:y2, x1:x2]
            embedding = data["embedding"]

            best_score = data["confidence"] / 100
            if best_score > SIMILARITY_THRESHOLD:
                person_id = pid
                save_embedding(person_id, embedding)
            else:
                person_id = insert_person()
                save_embedding(person_id, embedding)

            known_embeddings = get_all_embeddings()

            folder = f"{DATASET_PATH}/person_{person_id}"
            os.makedirs(folder, exist_ok=True)
            count = len(os.listdir(folder))
            path = f"{folder}/{count+1}.jpg"
            cv2.imwrite(path, face_img)

            log_visit(person_id)
            print(f"💾 Saved Image → ID {person_id} → {path}")

    if key == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
