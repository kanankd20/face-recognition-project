import cv2
import numpy as np
import os
import time
from insightface.app import FaceAnalysis
from db import insert_person, save_embedding, get_all_embeddings, log_visit, get_visit_count

# ---------------- SETTINGS ----------------
SIMILARITY_THRESHOLD = 0.45
HIGH_CONFIDENCE_THRESHOLD = 0.75  # if confidence is high, save new embedding
DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

print("Loading InsightFace Model (buffalo_s)...")
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0, det_size=(320, 320))
print("Model Loaded Successfully")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera Started")
print("Press 's' to SAVE face")
print("Press 'q' to Quit\n")

last_status = "No Face Detected"
frame_count = 0
last_frame = None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    last_frame = frame.copy()

    # Skip alternate frames to reduce CPU/GPU load
    if frame_count % 2 != 0:
        cv2.imshow("Real-Time Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    faces = app.get(frame)
    known_embeddings = get_all_embeddings()

    current_person_id = None
    current_score = 0

    for face in faces:
        embedding = face.embedding.astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        x1, y1, x2, y2 = face.bbox.astype(int)

        best_score = 0
        person_id = None

        # Compare with DB
        for db_id, emb_list in known_embeddings.items():
            for db_emb in emb_list:
                db_emb = db_emb / np.linalg.norm(db_emb)
                score = np.dot(embedding, db_emb)

                if score > best_score:
                    best_score = score
                    person_id = db_id

        if best_score > SIMILARITY_THRESHOLD:
            current_person_id = person_id
            current_score = best_score
            last_status = f"Known | ID: {person_id}"

            # If confidence is very high, save new embedding for better future recognition
            if best_score > HIGH_CONFIDENCE_THRESHOLD:
                save_embedding(person_id, embedding)

        else:
            current_person_id = None
            last_status = "Unknown"

        # Draw ellipse
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        axes_x = int((x2 - x1) / 2)
        axes_y = int((y2 - y1) / 2)

        cv2.ellipse(frame, (center_x, center_y),
                    (axes_x, axes_y),
                    0, 0, 360, (0, 0, 255), 2)

        # Show status + confidence
        cv2.putText(frame, f"{last_status} | Conf: {round(best_score*100,2)}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------------- SAVE ONLY WHEN S PRESSED ----------------
    if key == ord('s') and len(faces) > 0:
        face = faces[0]
        embedding = face.embedding.astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = frame[y1:y2, x1:x2]

        known_embeddings = get_all_embeddings()

        best_score = 0
        matched_id = None

        for db_id, emb_list in known_embeddings.items():
            for db_emb in emb_list:
                db_emb = db_emb / np.linalg.norm(db_emb)
                score = np.dot(embedding, db_emb)

                if score > best_score:
                    best_score = score
                    matched_id = db_id

        # If known person
        if best_score > SIMILARITY_THRESHOLD:
            person_id = matched_id
            print(f"\nRecognized Person")
            print(f"ID: {person_id}")
            print(f"Confidence: {round(best_score*100,2)}%")

            # Save new embedding to improve recognition
            save_embedding(person_id, embedding)

        # If new person
        else:
            person_id = insert_person()
            save_embedding(person_id, embedding)
            print(f"\nNew Person Registered")
            print(f"New ID: {person_id}")

        # Save image to folder
        person_folder = os.path.join(DATASET_PATH, f"person_{person_id}")
        os.makedirs(person_folder, exist_ok=True)

        timestamp = int(time.time())
        image_path = os.path.join(person_folder, f"{timestamp}.jpg")

        cv2.imwrite(image_path, face_img)

        log_visit(person_id)

        visits = get_visit_count(person_id)
        print(f"Total Visits: {visits}")
        print("Image Saved Successfully\n")

    if key == ord('q'):
        break

    time.sleep(0.01)  # Reduce CPU load

cap.release()
cv2.destroyAllWindows()

# Show last frame after quitting
if last_frame is not None:
    cv2.imshow("Final Frame", last_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nFinal Status:", last_status)
print("Program Closed Successfully")