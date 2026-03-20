# 🧠 AI Face Recognition System with SQL Dashboard

A real-time **Face Recognition System** built using **InsightFace, OpenCV, and SQL Server**, featuring:

* 🎯 Real-time face detection & recognition
* 🧑 Persistent identity tracking (same ID across angles)
* 💾 Automatic image storage
* 📊 Visit tracking & analytics
* 🌐 Live web dashboard (Flask)
* ⚡ Optimized for low CPU usage

---

# 🚀 Features

## 🎥 Face Recognition

* Detects and recognizes faces in real-time
* Uses **InsightFace (buffalo_s)** lightweight model
* Assigns **unique ID per person**
* Maintains identity across different angles

## 💾 Data Storage

* Saves face embeddings in SQL Server
* Stores multiple images per person
* Folder structure:

```
dataset/
   ├── person_1/
   ├── person_2/
```

## 📊 Analytics

* Total persons detected
* Total visits
* Visits per person
* Unknown detection count
* Confidence score display

## 🌐 Dashboard

* Built using Flask
* Displays:

  * Total persons
  * Total visits
  * Visit frequency per person

---

# 🛠️ Tech Stack

* Python 3.11
* OpenCV
* InsightFace
* NumPy
* Flask
* SQL Server (SSMS)
* pyodbc

---

# 🗄️ Database Setup (SQL Server)

## Step 1: Create Database

```sql
CREATE DATABASE FaceRecognitionDB;
```

---

## Step 2: Use Database

```sql
USE FaceRecognitionDB;
```

---

## Step 3: Create Tables

```sql
CREATE TABLE persons (
    person_id INT IDENTITY(1,1) PRIMARY KEY
);

CREATE TABLE embeddings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    person_id INT,
    embedding VARBINARY(MAX)
);

CREATE TABLE visits (
    id INT IDENTITY(1,1) PRIMARY KEY,
    person_id INT,
    visit_time DATETIME DEFAULT GETDATE()
);
```

---

# ⚙️ Installation

## 1️⃣ Clone Project

```bash
git clone <your-repo-link>
cd project1
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv311
.venv311\Scripts\activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔌 Database Connection

Update in `db.py` and `dashboard.py`:

```python
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=FaceRecognitionDB;"
    "Trusted_Connection=yes;"
)
```

---

# ▶️ How to Run

## 🎥 Start Face Recognition

```bash
python recognition.py
```

### Controls:

* **S** → Save face
* **Q** → Quit

---

## 🌐 Start Dashboard

```bash
python dashboard.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 📊 SQL Analytics Queries

## Total Persons

```sql
SELECT COUNT(*) FROM persons;
```

## Total Visits

```sql
SELECT COUNT(*) FROM visits;
```

## Visits Per Person

```sql
SELECT person_id, COUNT(*) as visits
FROM visits
GROUP BY person_id
ORDER BY visits DESC;
```

---

# ⚡ Performance Optimization

* Frame skipping (`PROCESS_EVERY = 5`)
* Low resolution (640x480)
* Single face detection
* Lightweight model (buffalo_s)
* CPU cooling delay

---

# 🧪 Expected Output

* Smooth camera feed
* Stable recognition
* No overheating
* Real-time analytics
* Images saved correctly

---

# 🚨 Troubleshooting

## ❌ Camera not opening

* Check webcam permissions
* Try changing:

```python
cv2.VideoCapture(0)
```

---

## ❌ SQL Error (Invalid table)

Run:

```sql
USE FaceRecognitionDB;
```

---

## ❌ Images not saving

* Ensure `dataset/` folder exists
* Press **S key** while face is visible

---

# 🔥 Future Enhancements

* 📊 Graph analytics (charts)
* 🎯 Face tracking (no flicker)
* 🌐 Live camera in dashboard
* 🔐 Admin login panel
* ☁️ Cloud deployment


