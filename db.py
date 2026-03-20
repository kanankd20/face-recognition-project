import pyodbc
import numpy as np

# ---------- CONNECTION ----------
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=FaceRecognitionDB;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# ---------- INSERT PERSON ----------
def insert_person():
    cursor.execute("INSERT INTO persons DEFAULT VALUES")
    conn.commit()
    cursor.execute("SELECT @@IDENTITY")
    return int(cursor.fetchone()[0])

# ---------- SAVE EMBEDDING ----------
def save_embedding(person_id, embedding):
    emb_bytes = embedding.astype(np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
        (person_id, emb_bytes)
    )
    conn.commit()

# ---------- GET ALL EMBEDDINGS ----------
def get_all_embeddings():
    cursor.execute("SELECT person_id, embedding FROM embeddings")
    rows = cursor.fetchall()

    data = {}

    for row in rows:
        person_id = row[0]
        emb = np.frombuffer(row[1], dtype=np.float32)

        if person_id not in data:
            data[person_id] = []

        data[person_id].append(emb)

    return data

# ---------- VISITS ----------
def log_visit(person_id):
    cursor.execute(
        "INSERT INTO visits (person_id) VALUES (?)",
        (person_id,)
    )
    conn.commit()

def get_visit_count(person_id):
    cursor.execute(
        "SELECT COUNT(*) FROM visits WHERE person_id=?",
        (person_id,)
    )
    return cursor.fetchone()[0]
# ---------- ANALYTICS ----------
def get_total_persons():
    cursor.execute("SELECT COUNT(*) FROM persons")
    return cursor.fetchone()[0]

def get_total_visits():
    cursor.execute("SELECT COUNT(*) FROM visits")
    return cursor.fetchone()[0]

def get_top_person():
    cursor.execute("""
        SELECT TOP 1 person_id, COUNT(*) as visits
        FROM visits
        GROUP BY person_id
        ORDER BY visits DESC
    """)
    row = cursor.fetchone()
    return row if row else (None, 0)
