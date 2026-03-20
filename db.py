import pyodbc
import numpy as np
from collections import defaultdict

def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=kanu\\SQLEXPRESS;"
        "DATABASE=FaceRecognitionDB;"
        "Trusted_Connection=yes;"
    )

# Create person
def insert_person():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Persons DEFAULT VALUES")
    conn.commit()

    cursor.execute("SELECT SCOPE_IDENTITY()")
    person_id = int(cursor.fetchone()[0])

    conn.close()
    return person_id

# Save embedding
def save_embedding(person_id, embedding):
    conn = get_connection()
    cursor = conn.cursor()

    embedding_bytes = embedding.astype(np.float32).tobytes()

    cursor.execute(
        "INSERT INTO FaceEmbeddings (PersonID, Embedding) VALUES (?, ?)",
        person_id,
        pyodbc.Binary(embedding_bytes)
    )

    conn.commit()
    conn.close()

# Load all embeddings (Grouped)
def get_all_embeddings():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT PersonID, Embedding FROM FaceEmbeddings")
    rows = cursor.fetchall()
    conn.close()

    grouped = defaultdict(list)

    for row in rows:
        person_id = row[0]
        embedding = np.frombuffer(row[1], dtype=np.float32)
        grouped[person_id].append(embedding)

    return grouped

# Log visit
def log_visit(person_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO VisitHistory (PersonID) VALUES (?)", person_id)
    conn.commit()
    conn.close()

# Visit count
def get_visit_count(person_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM VisitHistory WHERE PersonID = ?",
        person_id
    )

    count = cursor.fetchone()[0]
    conn.close()
    return count

# Save person image path (optional table)
def save_person_image(person_id, image_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO PersonImages (PersonID, ImagePath) VALUES (?, ?)",
        person_id,
        image_path
    )

    conn.commit()
    conn.close()

def get_person_images(person_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ImagePath FROM PersonImages WHERE PersonID = ?", person_id)
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]