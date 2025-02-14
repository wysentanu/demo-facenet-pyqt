import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
import sqlite_vec
from tqdm import tqdm

database_path = "faces.db" # Default save to faces.db
dataset_dir = "faces"

mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def connect_database(database_path):
    db = sqlite3.connect(database_path)
    db.enable_load_extension(True)  # Setup SQLite vector extension
    sqlite_vec.load(db)  # Setup SQLite vector extension
    db.enable_load_extension(False)  # Setup SQLite vector extension

    return db

def _align_face(img, box):
    """Aligns a detected face using landmarks."""
    x1, y1, x2, y2 = box.astype(int)
    face_img = img[y1:y2, x1:x2]
    return face_img


def _detect_and_embed(img):
    """Detects faces, aligns them, and returns embeddings."""
    boxes, _ = mtcnn.detect(img)  # Detect face
    if boxes is None:
        return [], []

    embeddings = []
    aligned_faces = []

    for box in boxes:
        aligned_face = _align_face(img, box)
        aligned_faces.append(aligned_face)

    if aligned_faces:
        faces = mtcnn.extract(img, boxes, save_path=None)  # Extracting faces from original image with boxes
        if faces is not None:
            for face in faces:
                face = face.unsqueeze(0)
                with torch.no_grad():
                    embedding = facenet(face).detach().cpu().numpy()  # Extract embedding
                    embeddings.append(embedding)
        else:
            print("Warning: Could not extract faces. Skipping.")
    return boxes, embeddings


def process_directory(base_dir):
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):
            for image_file in tqdm(os.listdir(person_dir), desc=f"Processing {person_name}"):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, image_file)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, embeddings = _detect_and_embed(img)

                    if embeddings:
                        db = connect_database(database_path)
                        cursor = db.cursor()
                        for embedding in embeddings:
                            cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)",
                                           (person_name, embedding.astype(np.float32)))
                        db.commit()
                        db.close()


def train():
    # Process all images in the directory structure and store embeddings
    if os.path.exists(dataset_dir):
        print("Processing face images...")

        # Connect to the database
        db = sqlite3.connect(database_path)
        db.enable_load_extension(True)  # Setup SQLite vector extension
        sqlite_vec.load(db)  # Setup SQLite vector extension
        db.enable_load_extension(False)  # Setup SQLite vector extension

        # Create database if not exists
        cursor = db.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS face_embeddings using vec0(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding FLOAT[512]
            )
        """)
        db.commit()
        db.close()

        process_directory(dataset_dir)

        print("\nAll face embeddings have been stored in the database.")
    else:
        print(f"Please create a '{dataset_dir}' directory with person folders containing face images.")
        print("Example structure:")
        print("faces/")
        print("  ├── person1/")
        print("  │   ├── image1.jpg")
        print("  │   ├── image2.jpg")
        print("  │   └── ...")
        print("  ├── person2/")
        print("  │   ├── image1.jpg")
        print("  │   ├── image2.jpg")
        print("  │   └── ...")
        print("  └── ...")
        return

if __name__ == "__main__":
    train()