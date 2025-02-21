import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
import sqlite_vec
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import hashlib
import colorsys

database_path = "faces.db" # Default save to faces.db
dataset_dir = "faces"
augment_data = True  # Set to True to enable data augmentation

mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def name_to_hex(name: str) -> str:
    """
    Convert a given name (string) into a vibrant hex color code.

    The function uses the MD5 hash of the name to generate a hue value (0-360 degrees).
    It then uses fixed saturation and lightness values (0.8 and 0.4 respectively) to ensure
    that the generated color is vibrant (covering red, blue, green spectrum) and dark enough
    for white text, while avoiding grey or monochrome colors.

    Parameters:
    - name (str): The input string to convert.

    Returns:
    - str: A hex color string, e.g., "#d23f78".
    """
    # Generate an MD5 hash of the input name
    hash_object = hashlib.md5(name.encode('utf-8'))
    # Use the full hash value to derive a hue between 0 and 360 degrees
    hue = (int(hash_object.hexdigest(), 16) % 360) / 360.0

    # Set fixed saturation and lightness for vibrant color:
    # - High saturation (0.8) ensures the color is not grey.
    # - Lightness (0.4) gives a darker shade that contrasts well with white text.
    saturation = 0.8
    lightness = 0.4

    # Convert HLS (note: colorsys uses HLS ordering: hue, lightness, saturation) to RGB.
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Convert RGB values from 0-1 range to 0-255 and format as hex.
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def connect_database(database_path):
    db = sqlite3.connect(database_path)
    db.enable_load_extension(True)  # Setup SQLite vector extension
    sqlite_vec.load(db)  # Setup SQLite vector extension
    db.enable_load_extension(False)  # Setup SQLite vector extension

    return db

def align_face(img, box):
    """Aligns a detected face using landmarks."""
    x1, y1, x2, y2 = box.astype(int)
    face_img = img[y1:y2, x1:x2]
    return face_img


def detect_and_embed(img):
    """Detects faces, aligns them, and returns embeddings."""
    boxes, _ = mtcnn.detect(img)  # Detect face
    if boxes is None:
        return [], []

    embeddings = []
    aligned_faces = []

    for box in boxes:
        aligned_face = align_face(img, box)
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
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    ])

    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        box_color = name_to_hex(person_name)
        if os.path.isdir(person_dir):
            for image_file in tqdm(os.listdir(person_dir), desc=f"Processing {person_name}"):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, image_file)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, embeddings = detect_and_embed(img)

                    if embeddings:
                        db = connect_database(database_path)
                        cursor = db.cursor()
                        for embedding in embeddings:
                            cursor.execute("INSERT INTO face_embeddings (name, boxcolor, embedding) VALUES (?,?,?)",
                                           (person_name, box_color, embedding.astype(np.float32)))
                        db.commit()
                        db.close()

                    if augment_data:
                        pil_img = Image.fromarray(img)
                        for _ in range(5):  # Create 5 augmentations per image
                            augmented_img = augmentations(pil_img)
                            augmented_img = np.array(augmented_img)  # Convert back to NumPy
                            _, embeddings = detect_and_embed(augmented_img)
                            if embeddings:
                                db = connect_database(database_path)
                                cursor = db.cursor()
                                for embedding in embeddings:
                                    cursor.execute("INSERT INTO face_embeddings (name, boxcolor, embedding) VALUES (?,?,?)",
                                                   (person_name, box_color, embedding.astype(np.float32)))
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
                boxcolor TEXT NOT NULL,
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