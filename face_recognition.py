import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import sqlite3
import sqlite_vec

def cosine_similarity(emb1, emb2):
    """Computes the cosine similarity between two embeddings."""
    return np.dot(emb1.flatten(), emb2.flatten()) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

class FaceRecognition:
    def __init__(self, mtcnn, database_path=None):
        self.mtcnn = mtcnn
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.database_path = database_path
    
    def _connect_database(self):
        db = sqlite3.connect(self.database_path)
        db.enable_load_extension(True) # Setup SQLite vector extension
        sqlite_vec.load(db) # Setup SQLite vector extension
        db.enable_load_extension(False) # Setup SQLite vector extension

        return db

    def _align_face(self, img, box):
        """Aligns a detected face using landmarks."""
        x1, y1, x2, y2 = box.astype(int)
        face_img = img[y1:y2, x1:x2]
        return face_img

    def _get_embed(self, img, boxes):
        """Aligns faces, and returns embeddings."""
        embeddings = []
        aligned_faces = []

        for box in boxes:
            aligned_face = self._align_face(img, box)
            aligned_faces.append(aligned_face)

        if aligned_faces:
            faces = self.mtcnn.extract(img, boxes, save_path=None) # Extracting faces from original image with boxes
            if faces is not None:
                for face in faces:
                    face = face.unsqueeze(0)
                    with torch.no_grad():
                        embedding = self.facenet(face).detach().cpu().numpy() # Extract embedding
                        embeddings.append(embedding)
            else:
                print("Warning: Could not extract faces. Skipping.")
        return boxes, embeddings

    def identify_face(self, boxes, embeddings, threshold=0.6):  # Add boxes as parameter
        if self.database_path is None:
            return ["Unknown"] * len(boxes) if boxes is not None else []  # Return only unknown names
        
        db = self._connect_database()
        cursor = db.cursor()

        identified_names = []
        if embeddings: # Check if embeddings exist
            for embedding in embeddings:
                row = db.execute(
                    """
                    SELECT
                        name,
                        distance,
                        embedding
                    FROM face_embeddings
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT 1
                    """,
                    [embedding.astype(np.float32)],
                ).fetchone()

                if row:
                    row_name, row_distance, row_embedding = row
                    np_row_embedding = np.frombuffer(row_embedding, dtype=np.float32)
                    np_row_embedding = np_row_embedding.reshape(1, -1)

                    similarity = cosine_similarity(embedding, np_row_embedding)
                    
                    if similarity > threshold:
                        identified_names.append(f"({similarity*100:.0f}%) {row_name}")
                    else:
                        identified_names.append("Unknown")
                else:
                    identified_names.append("Unknown")
        else:
            identified_names = ["Unknown"] * len(boxes) if boxes is not None else [] # Handle case where no embeddings

        db.close()
        return identified_names  # Return only identified names