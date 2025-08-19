# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class OcrResult(db.Model):
    __tablename__ = 'ocr_results'

    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    extracted_data = db.Column(db.JSON, nullable=False)
    face_image_path = db.Column(db.String(255), nullable=True)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<OcrResult {self.id}: {self.original_filename}>'