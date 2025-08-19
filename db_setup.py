# db_setup.py
from app import app
from models import db

with app.app_context():
    print("Creating database tables...")
    db.create_all()
    print("Tables created successfully!")