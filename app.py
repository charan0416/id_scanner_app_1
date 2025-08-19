# app.py
import os
import base64
import uuid
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from models import db, OcrResult
from utils import (
    combine_images_to_pdf,
    detect_and_extract_face,
    call_gemini_ocr,
    convert_pdf_first_page_to_image,
    preprocess_image_for_ocr
)

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# --- App Configuration ---
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_FACES_FOLDER'] = 'extracted_faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# --- Database Configuration ---
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FACES_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Core Processing Logic ---
def run_ocr_pipeline(file_to_process, original_filename, face_source_image_path):
    """
    Runs the full OCR, face detection, and database saving pipeline.
    This function is called by all endpoints after files are prepared.
    """
    # Step 1: Call Gemini for OCR extraction
    extracted_data = call_gemini_ocr(file_to_process)
    if "error" in extracted_data:
        return extracted_data, 500

    # Step 2: Detect and extract face from the source image
    face_path = None
    if face_source_image_path:
        face_path = detect_and_extract_face(face_source_image_path, app.config['EXTRACTED_FACES_FOLDER'])

    # Step 3: Save results to the database
    new_result = OcrResult(
        original_filename=original_filename,
        extracted_data=extracted_data,
        face_image_path=face_path
    )
    db.session.add(new_result)
    db.session.commit()

    # Step 4: Return success response
    return {"success": True, "id": new_result.id, "data": extracted_data, "face_image": face_path}, 200


# --- File Handling Logic for Form/File Uploads ---
def process_uploaded_files(files):
    """Handles file uploads from forms, prepares them, and calls the OCR pipeline."""
    single_file = files.get('single_file')
    side1_file = files.get('side1_file')
    side2_file = files.get('side2_file')

    file_to_process = None
    original_filename = ""
    face_source_image_path = None

    if single_file and single_file.filename != '' and allowed_file(single_file.filename):
        filename = secure_filename(single_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        single_file.save(filepath)

        original_filename = filename
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            print("Single image detected. Pre-processing for OCR...")
            face_source_image_path = filepath
            file_to_process = preprocess_image_for_ocr(filepath, app.config['UPLOAD_FOLDER'])
        elif file_extension == 'pdf':
            print("PDF detected, skipping image pre-processing for OCR step.")
            file_to_process = filepath
            face_source_image_path = convert_pdf_first_page_to_image(
                filepath, app.config['UPLOAD_FOLDER']
            )
            if not face_source_image_path:
                print("Could not extract image from PDF for face detection.")

    elif side1_file and side2_file and allowed_file(side1_file.filename) and allowed_file(side2_file.filename):
        s1_filename = secure_filename(side1_file.filename)
        s2_filename = secure_filename(side2_file.filename)
        s1_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s1_filename)
        s2_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s2_filename)
        side1_file.save(s1_filepath)
        side2_file.save(s2_filepath)

        face_source_image_path = s1_filepath  # Use side 1 for face detection

        pdf_path, pdf_filename = combine_images_to_pdf(s1_filepath, s2_filepath, app.config['UPLOAD_FOLDER'])
        if pdf_path:
            file_to_process = pdf_path
            original_filename = pdf_filename
        else:
            return {"error": "Failed to combine images into PDF"}, 400
    else:
        return {"error": "Invalid file submission. Please provide a single file or two side images."}, 400

    if not file_to_process:
        return {"error": "File to process could not be determined."}, 400

    return run_ocr_pipeline(file_to_process, original_filename, face_source_image_path)


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    results = OcrResult.query.order_by(OcrResult.processed_at.desc()).all()
    return render_template('history.html', results=results)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/extracted_faces/<path:filename>')
def serve_face(filename):
    return send_from_directory(app.config['EXTRACTED_FACES_FOLDER'], filename)


# --- Endpoint for Web Form Uploads ---
@app.route('/upload', methods=['POST'])
def upload_from_form():
    if not request.files:
        return "No file part", 400

    files = {
        'single_file': request.files.get('single_file'),
        'side1_file': request.files.get('side1_file'),
        'side2_file': request.files.get('side2_file'),
    }

    result, status_code = process_uploaded_files(files)

    if status_code == 200:
        return redirect(url_for('history'))
    else:
        # Simple error display for the web form
        error_message = result.get('error', 'An unknown error occurred.')
        return f"Error: {error_message}", status_code


# --- Endpoint for API File Uploads ---
@app.route('/api/ocr', methods=['POST'])
def upload_from_api():
    if not request.files:
        return jsonify({"error": "No file part in the request"}), 400
    files = {
        'single_file': request.files.get('file'),
        'side1_file': request.files.get('side1'),
        'side2_file': request.files.get('side2'),
    }
    result, status_code = process_uploaded_files(files)
    return jsonify(result), status_code


# --- NEW: Endpoint for API Base64 Uploads ---
@app.route('/api/ocr/base64', methods=['POST'])
def upload_from_base64():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    b64_single = data.get('image_base64')
    b64_side1 = data.get('side1_base64')
    b64_side2 = data.get('side2_base64')

    try:
        if b64_single:
            # Handle single Base64 image
            img_data = base64.b64decode(b64_single)
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(img_data)

            # Pre-process the image for OCR
            processed_path = preprocess_image_for_ocr(filepath, app.config['UPLOAD_FOLDER'])

            # Run the pipeline
            result, status_code = run_ocr_pipeline(processed_path, filename, filepath)
            return jsonify(result), status_code

        elif b64_side1 and b64_side2:
            # Handle two Base64 images
            s1_data = base64.b64decode(b64_side1)
            s2_data = base64.b64decode(b64_side2)

            s1_filename = f"{uuid.uuid4()}.jpg"
            s2_filename = f"{uuid.uuid4()}.jpg"
            s1_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s1_filename)
            s2_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s2_filename)

            with open(s1_filepath, 'wb') as f:
                f.write(s1_data)
            with open(s2_filepath, 'wb') as f:
                f.write(s2_data)

            # Combine into PDF
            pdf_path, pdf_filename = combine_images_to_pdf(s1_filepath, s2_filepath, app.config['UPLOAD_FOLDER'])
            if not pdf_path:
                return jsonify({"error": "Failed to combine images into PDF"}), 500

            # Run the pipeline
            result, status_code = run_ocr_pipeline(pdf_path, pdf_filename, s1_filepath)
            return jsonify(result), status_code

        else:
            return jsonify({"error": "Provide either 'image_base64' or both 'side1_base64' and 'side2_base64'"}), 400

    except base64.binascii.Error as e:
        return jsonify({"error": f"Invalid Base64 string: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')