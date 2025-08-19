Advanced OCR and Face Extraction Web Application
This is a comprehensive Flask web application designed to perform Optical Character Recognition (OCR) on identity documents, specifically Philippine Driver's Licenses. It uses Google's Gemini AI for highly accurate, structured data extraction and OpenCV for advanced image pre-processing, including perspective correction and face detection.
The application provides a user-friendly web interface for uploads, a camera capture feature, and a robust REST API for programmatic access.
Features
Multiple Upload Methods:
Web Form: Upload single or two-sided images (JPG, PNG) and PDFs directly.
Live Camera: Capture an image using your device's camera for instant processing.
REST API: Programmatically submit files via multipart/form-data or Base64-encoded JSON payloads.
Advanced Image Pre-processing:
Perspective Correction: Automatically detects the corners of an ID card in a photo taken at an angle and warps it into a flat, rectangular image.
OCR Enhancement: Applies contrast adjustment and sharpening to improve text legibility for the AI model.
Intelligent OCR with Google Gemini:
Extracts all relevant fields from the ID card into a structured JSON format.
Parses full names into lastName, firstName, and middleName, correctly handling multi-word middle names.
Enriches data by mapping DL Codes and Conditions to their full text descriptions.
Robust Two-Pass Face Detection:
A resilient computer vision pipeline attempts to detect the face on the corrected image.
If it fails, it applies a targeted sharpening enhancement and tries again, ensuring high success rates even on blurry or low-quality images.
Persistent History:
All successful OCR results, including the extracted JSON data and the cropped face image, are saved to a PostgreSQL database.
A "History" page allows users to review all past submissions.
Technology Stack
Backend: Flask, Flask-SQLAlchemy
Database: PostgreSQL
OCR Engine: Google Gemini AI (gemini-1.5-flash)
Image Processing: OpenCV, PyMuPDF (for PDFs), Pillow
Frontend: Jinja2 Templates, Bootstrap 5, JavaScript (for Camera and Base64 API)
Environment: Python 3.8+, Pip with Virtual Environment
Setup and Installation
Follow these steps to get the application running on your local machine.
1. Prerequisites
Python 3.8+ and Pip: Download from python.org. During installation on Windows, ensure you check the box that says "Add Python to PATH".
Git: Download from git-scm.com.
Docker Desktop: The recommended way to run PostgreSQL. Download from docker.com.
Google AI API Key: Get one from Google AI Studio.
2. Clone the Repository
Open a terminal (like Terminal on macOS/Linux or PowerShell/Git Bash on Windows) and run:
code
Bash
git clone <your-repo-url>
cd <your-project-folder>
3. Download Face Detection Models
The OpenCV face detector requires two model files. Download them and place them in the root of your project folder.
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
4. Configure Environment Variables
Create a file named .env in the root of the project directory. Do not include quotes around the values. Copy the content below and fill in your credentials.
code
Ini
# .env

# --- Google Gemini AI API Key ---
GOOGLE_API_KEY=PASTE_YOUR_GOOGLE_AI_API_KEY_HERE

# --- PostgreSQL Database Connection ---
# These credentials must match your database setup
DB_USER=postgres
DB_PASSWORD=your_strong_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ocr_db
5. Set Up Python Environment
It is highly recommended to use a virtual environment.
code
Bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS and Linux:
source venv/bin/activate
# On Windows (in PowerShell):
# .\venv\Scripts\Activate.ps1
# On Windows (in Command Prompt):
# .\venv\Scripts\activate.bat

# Install the required packages
pip install -r requirements.txt
6. Set Up the PostgreSQL Database with Docker
Using Docker is the most consistent method across all operating systems.
Start Docker Desktop.
Run the following command in your terminal. This will download the PostgreSQL image and start a container.
code
Bash
docker run --name ocr-db \
  -e POSTGRES_USER="postgres" \
  -e POSTGRES_PASSWORD="your_strong_password" \
  -e POSTGRES_DB="ocr_db" \
  -p 5432:5432 \
  -d postgres
Note: The password must match what you put in your .env file.
IMPORTANT: After running this command, wait 15-20 seconds for the database to fully initialize inside the container before proceeding to the next step.
7. Initialize the Database Tables
With your virtual environment activated and the database running, create the application's tables.
code
Bash
python db_setup.py
You should see the message: Tables created successfully!
Running the Application
Make sure your virtual environment is activated.
Make sure your PostgreSQL Docker container is running. (Check with docker ps).
Start the Flask server:
code
Bash
python app.py
Open your web browser and navigate to http://127.0.0.1:5000.
API Usage
The application exposes a powerful REST API for programmatic integration.
Endpoint: /api/ocr/base64
Method: POST
Headers: Content-Type: application/json
Body: A JSON object containing Base64-encoded image strings.
Scenario 1: Single Image
code
JSON
{
    "image_base64": "UklGRlq3AABXRUJQVlA... (very long Base64 string)"
}
Scenario 2: Two Images (Front and Back)
code
JSON
{
    "side1_base64": "UklGRlq3AABXRUJQVlA... (Base64 for front image)",
    "side2_base64": "iVBORw0KGgoAAAANSUhEUgA... (Base64 for back image)"
}
Example curl Request (Cross-Platform)
The command to convert an image to Base64 varies slightly by OS.
On macOS: base64 -i /path/to/image.jpg
On Linux: base64 -w 0 /path/to/image.jpg
On Windows (PowerShell): [Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\path\to\image.jpg'))
You can pre-convert the image and paste the string into an API client like Postman or use the command line directly.
Example curl for macOS/Linux:
code
Bash
curl -X POST http://127.0.0.1:5000/api/ocr/base64 \
-H "Content-Type: application/json" \
-d '{ "image_base64": "'$(base64 -i /path/to/your/image.jpg)'" }'
Project Structure
code
Code
.
├── .env                  # Environment variables (you create this)
├── app.py                # Main Flask application logic and routes
├── db_setup.py           # Script to initialize database tables
├── deploy.prototxt       # OpenCV model dependency
├── models.py             # SQLAlchemy database model
├── requirements.txt      # Python package dependencies
├── res10_300x300_ssd_iter_140000.caffemodel # OpenCV model dependency
├── static/
│   └── css/
│       └── style.css     # Custom stylesheets
├── templates/
│   ├── history.html      # Page to display OCR results
│   ├── index.html        # Main upload page
│   └── layout.html       # Base HTML template with navbar
└── utils.py              # Core logic for OCR, face detection, image processing
Troubleshooting
FATAL: database "ocr_db" does not exist: This usually means you ran python db_setup.py too quickly after starting the Docker container. Stop/remove the container (docker stop ocr-db, docker rm ocr-db), start it again, wait 15-20 seconds, and then run the setup script.
FATAL: role "your_user" does not exist: Your DB_USER in the .env file does not match the user in your PostgreSQL database. Ensure they are identical.
Docker TLS handshake timeout: This is a network issue. Try restarting Docker Desktop, disabling any VPN/proxy, or changing Docker's DNS settings to 8.8.8.8 (in Settings -> Docker Engine).
Command Not Found Errors (e.g., python or pip): This typically means Python was not added to your system's PATH during installation. Re-install Python and ensure the "Add to PATH" checkbox is selected.