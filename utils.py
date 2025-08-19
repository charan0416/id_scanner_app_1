# utils.py
import os
import uuid
import json
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import fitz

# Configure the Gemini API key
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except AttributeError as e:
    print(f"Error: GOOGLE_API_KEY not found. Please set it in your .env file. {e}")
    exit()


# --- File and Image Processing ---

def preprocess_image_for_ocr(image_path, output_folder):
    """
    Enhances an image for better OCR results by converting to grayscale,
    adjusting contrast, and sharpening.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"!!! PREPROCESSING ERROR: Could not read image file at {image_path}")
            return image_path  # Return original path if reading fails

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 3. Apply Sharpening Kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, sharpening_kernel)

        # Save the processed image
        processed_filename = f"processed_{uuid.uuid4()}.jpg"
        processed_filepath = os.path.join(output_folder, processed_filename)
        cv2.imwrite(processed_filepath, sharpened)
        print(f">>> IMAGE PREPROCESSING: Enhanced image saved to {processed_filepath}")
        return processed_filepath
    except Exception as e:
        print(f"!!! IMAGE PREPROCESSING CRITICAL ERROR: An exception occurred: {e}")
        return image_path  # Return original path on error


def combine_images_to_pdf(image1_path, image2_path, output_folder):
    """Combines two images into a single PDF."""
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(output_folder, pdf_filename)
        img1.save(pdf_path, save_all=True, append_images=[img2])
        return pdf_path, pdf_filename
    except Exception as e:
        print(f"Error combining images to PDF: {e}")
        return None, None


def convert_pdf_first_page_to_image(pdf_path, output_folder):
    """Converts the first page of a PDF to a JPG image and returns the path."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close();
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap()
        image_filename = f"pdf_page_{uuid.uuid4()}.jpg"
        image_filepath = os.path.join(output_folder, image_filename)
        pix.save(image_filepath)
        doc.close()
        return image_filepath
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None


# --- STABLE OPENCV DNN FACE DETECTION ---
def detect_and_extract_face(image_path, output_folder):
    """Detects a face using a stable OpenCV DNN model."""
    try:
        # These model files must be in your root project folder
        prototxt_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        img = cv2.imread(image_path)
        if img is None:
            print(f"!!! FACE DETECTION ERROR: Could not read image file at {image_path}")
            return None

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        best_detection_index = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, best_detection_index, 2]

        if confidence > 0.3:  # Using a lenient threshold for scanned documents
            print(f">>> FACE DETECTION: Found a face with confidence: {confidence:.2f}")
            box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            padding = 20
            startX = max(0, startX - padding);
            startY = max(0, startY - padding)
            endX = min(w, endX + padding);
            endY = min(h, endY + padding)

            face_cropped = img[startY:endY, startX:endX]

            if face_cropped.size == 0: return None

            face_filename = f"face_{uuid.uuid4()}.jpg"
            face_filepath = os.path.join(output_folder, face_filename)
            cv2.imwrite(face_filepath, face_cropped)
            print(f">>> FACE DETECTION SUCCESS: Face extracted to {face_filepath}")
            return face_filepath
        else:
            print(f">>> FACE DETECTION: No faces found with sufficient confidence (Max Confidence: {confidence:.2f}).")
            return None
    except Exception as e:
        print(f"!!! FACE DETECTION CRITICAL ERROR: An exception occurred: {e}")
        return None


# --- OCR with Google Gemini (UPDATED WITH FINAL PROMPT) ---
def call_gemini_ocr(file_path):
    """Performs OCR on an image or PDF file using Gemini and enriches the data."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        uploaded_file = genai.upload_file(path=file_path)

        # --- NEW, HIGHLY DETAILED PROMPT FOR EXTRACTION, PARSING, AND ENRICHMENT ---
        prompt = """
        Analyze the provided image(s) of a Philippine Driver's License. Extract, parse, and enrich the information into a single, structured JSON object.

        **CRITICAL INSTRUCTIONS:**
        1.  **JSON ONLY:** The entire output MUST be a single, valid JSON object. Do not wrap it in markdown. Your response must start with `{` and end with `}`.
        2.  **PARSE THE NAME:** From the full name (e.g., "TORRALBA, ROLDAN UNNE CASTRO"), extract the `lastName`, `firstName`, and `middleName`.
            - The `lastName` is the part before the comma.
            - The `firstName` is the first word after the comma and space.
            - The `middleName` is **all remaining text** on that line after the first name. This can include multiple words (like "UNNE CASTRO"). If no text remains, the value should be `null`.
        3.  **MAP DL CODES:** Based on the extracted `dlCodes`, create a `dlCodesDetails` object by mapping each code to its full description from the provided list.
        4.  **HANDLE CONDITIONS:**
            - For the `conditions` field, extract the exact value from the card (e.g., "1", "4", "1, 4"). If no conditions are listed, this field MUST be the string "NONE".
            - For the `conditionsDetails` field, this MUST ALWAYS be an array containing the full text of all five possible conditions, as listed below. This serves as a static reference list in every output.
        5.  **EXTRACT ORGAN DONATION:** For the `organDonation` field, you must extract the complete sentence written under that section (e.g., "I WILL NOT DONATE ANY ORGAN"). Do not summarize. If the section is blank or unreadable, the value should be `null`.
        6.  **ADHERE TO SCHEMA:** Use the exact field names and data types specified below. If a field is not present or unreadable, its value must be `null`.

        **FIELD-MAPPING DICTIONARIES (FOR YOUR REFERENCE):**

        *   **DL Codes Details Mapping:**
            - "A": "MOTORCYCLE"
            - "A1": "TRICYCLE"
            - "B": "UP TO 5000 KGS GVW/8 SEATS"
            - "B1": "UP TO 5000 KGS GVW/9 OR MORE SEATS"
            - "B2": "GOODS < 3500 KGS GVW"
            - "BE": "TRAILERS < 3500 KGS"
            - "C": "GOODS > 3500 KGS GVW"
            - "CE": "ARTICULATED > 3500 KGS COMBINED GVW"
            - "D": "BUS > 5000 KGS GVW/9 OR MORE SEATS"

        *   **Conditions Details Mapping (Use this to populate the static list):**
            - "1": "WEAR CORRECTIVE LENSES"
            - "2": "DRIVE ONLY W/SPECIAL EQPT FOR UPPER/LOWER LIMBS"
            - "3": "DRIVE CUSTOMIZED MOTOR VEHICLE ONLY"
            - "4": "DAYLIGHT DRIVING ONLY"
            - "5": "HEARING AID REQUIRED"

        **REQUIRED JSON OUTPUT SCHEMA:**

        {
          "fullName": "string",
          "lastName": "string",
          "firstName": "string",
          "middleName": "string_or_null",
          "nationality": "string",
          "sex": "string",
          "dateOfBirth": "string (YYYY/MM/DD)",
          "weight": "integer",
          "height": "float",
          "address": "string",
          "licenseNumber": "string",
          "expirationDate": "string (YYYY/MM/DD)",
          "agencyCode": "string",
          "bloodType": "string",
          "eyesColor": "string",
          "dlCodes": ["array_of_strings"],
          "dlCodesDetails": { "object_key_value_pairs" },
          "conditions": "string (The value from the card, e.g., '1' or 'NONE')",
          "conditionsDetails": ["WEAR CORRECTIVE LENSES", "DRIVE ONLY W/SPECIAL EQPT FOR UPPER/LOWER LIMBS", "DRIVE CUSTOMIZED MOTOR VEHICLE ONLY", "DAYLIGHT DRIVING ONLY", "HEARING AID REQUIRED"],
          "serialNumber": "integer",
          "emergencyContactName": "string",
          "emergencyContactAddress": "string",
          "emergencyContactTel": "string",
          "organDonation": "string (e.g., 'I WILL NOT DONATE ANY ORGAN' or null)"
        }
        """

        response = model.generate_content(
            [prompt, uploaded_file],
            safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
        )

        print(f"--- RAW GEMINI RESPONSE ---\n{response.text}\n-------------------------")

        if not response.text.strip():
            print("!!! GEMINI ERROR: Received an empty response. Likely due to safety filters.")
            if response.prompt_feedback:
                print(f"Prompt Feedback from API: {response.prompt_feedback}")
            return {"error": "API returned an empty response, likely blocked by safety filters."}

        # Clean the response just in case the model adds markdown despite instructions
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        data = json.loads(cleaned_text)
        return data

    except json.JSONDecodeError as e:
        print(f"!!! JSON DECODE ERROR: Failed to parse Gemini response as JSON. Error: {e}")
        return {"error": f"Failed to parse non-JSON response from API.", "raw_response": response.text}
    except Exception as e:
        print(f"!!! An unexpected error occurred calling Gemini API: {e}")
        return {"error": str(e)}