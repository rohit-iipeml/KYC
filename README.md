# apex-markets-kyc-ml-api
🪪 KYC Automation System (ID Card Verification)

This project is an end-to-end KYC (Know Your Customer) automation system that leverages YOLO object detection, PyTesseract OCR, and DeepFace for identity verification. It provides APIs for ID card information extraction, face verification, and signature matching.

The system was built by first annotating ID cards (photos, signatures, and textual fields), training YOLO models, and integrating them with OCR and similarity metrics.

🚀 Features

ID Card Information Extraction

Detects fields like Name, DOB, ID Number, Expiry Date, Address, etc. using YOLO.

Uses PyTesseract OCR for text extraction.

Face Verification

Compares ID card photo against a selfie using DeepFace (Facenet).

Returns verification result with similarity score.

Signature Verification

Extracts signature from ID card using YOLO.

Matches with a drawn/uploaded signature using SSIM + contour similarity metrics.

REST API Endpoints (FastAPI)

/extract-info/ → Extract structured info from ID card.

/face-match/ → Verify ID card photo vs. selfie.

/compare-images/ → Match signatures.

🛠️ Tech Stack

Python 3.8+

FastAPI
 – API Framework

YOLO (Ultralytics)
 – Object Detection

PyTesseract
 – OCR Engine

DeepFace
 – Face Verification

OpenCV
 – Image Preprocessing & Similarity

scikit-image
 – Structural Similarity (SSIM)

📂 Project Structure
├── controllers/
│   └── kyc_controller.py       # FastAPI routes
├── ml_models/                  # Pretrained YOLO models
│   ├── face_dtcn.pt
│   ├── info_extrcn_new_latest.pt
│   └── sign_model.pt
├── models/
│   └── file_model.py           # Pydantic data models
├── services/
│   └── kyc_service.py          # Core logic (OCR, matching, preprocessing)
├── uploads/                    # Uploaded test files
│   ├── id_card.jpg
│   ├── selfie.jpg
│   └── image1.jpg, image2.jpg
├── main.py                     # Entry point (FastAPI app + Uvicorn)
├── Requirements.txt            # Dependencies
└── README.md                   # Project Documentation

⚡ Setup & Installation

Clone the repository

git clone https://github.com/yourusername/kyc-automation.git
cd kyc-automation


Create a virtual environment

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r Requirements.txt


Run the FastAPI server

python main.py


Access APIs

Swagger UI: http://127.0.0.1:8080/docs

Redoc: http://127.0.0.1:8080/redoc

📌 API Endpoints
1. Extract Information from ID Card
POST /extract-info/


Input: { "file_URL": "<ID card image URL>" }
Output: Extracted fields as JSON.

2. Face Verification
POST /face-match/


Input: { "file_URL_1": "<ID card URL>", "file_URL_2": "<Selfie URL>" }
Output: Verification result + similarity score.

3. Signature Matching
POST /compare-images/


Input: { "file_URL_1": "<ID card URL>", "file_URL_2": "<Signature URL>" }
Output: Similarity index + verification result.

📸 Example Use Case

Upload ID card & selfie → Verify if the person is the same.

Upload ID card & drawn signature → Verify authenticity.

Extract structured information for KYC automation workflows.

🔮 Future Improvements

Multi-language OCR support.

Enhanced anti-spoofing for face verification.

Deployable Docker image for scalable usage.

🤝 Contributions

Feel free to fork the repo, raise issues, or submit PRs to improve the pipeline.
