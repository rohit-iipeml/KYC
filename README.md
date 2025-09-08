# 🪪 KYC Automation System (ID Card Verification)

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20Framework-green?logo=fastapi)](https://fastapi.tiangolo.com/)  
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange?logo=yolo)](https://github.com/ultralytics/ultralytics)  

> End-to-end **KYC (Know Your Customer) automation system** built with  
> **YOLO + PyTesseract + DeepFace + FastAPI**.  
> Provides APIs for **ID card information extraction, face verification, and signature matching**.  

---

## ✨ Features

- 📄 **ID Card Information Extraction**  
  Detects fields (*Name, DOB, ID Number, Expiry Date, Address, etc.*) using YOLO + PyTesseract.  

- 🙂 **Face Verification**  
  Compares ID card photo vs. selfie using **DeepFace (Facenet)** with similarity scoring.  

- ✍️ **Signature Verification**  
  Extracts signatures with YOLO & compares using **SSIM + contour similarity metrics**.  

- ⚡ **FastAPI REST APIs**  
  - `/extract-info/` → Extract structured info from ID cards.  
  - `/face-match/` → Verify ID card photo vs. selfie.  
  - `/compare-images/` → Match signatures.  

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+  
- **Framework**: FastAPI, Uvicorn  
- **Models**: YOLO (Ultralytics), DeepFace (Facenet)  
- **Libraries**: OpenCV, PyTesseract, scikit-image  

---

## 📂 Project Structure

├── controllers/
│ └── kyc_controller.py # API routes
├── ml_models/ # Pretrained YOLO models
│ ├── face_dtcn.pt
│ ├── info_extrcn_new_latest.pt
│ └── sign_model.pt
├── models/
│ └── file_model.py # Pydantic schemas
├── services/
│ └── kyc_service.py # Core logic (OCR, matching, preprocessing)
├── uploads/ # Uploaded test images
│ ├── id_card.jpg
│ ├── selfie.jpg
│ └── image1.jpg, image2.jpg
├── main.py # Entry point
├── Requirements.txt # Dependencies
└── README.md # Documentation

---

## ⚡ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/kyc-automation.git
cd kyc-automation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r Requirements.txt

# Run the FastAPI server
python main.py

➡️ Open Swagger UI at: http://127.0.0.1:8080/docs

➡️ Or Redoc at: http://127.0.0.1:8080/redoc

📌 API Endpoints
🔹 Extract Information
POST /extract-info/


Input:

{ "file_URL": "<ID card image URL>" }


Output: JSON with extracted fields

🔹 Face Verification
POST /face-match/


Input:

{ "file_URL_1": "<ID card URL>", "file_URL_2": "<Selfie URL>" }


Output: Verification result + similarity score

🔹 Signature Matching
POST /compare-images/


Input:

{ "file_URL_1": "<ID card URL>", "file_URL_2": "<Signature URL>" }


Output: Similarity index + verification result

📸 Example Workflow

Upload ID card & selfie → System verifies if same person.

Upload ID card & signature → System validates authenticity.

Extract structured ID info → Automate KYC workflows.

🔮 Future Enhancements

🌍 Multi-language OCR

🛡️ Face anti-spoofing detection

🐳 Dockerized deployment

🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to fork the repo & submit PRs 🚀
