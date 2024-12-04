import numpy as np
import cv2
import pytesseract
from fastapi import UploadFile, File
from deepface import DeepFace
from skimage.metrics import structural_similarity as ssim
import os
import shutil

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised_img = cv2.fastNlMeansDenoising(thresh_img, None, 10, 7, 21)

    return denoised_img

def extracted_information_from_boxes(model, image, class_names):
    extracted_info_preprocessed = {}
    results = model(image,conf=0.5)
    boxes = results[0].boxes
    boxes = boxes.data

    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max, confidence, class_id = bbox.cpu().numpy().astype(float)
        class_id = int(class_id)

        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        cropped_img = image[y_min:y_max, x_min:x_max]

        preprocessed_img = preprocess_image(cropped_img)

        extracted_text_preprocessed = pytesseract.image_to_string(preprocessed_img,config='--psm 7')

        class_name = class_names[class_id]

        extracted_info_preprocessed[class_name] = extracted_text_preprocessed.strip()
    
    return extracted_info_preprocessed

def extract_signature(image, signature_model):
    results = signature_model(image)
    boxes = results[0].boxes

    xyxy_bbox = boxes.xyxy
    xmin = int(xyxy_bbox[0][0])
    ymin = int(xyxy_bbox[0][1])
    xmax = int(xyxy_bbox[0][2])
    ymax = int(xyxy_bbox[0][3])

    signature_image = image[ymin:ymax, xmin:xmax]

    return signature_image


# Define the directory to save uploaded files
UPLOAD_DIR = "uploads"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize DeepFace models
DeepFace.build_model("Facenet")
DeepFace.build_model("Facenet")

def verify_faces(image1, image2):
    result = DeepFace.verify(image1, image2, model_name="Facenet")
    verified = result['verified']
    similarity_score = result['distance']
    return verified, similarity_score

def upload_file(uploaded_file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(file_path, "wb") as file:
        shutil.copyfileobj(uploaded_file.file, file)
    return file_path

def remove_white_space(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect

def match_images(img1, img2):
    img1 = remove_white_space(img1)
    img2 = remove_white_space(img2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_resized = cv2.resize(img1_gray, (300, 300))
    img2_resized = cv2.resize(img2_gray, (300, 300))

    # Calculate SSIM
    ssim_value = ssim(img1_resized, img2_resized, gaussian_weights=True, sigma=1.2, use_sample_covariance=False) * 100

    # Calculate shape similarity using cv2.matchShapes
    contours1, _ = cv2.findContours(img1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if contours1 and contours2:
        contour_similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0)
    else:
        contour_similarity = None

    return ssim_value, contour_similarity