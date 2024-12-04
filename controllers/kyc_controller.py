import cv2
import os
import numpy as np
import requests
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from services.kyc_service import *
from models.file_model import *

app = FastAPI()

UPLOAD_DIR = "uploads"

class ModelPaths:
    SIGNATURE_MODEL = './ml_models/sign_model.pt'
    INFO_EXTRACTION_MODEL = './ml_models/info_extrcn_new_latest.pt'

@app.post("/compare-images/")
async def compare_images(files: ImageURLModelTwo):
    try:
        id_card_response = requests.get(files.file_URL_1)
        signature_response = requests.get(files.file_URL_2)

        if id_card_response.status_code != 200 or signature_response.status_code != 200:
            return {"error": "Failed to fetch one or more images from provided URLs"}

        id_card_image = cv2.imdecode(np.frombuffer(id_card_response.content, np.uint8), cv2.IMREAD_COLOR)
        signature_image = cv2.imdecode(np.frombuffer(signature_response.content, np.uint8), cv2.IMREAD_COLOR)

        model = YOLO(ModelPaths.SIGNATURE_MODEL)
        model_image1 = extract_signature(id_card_image, model)

        ssim_value, contour_similarity = match_images(model_image1, signature_image)

        contour_similarity_threshold = 0.25
        ssim_threshold = 50

        if ssim_value is not None and contour_similarity is not None:
            similarity_index = ssim_value * 100
            contour_similarity_index = contour_similarity * 100

            if ssim_value >= ssim_threshold and contour_similarity <= contour_similarity_threshold:
                match_result = True
                message = "Successfully verified"
            else:
                match_result = False
                message = "We are unable to match the signature document and drawn signature."
        else:
            similarity_index = 0
            contour_similarity_index = 0
            match_result = False
            message = "We are unable to match the signature document and drawn signature."

        return {
            "similarity_index": similarity_index,
            "contour_similarity": contour_similarity_index,
            "match": match_result,
            "message": message
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.post("/extract-info/")
async def extract_info_from_image(files: ImageURLModel):
    try:
        id_card_response = requests.get(files.file_URL)

        if id_card_response.status_code != 200:
            return {"error": "Failed to fetch the image from the provided URL"}

        id_card_image = cv2.imdecode(np.frombuffer(id_card_response.content, np.uint8), cv2.IMREAD_COLOR)
        model = YOLO(ModelPaths.INFO_EXTRACTION_MODEL)

        class_names = ['country','id_type','id_num','issue_date','exp_date','dob','sex','fname','mname','lname','add_1','add_2','add_3']

        extracted_info = extracted_information_from_boxes(model, id_card_image, class_names)
        if extracted_info is None:
            return {"message": "We are unable to extract information from the given document, please upload a crisp/vivid/bright document/selfie again."}
        else:
            return JSONResponse(content=extracted_info)
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.post("/face-match/")
async def face_match(files: ImageURLModelTwo):
    # return True
    try:
        id_card_response = requests.get(files.file_URL_1)
        selfie_response = requests.get(files.file_URL_2)

        if id_card_response.status_code != 200 or selfie_response.status_code != 200:
            return {"error": "Failed to fetch one or more images from provided URLs"}

        id_card_image = cv2.imdecode(np.frombuffer(id_card_response.content, np.uint8), cv2.IMREAD_COLOR)
        selfie_image = cv2.imdecode(np.frombuffer(selfie_response.content, np.uint8), cv2.IMREAD_COLOR)

        # Save decoded images to temporary files
        id_card_path = os.path.join(UPLOAD_DIR, "id_card.jpg")
        cv2.imwrite(id_card_path, id_card_image)

        selfie_path = os.path.join(UPLOAD_DIR, "selfie.jpg")
        cv2.imwrite(selfie_path, selfie_image)

        is_same_person, similarity_score = verify_faces(id_card_path, selfie_path)

        if is_same_person:
            verification_result = "Successfully verified"
        else:
            verification_result = "We are unable to match the Document profile picture and Selfie submitted by you."
            # return {"result": verification_result, "message": "We are unable to match the Document profile picture and Selfie submitted by you."}

        return {
            "result": is_same_person,
            "similarity_score": (1-similarity_score)*100,
            "message": verification_result
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
