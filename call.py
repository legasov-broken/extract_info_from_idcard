
import json
import cv2
from flask import Flask, request, jsonify

from func.crop_image import crop_image
from func.detect_word import detect_word
from func.OCR import OCR
from func.arr_to_dict import Convert
from func.crop_face import crop_face

from func.processing_tools import url_to_image

app = Flask(__name__)

@app.route('/api/v1/extract_in4', methods=['POST'])
def api():
    data = json.loads(request.data)
    img_url = data['img_url']
    ima = url_to_image(img_url)



    # weight of crop_image
    crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

    # weight of detect_word
    detect_word_model_path = './weight/idcard_yolo_OCR/OCR_3000.pt'

    a = crop_image(ima,crop_img_model_path)
    id, name, dob, sex, nati, place, resi, date = detect_word(a,detect_word_model_path)
    in4 = OCR(id, name, dob, sex, nati, place, resi, date)
    infor = Convert(in4)

    
    return jsonify(infor)
app.run()

@app.route('/api/v1/extract_face', methods =['POST'])
def face_api():
    data = json.loads(request.data)
    img_url = data['img_url']
    ima = url_to_image(img_url)
    

    # weight of crop_image
    crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

    # weight of detect_face
    crop_face_model_path = './weight/idcard_extract_face/face_detect.pt'

#------------------------------------------------------------------------------------------------


# output
    a = crop_image(ima,crop_img_model_path)
    loc_face = crop_face(a, crop_face_model_path)
    
    return jsonify(loc_face)
app.run()