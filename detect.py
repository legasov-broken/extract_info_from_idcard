import cv2
from flask import Flask, request, jsonify

from func.crop_image import crop_image
from func.detect_word import detect_word
from func.OCR import OCR
from func.arr_to_dict import Convert

from func.processing_tools import url_to_image

# insert path here

#------------------------------------------------------------------------------------------------

# input image
img = cv2.imread('./test_image/7.jpg') 
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# weight of crop_image
crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

# weight of detect_word
detect_word_model_path = './weight/idcard_yolo_OCR/OCR_3000.pt'

#------------------------------------------------------------------------------------------------


# output
inp = crop_image(image,crop_img_model_path)
id, name, dob, sex, nati, place, resi, date = detect_word(inp,detect_word_model_path)
in4 = OCR(id, name, dob, sex, nati, place, resi, date)

print(Convert(in4))