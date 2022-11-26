from lib import *
from crop_image import crop_image
from detect_word import detect_word
from OCR import OCR
from arr_to_dict import Convert

# insert path here

#------------------------------------------------------------------------------------------------

# input image
image = cv2.imread('./test_image/IMG_6502.JPG') 

# weight of crop_image
crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

# weight of detect_word
detect_word_model_path = './weight/idcard_yolo_OCR/OCR_3000.pt'

#------------------------------------------------------------------------------------------------


a = crop_image(image,crop_img_model_path)

id, name, dob, sex, nati, place, resi, date = detect_word(a,detect_word_model_path)

in4 = OCR(id, name, dob, sex, nati, place, resi, date)

print(Convert(in4))