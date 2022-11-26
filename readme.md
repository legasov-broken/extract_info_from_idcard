# Extract new Vietnamese ID Card from image use YOLOv7 and VietOCR

The idea of my project's pipeline is from: 
>https://fpt.ai/technical-view-fvi-end-end-vietnamese-id-card-ocr


0. (Optional) Create virtual environment use Anaconda

`conda create -n id_card`
`conda activate id_card`


1. Install requirement library

`pip install -r requirements.txt`

2. Change path in `detect.py`

You can change the path of the image that need to extract information in:

`image = cv2.imread(__your path here__)` 

3. Download weight

Put weights in this folder or you can modify its path in `detect.py`

Download link:
>https://drive.google.com/drive/folders/1bA06kCuRYHo19H3DQo1CcTDcFQ8Khyih?usp=share_link

If google drive archives it, unzip/rar in this folder.

4. Run `detect.py`

...and see the result.

### Hope my small project could help you!

<sub>*from minelove with love*<sub>



