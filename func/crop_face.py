from func.lib import *

from func.letterbox import letterbox

def crop_face(img_1,detect_word_model_path):
    image = img_1
    img = letterbox(image, 640, stride=32)[0]
    image_1 = img[:, :, ::-1].transpose(2, 0, 1)
    device = select_device('cpu')
    image_1 = torch.from_numpy(np.array(image_1)).to(device).float()
    image_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    image_1 = image_1.unsqueeze(0)


    model = attempt_load(detect_word_model_path,map_location=device)
    pred = model(image_1.to('cpu'))[0]
    pred = non_max_suppression(pred, 0.9)
    pred = pred[0].detach().numpy()
    face_loc = pred[0][:4]

    return face_loc

