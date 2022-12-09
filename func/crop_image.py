from func.lib import *

from func.letterbox import letterbox


def crop_image(img_1,crop_img_model_path):
    image = img_1
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image,960,stride=32)[0]
    # image = cv2.resize(image, [1280, 1280])

    cap = img_1
    # cap = cv2.resize(cap, [1280, 1280])
    cap = letterbox(cap,960,stride=32)[0]

    image_1 = torch.tensor(image)
    image_1 = torch.transpose(image_1, 2, 0)
    image_1 = torch.unsqueeze(image_1, 0)
    image_1 = image_1.float()
    image_1/=255.0
    device = select_device('cpu')
    model = attempt_load(crop_img_model_path, map_location=None)
    pred = model(image_1.to('cpu'))[0]
    pred = non_max_suppression(pred, 0.5)


    pred = pred[0].detach().numpy()

    # caculate 4th corner if pred has 3 corners
    boxes = np.transpose([(pred[:,1]+pred[:,3])/2,(pred[:,0]+pred[:,2])/2,pred[:,5]])


    boxes1 = boxes
    boxes2 = boxes
    # boxes = boxes.astype(int)

    if len(boxes1) == 3 and len(np.unique(boxes1[:,2])) == 3:
        for i in range(0,4):
            if i not in boxes[:,2]:
                if abs(boxes[0,0]-boxes[1,0])<100:
                    x_0 = boxes[2,0]
                if abs(boxes[0,0]-boxes[2,0])<100:
                    x_0 = boxes[1,0]
                if abs(boxes[1,0]-boxes[2,0])<100:
                    x_0 = boxes[0,0]
                if abs(boxes[0,1]-boxes[1,1])<100:
                    y_0 = boxes[2,1]
                if abs(boxes[0,1]-boxes[2,1])<100:
                    y_0 = boxes[1,1]
                if abs(boxes[1,1]-boxes[2,1])<100:
                    y_0 = boxes[0,1]
                a_0 =[x_0,y_0,i]
                boxes = np.vstack([boxes,a_0])
                break
            
        # caculate location of ID card
    if len(boxes) == 4 and len(np.unique(boxes[:,2])) == 4:
        
        label = {'top_left':0, 'top_right':1, 'bottom_left':2,'bottom_right':3}
        for i in boxes:
            if i[2] == 0:
                label['top_left'] = [i[0], i[1]]
            if i[2] == 1:
                label['top_right'] = [i[0], i[1]]
            if i[2] == 2:
                label['bottom_left'] = [i[0], i[1]]
            if i[2] == 3:
                label['bottom_right'] = [i[0], i[1]]

        # transpose ID card to standard size
        pts1 = np.float32([label['top_left'], label['top_right'],
                            label['bottom_left'], label['bottom_right']])
        pts2 = np.float32([[0, 0], [500, 0],
                            [0, 300], [500, 300]])
            
    
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(cap, matrix, (500, 300))
        out = result

    return out

