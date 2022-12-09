from func.lib import *

from func.letterbox import letterbox


def detect_word(result,detect_word_model_path):
    image = result
    img = letterbox(image, 640, stride=32)[0]
    image_1 = img[:, :, ::-1].transpose(2, 0, 1)

    device = select_device('cpu')
    image_1 = torch.from_numpy(np.array(image_1)).to(device).float()
    image_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    image_1 = image_1.unsqueeze(0)


    model = attempt_load(detect_word_model_path,map_location=device)
    pred = model(image_1.to('cpu'))[0]
    pred = non_max_suppression(pred, 0.6)

    pred = pred[0].detach().numpy()
    pred = np.delete(pred, 4, 1)
    pred_ = pred

    # extract name, place, residence

    box_1 = []
    box_5 = []
    box_6 = []
    for j in range(len(pred[:,4])):
        if pred[j,4] == 1:
            box_1 = np.append(box_1, pred[j,:], axis=0)
        if pred[j,4] == 5:
            box_5 = np.append(box_5, pred[j,:], axis=0)
        if pred[j,4] == 6:
            box_6 = np.append(box_6, pred[j,:], axis=0)
    box_1 = np.reshape(box_1,(-1, 5))
    box_5 = np.reshape(box_5,(-1, 5))
    box_6 = np.reshape(box_6,(-1, 5))

    # name

    box_1_a = box_1
    box_1 = sorted(box_1, key=lambda x:x[1])
    box_1 = np.array(box_1)

    minus = []
    for i in range(len(box_1)-1):
        _minus = box_1[i+1,1] - box_1[i,1]
        minus.append(_minus)

    minus = np.array(minus)

    _box_1 = []
    box_1_ = []

    for j in range(0,len(minus)):
        if minus[j]>10:
            _box_1 = box_1[:j+1,:]
            box_1_ = box_1[j+1:,:]
        
    if box_1_ == [] and _box_1 ==[]:
        box_1 = sorted(box_1_a, key=lambda x:x[0])
        box_1 = np.array(box_1)
    
    else:
        _box_1 = np.reshape(_box_1,(-1, 5))  
        _box_1 = np.unique(_box_1, axis = 0)
        _box_1 = sorted(_box_1, key=lambda x:x[0])
        _box_1 = np.array(_box_1)

        box_1_ = np.reshape(box_1_,(-1, 5))  
        box_1_ = np.unique(box_1_, axis = 0)
        box_1_ = sorted(box_1_, key=lambda x:x[0])
        box_1_ = np.array(box_1_)

        box_1 = np.vstack([_box_1,box_1_])
        
    # place

    box_5_a = box_5
    box_5 = sorted(box_5, key=lambda x:x[1])
    box_5 = np.array(box_5)

    minus = []
    for i in range(len(box_5)-1):
        _minus = box_5[i+1,1] - box_5[i,1]
        minus.append(_minus)

    minus = np.array(minus)

    _box_5 = []
    box_5_ = []
    for j in range(0,len(minus)):
        if minus[j]>50:
            _box_5 = box_5[:j+5,:]
            box_5_ = box_5[j+5:,:]
        
    if box_5_ == [] and _box_5 ==[]:
        box_5 = sorted(box_5_a, key=lambda x:x[0])
        box_5 = np.array(box_5)
    else:
        _box_5 = np.reshape(_box_5,(-5, 5))  
        _box_5 = np.unique(_box_5, axis = 0)
        _box_5 = sorted(_box_5, key=lambda x:x[0])
        _box_5 = np.array(_box_5)

        box_5_ = np.reshape(box_5_,(-5, 5))  
        box_5_ = np.unique(box_5_, axis = 0)
        box_5_ = sorted(box_5_, key=lambda x:x[0])
        box_5_ = np.array(box_5_)

        box_5 = np.vstack([_box_5,box_5_])

    # residence

    box_6 = sorted(box_6, key=lambda x:x[1])
    box_6 = np.array(box_6)

    minus = []
    for i in range(len(box_6)-1):
        _minus = box_6[i+1,1] - box_6[i,1]
        minus.append(_minus)

    minus = np.array(minus)

    _box_6 = []
    box_6_ = []
    for j in range(0,len(minus)):
        if minus[j]>10:
            _box_6 = box_6[:j+1,:]
            box_6_ = box_6[j+1:,:]

        _box_6 = np.reshape(_box_6,(-1, 5))  
        _box_6 = np.unique(_box_6, axis = 0)
        _box_6 = sorted(_box_6, key=lambda x:x[0])
        _box_6 = np.array(_box_6)

        box_6_ = np.reshape(box_6_,(-1, 5))  
        box_6_ = np.unique(box_6_, axis = 0)
        box_6_ = sorted(box_6_, key=lambda x:x[0])
        box_6_ = np.array(box_6_)

        box_6 = np.vstack([_box_6,box_6_])

    image__ = image
    image__ = letterbox(image__, 640, stride=32)[0]

    for i in range(0,len(pred_[:,4])):    
        if pred_[i,4] == 0:
            id = image__[int(pred_[i][1]):int(pred_[i][3]),int(pred_[i][0]):int(pred_[i][2])]
        if pred_[i,4] == 2:
            dob = image__[int(pred_[i][1]):int(pred_[i][3]),int(pred_[i][0]):int(pred_[i][2])]
        if pred_[i,4] == 3:
            sex = image__[int(pred_[i][1]):int(pred_[i][3]),int(pred_[i][0]):int(pred_[i][2])]
        if pred_[i,4] == 4:
            nati = image__[int(pred_[i][1]):int(pred_[i][3]),int(pred_[i][0]):int(pred_[i][2])]
        if pred_[i,4] == 7:
            date = image__[int(pred_[i][1]):int(pred_[i][3]),int(pred_[i][0]):int(pred_[i][2])]

    name = []
    place = []
    resi = []

    for i in range(0,len(box_1[:,4])):
        name_ = image__[int(box_1[i][1]):int(box_1[i][3]),int(box_1[i][0]):int(box_1[i][2])]
        name.append(name_)
    
    for i in range(0,len(box_5[:,4])):
        place_ = image__[int(box_5[i][1]):int(box_5[i][3]),int(box_5[i][0]):int(box_5[i][2])]
        place.append(place_)
    
    for i in range(0,len(box_6[:,4])):
        resi_ = image__[int(box_6[i][1]):int(box_6[i][3]),int(box_6[i][0]):int(box_6[i][2])]
        resi.append(resi_)

    return id, name, dob, sex, nati, place, resi, date