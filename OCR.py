from lib import *



def OCR(id, name, dob, sex, nati, place, resi, date): 
    
    config = Cfg.load_config_from_file('./vietocr/base.yml')
    config_1 = Cfg.load_config_from_file('./vietocr/vgg-seq2seq.yml')
    config.update(config_1)
    config['weights'] = './weight/seq2seqocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    # config['predictor']['beamsearch']=False
    # config['beamsearch']=True

    detector = Predictor(config)

#--------------single_model--------------

    id_ = ['id']
    id = PIL.Image.fromarray(id)
    _id = detector.predict(id)
    id_.append(_id)

    dob_ = ['dob']
    dob = PIL.Image.fromarray(dob)
    _dob = detector.predict(dob)
    dob_.append(_dob)

    sex_ = ['sex']
    sex = PIL.Image.fromarray(sex)
    _sex = detector.predict(sex)
    sex_.append(_sex)

    nati_ = ['nati']
    nati = PIL.Image.fromarray(nati)
    _nati = detector.predict(nati)
    nati_.append(_nati)

    date_ = ['date']
    date = PIL.Image.fromarray(date)
    _date = detector.predict(date)
    date_.append(_date)

#------------multi_label---------- 

    name_ = []
    for _name in name:
        _name = PIL.Image.fromarray(_name)
        name_1 = detector.predict(_name)
        name_.append(name_1)

    name_ = ' '.join(name_)
    name = ['name']
    name.append(name_)

    place_ = []
    for _place in place:
        _place = PIL.Image.fromarray(_place)
        place_1 = detector.predict(_place)
        place_.append(place_1)

    place_ = ' '.join(place_)
    place = ['place']
    place.append(place_)

    resi_ = []
    for _resi in resi:
        _resi = PIL.Image.fromarray(_resi)
        resi_1 = detector.predict(_resi)
        resi_.append(resi_1)

    resi_ = ' '.join(resi_)
    resi = ['resi']
    resi.append(resi_)
 
    infor = []

    infor = id_ + name + dob_ + sex_ + nati_ + place + resi + date_

    return infor