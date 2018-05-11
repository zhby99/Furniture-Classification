from _preprocess import *

def valid_reprocess():
	images = {}
        path = 'dataset/val/'
        with open('imaterialist-challenge-furniture-2018/validation.json') as json_data:
            d = json.load(json_data)
            labels = d['annotations']

        for l in labels:
            images[l['image_id']] = l['label_id']

        train_data = os.listdir("imaterialist-challenge-furniture-2018/valid__data")
        for d in train_data:
            if d=='.DS_Store': continue
            idx = d.split('.')[0]
            img = cv2.imread("imaterialist-challenge-furniture-2018/valid_data/"+d)
            if img is None:continue
            if os.path.isdir(path + str(images[int(idx)])) is False: os.makedirs(path + str(images[int(idx)]), mode=0o777)
            cv2.imwrite(path + str(images[int(idx)]) + '/' + d, img)

def test_reprocess():
		images = {}
        path = 'dataset/test/'
        with open('imaterialist-challenge-furniture-2018/test.json') as json_data:
            d = json.load(json_data)
            labels = d['annotations']

        for l in labels:
            images[l['image_id']] = l['label_id']

        train_data = os.listdir("imaterialist-challenge-furniture-2018/test_data")
        for d in train_data:
            if d=='.DS_Store': continue
            idx = d.split('.')[0]
            img = cv2.imread("imaterialist-challenge-furniture-2018/test_data/"+d)
            if img is None:continue
            if os.path.isdir(path + str(images[int(idx)])) is False: os.makedirs(path + str(images[int(idx)]), mode=0o777)
            cv2.imwrite(path + str(images[int(idx)]) + '/' + d, img)

def train_repreprocess():
        images = {}
        path = 'dataset/train/'
        with open('imaterialist-challenge-furniture-2018/train.json') as json_data:
            d = json.load(json_data)
            labels = d['annotations']

        for l in labels:
            images[l['image_id']] = l['label_id']

        train_data = os.listdir("imaterialist-challenge-furniture-2018/train_data")
        for d in train_data:
            if d=='.DS_Store': continue
            idx = d.split('.')[0]
            img = cv2.imread("imaterialist-challenge-furniture-2018/train_data/"+d)
            if img is None:continue
            if os.path.isdir(path + str(images[int(idx)])) is False: os.makedirs(path + str(images[int(idx)]), mode=0o777)
            cv2.imwrite(path + str(images[int(idx)]) + '/' + d, img)

train_repreprocess()
