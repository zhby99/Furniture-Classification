import json
import os
import numpy as np
from scipy import misc
import cv2

def preprocess():
	images = {}
	with open('imaterialist-challenge-furniture-2018/train.json') as json_data:
	    d = json.load(json_data)
	    labels = d['annotations']

	for l in labels:
		images[l['image_id']] = l['label_id']

	train_data = os.listdir("im/train_data")
	features = []
	labels = []
	baseheight = 560
	for d in train_data:
		if d=='.DS_Store': continue
		idx = d.split('.')[0]
		img = cv2.imread("im/train_data/"+d)
		temp = cv2.resize(img,(227,227), interpolation=cv2.INTER_CUBIC)
		features.append(temp)
		#cv2.imshow('img',temp)
		#cv2.waitKey(0)
		labels.append(images[int(idx)])

	return features, labels


def test_reprocess():
        path = 'dataset/test/2/'
        data = os.listdir("dataset/test/1")
        for d in data:
            if d=='.DS_Store': continue
            idx = d.split('.')[0]
            img = cv2.imread("dataset/test/1/"+d)
            if img is None:continue
            cv2.imwrite(path + d, img)

test_reprocess()
