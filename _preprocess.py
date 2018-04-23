import json
import os
import numpy as np
import cv2



def preprocess_train(truncated = True):
	images = {}
	with open('imaterialist-challenge-furniture-2018/train.json') as json_data:
	    d = json.load(json_data)
	    labels = d['annotations']

	for l in labels:
		images[l['image_id']] = l['label_id']

	train_data = os.listdir("imaterialist-challenge-furniture-2018/train_data")
	features = []
	labels = []
	baseheight = 560
	for d in train_data:
		if d=='.DS_Store': continue
		idx = d.split('.')[0]
		img = cv2.imread("imaterialist-challenge-furniture-2018/train_data/"+d)
		if img is None: continue
		temp = cv2.resize(img,(227,227), interpolation=cv2.INTER_CUBIC)
		features.append(temp)
		labels.append(images[int(idx)])
	#	if len(features) % 2000 == 0:
	#		print("Finished processing {}".format(len(features)))
		if len(features) % 50000 == 0 and truncated is True:
			return np.array(features).astype(float) / 255.0 ,np.array(labels)
	return np.array(features).astype(float) / 255.0 , np.array(labels)

def preprocess_valid():
	images = {}
	with open('imaterialist-challenge-furniture-2018/validation.json') as json_data:
	    d = json.load(json_data)
	    labels = d['annotations']

	for l in labels:
		images[l['image_id']] = l['label_id']

	train_data = os.listdir("imaterialist-challenge-furniture-2018/valid_data")
	features = []
	labels = []
	baseheight = 560
	for d in train_data:
		if d=='.DS_Store': continue
		idx = d.split('.')[0]
		img = cv2.imread("imaterialist-challenge-furniture-2018/valid_data/"+d)
		if img is None: continue
		temp = cv2.resize(img,(227,227), interpolation=cv2.INTER_CUBIC)
		features.append(temp)
		#cv2.imshow('img',temp)
		#cv2.waitKey(0)
		labels.append(images[int(idx)])

	return np.array(features).astype(float) / 255.0, np.array(labels)
