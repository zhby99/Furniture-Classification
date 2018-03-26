import json
import os
import numpy as np

images = {}
with open('imaterialist-challenge-furniture-2018/train.json') as json_data:
    d = json.load(json_data)
    labels = d['annotations']

for l in labels:
	images[l['image_id']] = l['label_id']

#print(images)

train_data = os.listdir("im/train_data")
print(train_data)
features = []
labels = []
for d in train_data:
	if d=='.DS_Store': continue
	idx = d.split('.')[0]
	features.append(np.array(d))

print(features)