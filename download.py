import os
import requests
import json
import shutil

os.chdir('imaterialist-challenge-furniture-2018/')

train_data = json.load(open('train.json'))

images_arr = train_data['images']
annotations_arr = train_data['annotations']
os.makedirs('train_data')
for img in images_arr:
    if img['image_id'] % 1000 == 0:
        print("Finished {} out of 194,828".format(img['image_id']))
    try:
        r = requests.get(img['url'][0],timeout=10)
        folder_path = 'train_data/'
        with open(folder_path+str(img['image_id'])+'.jpg','wb') as f:
            f.write(r.content)
        f.close()

    except Exception as e:
        print(e)
