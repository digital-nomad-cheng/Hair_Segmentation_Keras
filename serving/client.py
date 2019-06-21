import requests
import cv2
import numpy as np
import json
import os
import time


def get_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    resized_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = (img - np.mean(img)) / np.std(img)
    img = np.expand_dims(img, axis=0)
    return img, resized_img

imgs = [f for f in os.listdir('test') if not f.startswith('.')]
for _ in imgs:
    input_data, resized_img = get_img(os.path.join('test', _))
    data = {'inputs': {'input_image': input_data.tolist()}}
    t0 = time.time()
    r = requests.post("http://localhost:8501/v1/models/hair_seg:predict", json=data)
    t1 = time.time()
    print("time:", t1-t0)
    results = json.loads(r.content.decode('utf-8'))
    result_img = np.asarray(results['outputs'][0])*255
    result_img = np.clip(result_img, 0, 255).astype('uint8')
    count = 0
    pixel_sum = 0.
    for i in range(256):
        for j in range(256):
            if(result_img[i,j] > 250):
                pixel_sum += resized_img[i,j]
                count += 1
    print("pixel value:", pixel_sum/count)
    color_platte = np.zeros((256, 256, 3), dtype=np.uint8)
    color_platte[:,:,0] = int((pixel_sum/count)[0])
    color_platte[:,:,1] = int((pixel_sum/count)[1])
    color_platte[:,:,2] = int((pixel_sum/count)[2])
    cv2.imshow("mask", result_img)
    cv2.imshow("img", resized_img)
    cv2.imshow('color_platte', color_platte)
    cv2.waitKey(0)
