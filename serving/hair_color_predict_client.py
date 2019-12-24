import os
import time
import json

import cv2
import requests
import numpy as np

def get_input_and_resized_img(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    resized_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = (img - np.mean(img)) / np.std(img)
    img = np.expand_dims(img, axis=0)
    return img, resized_img

def get_pixel_value(pixel):
    return pixel[0] + pixel[1]*256 + pixel[2]*256*256

def get_rgb_values(pixel_value):
    pixel_1 = pixel_value % 256
    pixel_value = pixel_value // 256
    pixel_2 = pixel_value % 256
    pixel_value = pixel_value // 256
    pixel_3 = pixel_value % 256
    return pixel_1, pixel_2, pixel_3

def find_most_common_pixel(img):
    histogram = {}
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            pixel_val = get_pixel_value(img[i, j, :])
            if pixel_val in histogram:
                histogram[pixel_val] += 1
            else:
                histogram[pixel_val] = 1
    most_common_pixel_val = max(histogram, key=histogram.get)
    return get_rgb_values(most_common_pixel_val)

def get_pixel_val_img(img):
    h, w, c = img.shape
    pixel_val_img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            pixel_val_img[i, j] = get_pixel_value(img[i, j, :])
    return pixel_val_img

def cal_similar_color(resized_img, result_img):
    '''calculate the most common color using three channels histogram'''
    result_img[np.where(result_img > 250)] = 255
    result_img[np.where(result_img <= 250)] = 0
    pixel_val_img = get_pixel_val_img(resized_img)
    # only take into account where pixle value in result_img is 255
    hist = cv2.calcHist([pixel_val_img.astype('float32')], [0], result_img, [16777216], [0, 16777216])
    most_common_color = get_rgb_values(np.argmax(hist))
    return most_common_color

if __name__ == "__main__":
    data_path = '/Users/vincent/Documents/Dataset/test_imgs/'
    imgs = sorted([f for f in os.listdir(data_path) if not f.startswith('.')], reverse=True)
    for _ in imgs:
        img = cv2.imread(os.path.join(data_path, _), 1)
        h, w, c = img.shape
        print("img name:", _)
        input_data, resized_img = get_input_and_resized_img(os.path.join(data_path, _))
        data = {'inputs': {'input_image': input_data.tolist()}}
        t0 = time.time()
        r = requests.post("http://localhost:8500/v1/models/skin_seg:predict", json=data)
        t1 = time.time()
        print("time:", t1-t0)
        print(r)
        results = json.loads(r.content.decode('utf-8'))
        results = np.asarray(results['outputs'][0])
        result_img = np.clip(results*255, 0, 255).astype('uint8')
        
        most_common_color = cal_similar_color(resized_img, result_img)
        common_color_plate = np.zeros((224, 224, 3), dtype=np.uint8)
        common_color_plate[:, :, 0] = most_common_color[0]
        common_color_plate[:, :, 1] = most_common_color[1]
        common_color_plate[:, :, 2] = most_common_color[2]
        
        mask = cv2.resize(results, (w, h))
        mask = np.clip(mask*255, 0, 255).astype('uint8')
        # cv2.imwrite('mask_'+_, mask)
        cv2.imshow("img", img)
        cv2.imshow("color_plate_common", common_color_plate)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)

# docker run -it -p 8500:8501 -v "$(pwd)/serving/skin_seg/:/models/skin_seg" -e MODEL_NAME=skin_seg tensorflow/serving
