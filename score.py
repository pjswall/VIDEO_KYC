from deepface import DeepFace
import numpy as np
import cv2
import glob
import os
import pandas as pd
from tqdm import tqdm

# original_path = './data/Color/_cnsifd_s_'
# aging_path = './data/AGE_DATA/experiments/inference_results/30/_cnsifd_s_'

# Color_imgs = glob.glob(original_path + '*.jpg')
# aging_imgs = glob.glob(aging_path + '*.jpg')
# models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","SFace"]

# for i in Color_imgs:
#     print(i)
#     path1 = i
#     path2 = i
#     for model in models:
#         print(DeepFace.verify(i, i, model))

folders = ['./data/AGE_DATA/experiments/inference_results/Color' ,'./data/AGE_DATA/experiments/inference_results/30']

data = {
    'Img_Name' : [],
    'Age_Image'  : [],
    'model' : [],
    'verified' : [],
    'distance'  : [],
    'threshold' : []
}

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","SFace"]

def calculate_score(path1, path2, model):
    result = DeepFace.verify(path1, path2, model, enforce_detection = False)
    return result['verified'], result['distance'], result['threshold'], result['model']

count = 0

for folder in tqdm(folders) :

    for i in tqdm(glob.glob(folder + '/*.jpg')):
        # print(i)
        # print(i.split('/')[-1])
        if(count == 10):
            break
        for model in tqdm(models):
            verified, distance, threshold, model = calculate_score(i, i, model)
            data['Img_Name'].append(i)
            data['Age_Image'].append(i.split('/')[-1])
            data['model'].append(model)
            data['verified'].append(verified)
            data['distance'].append(distance)
            data['threshold'].append(threshold)

        count += 1
        

df = pd.DataFrame(data = data, columns = list(data.keys()))
print(df.head())
print(df.describe())
df.to_csv('score.csv', index = False)






    

