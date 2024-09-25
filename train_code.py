import os
import random
import pandas as pd
# import sklearn
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import sys
# from skimage.filters import threshold_sauvola
# import cv2
# import pytesseract
# from PIL import Image
# from scipy.spatial import distance as dist
# from imutils import perspective
# from imutils import contours
# import imutils
# import easyocr
# from src.utils import download_image
# from src.sanity import sanity_check
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from keras.utils import to_categorical
from src.sanity import sanity_check
import re
entity_unit_map = {
    'width': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'depth': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'height': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'item_weight': ['milligram',
        'kilogram',
        'microgram',
        'gram',
        'ounce',
        'pound',
        'ton'],
    'maximum_weight_recommendation': ['kilogram',
        'gram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'],
    'voltage': ['volt' ,'kilovolt', 'millivolt'],
    'wattage': [ 'watt','kilowatt'],
    'item_volume': ['millilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'centilitre',
        'pint',
        'quart']
}


def prediction (image_link, entity_name,pred):
    if pred != '  10 milligram':
        return pred
    output_path = 'rust/output/'+image_link[:-4]+'.txt'
    if not os.path.exists(output_path): 
        return pred
    txt_data = open(output_path, 'r')
    unit = entity_unit_map[entity_name][0]
    for line in txt_data:
        line = line.lower()
        if 'kg' in line:
            unit = 'kilogram'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                continue
            res = "".join(value[0]) + " " + unit
           
            return res
        if 'g' in line:
            unit = 'gram'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                continue
            if len(value) >1:
                res = "".join(value[-1]) + " " + unit
            
                return res 
            res = "".join(value[0]) + " " + unit
    
            return res
        if 'lbs' in line:
            unit = 'pound'
            
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                continue
            res = "".join(value[0]) + " " + unit
         
            return res
        
        
        if 'oz' in line:
            unit = 'ounce'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                    continue
            res = "".join(value[0]) + " " + unit
         
            return res
        if 'ton' in line:
            unit = 'ton'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                    continue
            res = "".join(value[0]) + " " + unit
    
            return res
        if 'mg' in line:
            unit = 'milligram'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                    continue
            res = "".join(value[0]) + " " + unit
        
            return res
        for unit in entity_unit_map[entity_name]:
            if unit in line:
                value = re.findall(r'[0-9]*\.?[0-9]+', line)
                if len(value) == 0:
                    continue
                res = "".join(value[0]) + " " + unit
      
                return res

    return ""


def prediction_volt(image_link, entity_name,pred):
    if pred != '  10 volt':
        return pred
    output_path = 'rust/output/'+image_link[:-4]+'.txt'
    if not os.path.exists(output_path) : 
        return pred
    txt_data = open(output_path, 'r')
    unit = entity_unit_map[entity_name][0]
    for line in txt_data:
        line = line.lower()
        if 'kv' in line:
            unit = 'kilovolt'
            value = re.findall(r'[0-9]+', line)
            if len(value) == 0:
                continue
            if len(value) >1:
                res = "[" + f"{value[-2]}, {value[-1]}" + "]" + unit
 
                return res
            res = "".join(value[0]) + " " + unit

            return res
        if 'mv' in line:
            unit = 'millivolt'
            lbs_index = line.find('mv')
            if lbs_index != -1:
                value_before_lbs = re.findall(r'[0-9]+', line[max(0,lbs_index-4):lbs_index])
                if value_before_lbs:
                    value = value_before_lbs
                    res = "".join(value[0]) + " " + unit
           
                    return res
            value = re.findall(r'[0-9]+', line)
            if len(value) == 0:
                continue
            res = "".join(value[0]) + " " + unit

            return res
        
        if 'v' in line:
            unit = 'volt'
            value = re.findall(r'[0-9]+', line)
            if len(value) == 0:
                    continue
            if len(value) >1:
                res = "[" + f"{value[-2]}, {value[-1]}" + "] " + unit
       
                return res
            res = "".join(value[0]) + " " + unit

            return res
        for unit in entity_unit_map[entity_name]:
            if unit in line:
                value = re.findall(r'[0-9]+', line)
                if len(value) == 0:
                    continue
                if len(value) >1:
                    res = "[" + f"{value[-2]}, {value[-1]}" + "]" + unit
        
                    return res
                res = "".join(value[0]) + " " + unit
  
                return res

    return ""

def prediction_wattage(image_link, entity_name,pred):
    if pred != '  10 watt':
        return pred
    output_path = 'rust/output_w/'+image_link[:-4]+'.txt'
    if not os.path.exists(output_path): 
        return pred
    txt_data = open(output_path, 'r')
    unit = entity_unit_map[entity_name][0]
    for line in txt_data:
        line = line.lower()
        if 'kw' in line:
            unit = 'kilowatt'
            ind_v = line.find('k')
            value = re.findall(r'[0-9]*\.?[0-9]+', line[0:ind_v])
            if len(value) == 0:
                continue
            if len(value) >1:
                res = "".join(value[-1]) + " " + unit

                return res
            res = "".join(value[0]) + " " + unit

            return res
        if 'w' in line:
            unit = 'watt'
            ind_v = line.find('w')
            value = re.findall(r'[0-9]*\.?[0-9]+', line[0:ind_v])
            if len(value) == 0:
                    continue
            if len(value) >1:
                res = "".join(value[-1]) + " " + unit

                return res
            res = "".join(value[0]) + " " + unit

            return res
        for unit in entity_unit_map[entity_name]:
            if unit in line:
                ind_v = line.find(unit[0])
                value = re.findall(r'[0-9]*\.?[0-9]+', line[0:ind_v])
                if len(value) == 0:
                    continue
                if len(value) >1:
                    res = "".join(value[-1]) + " " + unit

                    return res
                res = "".join(value[0]) + " " + unit

                return res

    return ""

def prediction_volume(image_link, entity_name,pred):
    if pred != '  10 millilitre':
        return pred
    output_path = 'rust/output_v/'+image_link[:-4]+'.txt'
    if not os.path.exists(output_path): 
        return pred
    txt_data = open(output_path, 'r')
    unit = entity_unit_map[entity_name][0]
    for line in txt_data:
        line = line.lower()
        if 'ml' in line:
            unit = 'millilitre'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                continue
            if len(value) >1:
                res = "".join(value[-1]) + " " + unit
           
                return res
            res = "".join(value[0]) + " " + unit
 
            return res
        if 'l' in line:
            unit = 'litre'
            value = re.findall(r'[0-9]*\.?[0-9]+', line)
            if len(value) == 0:
                    continue
            if len(value) >1:
                res = "".join(value[-1]) + " " + unit
      
                return res
            res = "".join(value[0]) + " " + unit

            return res
        for unit in entity_unit_map[entity_name]:
            if unit in line:
                value = re.findall(r'[0-9]*\.?[0-9]+', line)
                if len(value) == 0:
                    continue
                if len(value) >1:
                    res = "".join(value[-1]) + " " + unit
   
                    return res
                res = "".join(value[0]) + " " + unit
 
                return res

    return ""




# print(entity_unit_map['height'][0])
IMAGE_FOLDER = 'images/'
DATASET_FOLDER = 'dataset/'
train_data = pd.read_csv('dataset/train.csv')
un_entity_name = train_data['entity_name'].unique()
# ['item_weight' 'item_volume' 'voltage' 'wattage'
#  'maximum_weight_recommendation' 'height' 'depth' 'width']
# train_gp = train_data['group_id'].unique()
# print ("un ", len(train_gp))
# train_link = train_data['image_link'].unique()
# print ("link : " ,len(train_link))
# data_gid = {}
# for e in un_entity_name:
#     data_gid[e] = train_data[train_data['entity_name'] == e]['group_id'].unique()


# work_data = pd.read_csv('dataset/work.csv')
# work_data['image_link'] = work_data['image_link'].apply(lambda x: x.split('/')[-1])
# print("110EibNyclL.jpg" in set(work_data['image_link']))


# test_data = pd.read_csv('dataset/test.csv')
# print(test_data['entity_name'].value_counts())
# item_weight_links = test_data[(test_data['entity_name'] == 'item_volume')]['image_link'].unique()
# item_weight_links = pd.DataFrame(item_weight_links, columns=['image_link'])
# output_filename_link = os.path.join(DATASET_FOLDER, 'test_link_volume.csv')
# item_weight_links['image_link'].to_csv(output_filename_link, index=False)



# # group_data = test_data['group_id'].unique()
# # group_data_in_train = [g for g in group_data if g in train_gp]
# # print("Number of group_id in train_gp:", len(group_data_in_train))


# # Calculate the frequency of each image_link
# link_frequency = test_data['image_link'].value_counts()

# # Map the frequency to the test_data
# test_data['link_frequency'] = test_data['image_link'].map(link_frequency)

# # Sort test_data based on the frequency of image_link
# test_data_sorted = test_data.sort_values(by='link_frequency', ascending=False)

# # Drop the temporary 'link_frequency' column
# test_data_sorted = test_data_sorted.drop(columns=['link_frequency'])




# print("Number of image_link in test_data:", len(link_data))
# # use hashing to find the common image_link
# train_link_set = set(train_link)
# common_links = [link for link in link_data if link in train_link_set]
# print("Number of common image_link:", len(common_links))
# link_data_not_in_train = [l for l in link_data if l in train_link]
# print("Number of image_link not in train_link:", len(link_data_not_in_train))
# print (len(group_data))
# group_link = {}
# for g in group_data:
#     print("len  " ,len(test_data[test_data['group_id'] == g]))
#     group_link[g] = test_data[test_data['group_id'] == g]['image_link'].unique()
#     print(g, "  " , len(group_link[g]))
#     print()
# test_data_sorted['prediction'] = test_data_sorted.apply(lambda x: prediction(x['image_link'].split('/')[-1], x['group_id'], x['entity_name'], work_data), axis=1)

# print(test_data_sorted.head(2))

# output_filename = os.path.join(DATASET_FOLDER, 'test_out_2.csv')
# test_data_sorted[['index', 'image_link' , 'prediction', 'entity_name']].to_csv(output_filename, index=False)



modify_data = pd.read_csv('dataset/test_out_3.csv')
modify_data['prediction'] = modify_data.apply(lambda x: prediction(x['image_link'].split('/')[-1], x['entity_name'], x['prediction']), axis=1)
modify_data['prediction'] = modify_data.apply(lambda x: prediction_wattage(x['image_link'].split('/')[-1], x['entity_name'], x['prediction']), axis=1)
modify_data['prediction'] = modify_data.apply(lambda x: prediction_volume(x['image_link'].split('/')[-1], x['entity_name'], x['prediction']), axis=1)
for i in range(len(modify_data)):
    prediction = str(modify_data.loc[i, 'prediction'])
    if prediction == '':
        continue
    if prediction[:2] == '  ':
        modify_data.loc[i, 'prediction'] = prediction[2:]



output_filename_1 = os.path.join(DATASET_FOLDER, 'test_out_8.csv')
modify_data[['index','image_link' ,'prediction']].to_csv(output_filename_1, index=False)

# sanity_check('dataset/test.csv', output_filename_1)