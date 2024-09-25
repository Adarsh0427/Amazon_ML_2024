import os
import random
import pandas as pd
# import sklearn
import matplotlib.pyplot as plt
import numpy as np
import math
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
import sys
from skimage.filters import threshold_sauvola
import cv2
# import pytesseract
# from PIL import Image
# from scipy.spatial import distance as dist
# from imutils import perspective
# from imutils import contours
import imutils
import easyocr
# from src.utils import download_image
# from src.sanity import sanity_check
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from keras.utils import to_categorical
from src.constants import entity_unit_map
import re



# def predictor(image_link, category_id, entity_name):
#     parsed_text, features = image_text_ocr(image_link, category_id, entity_name)
#     def create_model(input_shape, num_classes):
#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dense(num_classes, activation='softmax'))
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model

#     # Assuming features is a list of feature vectors and labels is a list of corresponding labels
#     features = np.array(features)
#     labels = np.array([category_id] * len(features))  # Assuming category_id is the label for all features

#     # Convert labels to categorical
#     num_classes = len(set(labels))
#     labels = to_categorical(labels, num_classes)

#     # Create and train the model
#     input_shape = (features.shape[1], features.shape[2], features.shape[3])  # Adjust based on your feature shape
#     model = create_model(input_shape, num_classes)
#     model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)

#     # Predict the category for the given image
#     prediction = model.predict(features)
#     predicted_category = np.argmax(prediction, axis=1)
#     return predicted_category
    
    
    
# def image_text_ocr(image_link, category_id, entity_name):
#     image_name = image_link.split('/')[-1]
#     image_path_3 = os.path.join(IMAGE_FOLDER,'/', image_name)

#     img = cv2.imread(image_path_3)

#     # instance text detector
#     reader = easyocr.Reader(['en'], gpu=True)

#     # detect text on image
#     text_ = reader.readtext(img, detail=1, paragraph=False)

#     threshold = 0.35
    
#     res=[]
#     fetures_list = pd.DataFrame()
#     # draw bbox and text
#     for t_, t in enumerate(text_):
        
#         bbox, text, score = t
        
#         if score > threshold:
#             cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
#             selected_region = img[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]]
#             feature_1d = selected_region.flatten()
#             feature_series = pd.Series(feature_1d)
#             feature_list = feature_series.tolist()
#             return res.append(text)
        
#     return res, fetures_list
            

#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.show()
    
    
# if __name__ == "__main__":
#     DATASET_FOLDER = 'dataset/'
#     IMAGE_FOLDER =  'images/'
    
#     s_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
#     # for i in s_test['image_link']:
#     #     print(i)
#     #     download_image(i, os.path.join(IMAGE_FOLDER, 's_test/'))
    
#     s_test['prediction'] = s_test.apply(
#         lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    
#     output_filename = os.path.join(DATASET_FOLDER, 's_test_out.csv')
#     s_test[['index', 'prediction']].to_csv(output_filename, index=False)
    
#     sanity_check(os.path.join(DATASET_FOLDER, 'test.csv'), os.path.join(DATASET_FOLDER, 'test_out.csv'))
  
  

    
# def prediction(link,entity_name,reader,DATASET_FOLDER):
#     if (entity_name != "item_weight"):
#         return ""

#     img_file_name =  link.split('/')[-1]
#     image_path = os.path.join(DATASET_FOLDER, img_file_name)

#     img = cv2.imread(image_path)

#         # instance text detecto
#     par_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # detect text on image
#     try:
#         text_ = reader.readtext(par_img, detail=0,low_text=0.1,text_threshold=0.5 )
#     except Exception as e:
#         print(f"Error reading text from image: {e}")
#         text_ = []
        
#     res=""

#     unit_list = entity_unit_map[entity_name]
#     unit_list = unit_list.union({'g' , 'kg' , 'lbs', 'oz', 'gal'})
    
#     fetures_list = pd.DataFrame()
#         # draw bbox and text
#     for t_, t in enumerate(text_):
#         text = t.lower()
#         if 'g' in text or 'gram' in text or 'gm' in text:
#             numeric = re.findall(r'\d+\.?\d*', text)
#             if len(numeric) == 0:
#                 continue
#             ad_val = numeric[0] + " " + 'gram'
#             res = ad_val
#             break
#         if 'kg' in text or 'kilogram' in text:
#             numeric = re.findall(r'\d+\.?\d*', text)
#             if len(numeric) == 0:
#                 continue
#             ad_val = numeric[0] + " " + 'kilogram'
#             res = ad_val
#             break
#         if 'lbs' in text or 'pound' in text:
#             numeric = re.findall(r'\d+\.?\d*', text)
#             if len(numeric) == 0:
#                 continue
#             ad_val = numeric[0] + " " + 'pound'
#             res = ad_val
#             break
#         if 'oz' in text or 'ounce' in text:
#             numeric = re.findall(r'\d+\.?\d*', text)
#             if len(numeric) == 0:
#                 continue
#             ad_val = numeric[0] + " " + 'ounce'
#             res = ad_val
#             break
#         if 'gal' in text or 'gallon' in text:
#             numeric = re.findall(r'\d+\.?\d*', text)
#             if len(numeric) == 0:
#                 continue
#             ad_val = numeric[0] + " " + 'gallon'
#             res = ad_val
#             break
#     return res
# image_path

IMAGE_FOLDER = 'images/'
DATASET_FOLDER = 'dataset/'
test_data = pd.read_csv('dataset/test.csv')
print(test_data[test_data['entity_name'] == "item_weight"].count())
reader = easyocr.Reader(['en'], gpu=True)
image_path = ''
img = cv2.imread(image_path_3)

        # instance text detecto
par_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)