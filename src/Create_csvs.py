"""
This script is used to create the csvs with 'paths,label' rows. 
Create file s that diferentiate Train, Validation and test sets.
"""


import os
import pandas as pd
import random

PATH_TO_CLF_DIRS = "data\data_classification"
CLASSES = os.listdir(PATH_TO_CLF_DIRS)
RATIO_TRAIN = 0.70
RATIO_VAL = 0.25  # 30% of the original list
RATIO_TEST = 0.5 

filtered_classes = ['Apple_black_rot', 
                    'Apple_healthy', 
                    'Apple_rust', 
                    'Apple_scab',
                    'Blueberry_healthy', 
                    'Cherry_healthy', 
                    'Cherry_powdery_mildew',
                    'Corn_cercospora_leaf_spot',
                    'Corn_common_rust',
                    'Corn_healthy',
                    'Corn_northern_leaf_blight',
                    'Gauva_diseased',
                    'Gauva_healthy',
                    'Grape_black_rot',
                    'Grape_esca',
                    'Grape_healthy',
                    'Grape_leaf_blight',
                    'Lemon_diseased',
                    'Lemon_healthy',
                    'Mango_diseased',
                    'Mango_healthy',
                    'Orange_haunglongbing',
                    'Peach_bacterial_spot',
                    'Peach_healthy',
                    'Pepper_bell_bacterial_spot',
                    'Pepper_bell_healthy',
                    'Pomegranate_diseased',
                    'Pomegranate_healthy',
                    'Potato_early_blight',
                    'Potato_healthy',
                    'Potato_late_blight',
                    'Raspberry_healthy',
                    'Strawberry_healthy',
                    'Strawberry_leaf_scorch',
                    'Tomato_bacterial_spot',
                    'Tomato_early_blight',
                    'Tomato_healthy',
                    'Tomato_late_blight',
                    'Tomato_leaf_Mold',
                    'Tomato_mosaic_virus',
                    'Tomato_septoria_leaf_spot',
                    'Tomato_spider_mites',
                    'Tomato_target_Spot',
                    'Tomato_yellow_leaf_curl_virus']


current_class = 0
train_imgs  =[]
train_labels  =[]

val_imgs  =[]
val_labels  =[]

test_imgs  =[]
test_labels  =[]


for class_ in filtered_classes:
    images = os.listdir(os.path.join(PATH_TO_CLF_DIRS, class_))
    random.shuffle(images)

    lenTrain = int(len(images) * RATIO_TRAIN)
    lenVal = int(len(images) * RATIO_VAL)
    lenTest = len(images) - lenTrain - lenVal

    train_ = images[:lenTrain]
    val_ = images[lenTrain:lenTrain+lenVal]
    test_ = images[lenTrain+lenVal:]

    for img in train_:
        train_imgs.append(os.path.join(PATH_TO_CLF_DIRS, class_, img))
        train_labels.append(current_class)

    for img in val_:
        val_imgs.append(os.path.join(PATH_TO_CLF_DIRS, class_, img))
        val_labels.append(current_class)

    for img in test_:
        test_imgs.append(os.path.join(PATH_TO_CLF_DIRS, class_, img))
        test_labels.append(current_class)

    current_class+=1

TrainCsv = pd.DataFrame({'image_path': train_imgs, 'labels': train_labels})
TrainCsv.to_csv("TrainClf_Filtered.csv", index=False)    

ValCsv = pd.DataFrame({'image_path': val_imgs, 'labels': val_labels})
ValCsv.to_csv("ValClf_Filtered.csv", index=False)   

TestCsv = pd.DataFrame({'image_path': test_imgs, 'labels': test_labels})
TestCsv.to_csv("TestClf_Filtered.csv", index=False)   
