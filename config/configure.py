import os
"""
If you are planing to train the model. The data_ingestion stage requires
a kaggle API key and your user name. That is why you need to have 
.kaggle/ folder under config.
config-|
       |-.kaggle-|         
                 |-kaggle.json
       |-configure.py      
Make sure you replace your username under kaggle.json. 
"""
mask_images_path = os.getcwd() + "/data/lgg-mri-segmentation/kaggle_3m/*/*_mask.tif"
SAVE_MODEL_PATH = os.getcwd() + "/src/model/best_model.pth" 
SAVE_DATA_PATH = os.getcwd() + "/data"
DATASET_NAME = "mateuszbuda/lgg-mri-segmentation"