import gdown
import zipfile
import os
from src import logger
import config.configure as config
print(os.getcwd())
os.environ["KAGGLE_CONFIG_DIR"] = "config/.kaggle/"
import kaggle
class DataIngestion():
    def __init__(self):
        pass
    def download(self):
        logger.info(f"Downloading dataset into data/")
        dataset_name = config.DATASET_NAME

        # Replace '/path/to/your/folder' with the path to the folder where you want to save the dataset
        output_folder = config.SAVE_DATA_PATH
        
        # Download the dataset to the specified folder
        kaggle.api.dataset_download_files(dataset_name, path=output_folder, unzip=True)
        logger.info(f"Download Completed")

if __name__ == '__main__':
    data = DataIngestion()