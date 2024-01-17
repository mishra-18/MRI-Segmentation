# Deep Learning-Based Semantic Segmentation for Tumor Detection in MRI Brain Scans

## Overview
Our model is trained on a diverse dataset encompassing various tumor types, sizes, and locations, capturing the inherent heterogeneity of brain tumors encountered in clinical practice. The semantic segmentation architecture enables accurate localization and differentiation of tumor regions from surrounding healthy brain tissues, providing a valuable tool for early detection and characterization of lesions.

## Motivation
The integration of this deep learning-based segmentation approach into clinical workflows holds great promise for advancing the field of neuro-oncology. By automating the tumor detection process, our methodology not only expedites diagnosis but also provides clinicians with a reliable tool for precise delineation and monitoring of brain tumors, contributing to improved patient outcomes and treatment planning.

```This Project is running live on 🤗 Hugging Face. You can run the app on your machine by pulling the Docker 🐋 Image from docker hub.```
```Tutorial Notebook is available on Kaggle```

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space%20-ff5a1f.svg)]([https://huggingface.co/models](https://huggingface.co/spaces/smishr-18/MRISegmentation/tree/main))
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset/or/Kernel-20BEFF.svg)](https://www.kaggle.com/your_username/dataset-name-or-kernel-name)

## Results
* Hyperparameters
```
# STAGE_NAME = 'Training'
# MODEL_NAME = 'UNet'
# feature_layers = [64, 128, 256, 512]
EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
```
**Model Architecture**

  
  ![download](https://github.com/mishra-18/MRI-Segmentation/assets/155224614/2a5035f1-64fc-4b06-a2bf-7225b7fb3545)

|SET     | MERICS      | Loss/DiceScore |
|--------| ----------- | -----------    |
|Training| Loss        |  0.0054        |
|Training| Dice Score  |  0.8824        |
|Valid   | Loss        |  0.006         |
|Valid   | Dice Score  |  0.898         |
|Test    | Loss        |  0.012         |
|Test    | Dice Score  |  0.87          |

**Note** These are the results for 50 EPOCH, on training the model For 100 EPOCHS the Val Dice Score reached to .91 but test dice score remained around ~ 0.90.


**Ploting Results**

![output](https://github.com/mishra-18/MRI-Segmentation/assets/155224614/e62424c4-da9c-433e-9e72-aed4f089edc1)

## Usage

* Clone the repository
```
https://github.com/mishra-18/MRI-Segmentation.git
cd MRI-Segmentation
```
## Docker

* login to your docker with docker hub
```
sudo docker login -u <user-name>
```
* Pull the docker image
```
sudo docker pull mishra45/mris:latest
```
* Run the streamlit app
```
sudo docker run -p 8080:8051 mishra45/mris:latest
```
## Training

****Please Follow the API configure instructions in config/configure.py****
```
python main.py
```

* The dataset used for training is [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).There's no need to download, the Data Ingestion Stage will do it for you. given you configured your kaggle username under config/.
* Data Ingestion Stage: Starts downloading data into data/, Preprocess the data and prepare the dataloaders.
* Training Stage: Starts Training the model
* After Training is finished the model weights will be stored under src/model/ for inference. The project is already deloyed on [huggingface space](https://huggingface.co/spaces/smishr-18/MRISegmentation/tree/main) and you can perform inference from there or else run ```streamlit run app.py``` after training.
