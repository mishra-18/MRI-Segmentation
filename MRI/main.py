import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchsummary
import torchview
import config.configure as config
from src import logger
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocess import data_loaders
from src.pipelines.training import model_fit
from src.model.unet import UNet
## graphviiz
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>>>>  Starting {STAGE_NAME}  <<<<<<<<")
   data_ingestion = DataIngestion()
   data_ingestion.download()
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = 'Training'
BATCH_SIZE = 32
NUM_WORKERS = 3
EPOCHS = 50
PATH = config.SAVE_MODEL_PATH

try:
    logger.info(f'Preparing DataLoders')

    # getting the dataloaders
    train_loader, valid_loader = data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, train_split=True)

    # fitting the model
    loss_fn = nn.BCEWithLogitsLoss()
    in_channels = 3
    out_channels = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = [64, 128, 256, 512]
    model = UNet(in_channels=in_channels, out_channels=out_channels, features=features)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    

    # starting the training stage
    logger.info(f"Strating {STAGE_NAME} Stage \n\n ==============")

    summary = model_fit(
            epochs=EPOCHS,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            PATH=PATH
        )
    
    
except Exception as e:
    logger.exception(e)
    raise e