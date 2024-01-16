import torch
import torch.nn as nn
from tqdm import tqdm
from typing import *
from src import logger

def predict_mask(
        data: Any,
        device: Any,
        model: nn.Module,
        inference: bool,
        valid_loader=None,
        criterion=None,
):
    """
    predicts mask for the image
    Args:
        data (Any): image data for predicting
        model (nn.Module): model for training
        device (0/'cud'/'cpu'/Any): name of device
        inference (bool): Whether to evaluate or predict
        valid_Loader (nn.Module): test loader for training
        criterion (nn.Module): loss criteria

    Example:
    >>>     train(
    >>>     data = torch.FloatTensor,
    >>>     model=model,
    >>>     device=0/'cuda'/'cpu'
    >>>     ingerence=0
    >>>     valid_loader= test_loader
    >>>     criterion= fn_loss
    """

    if inference:

        with torch.no_grad():
            image = data.type(torch.FloatTensor).to(device)
            model = model.to(device)
            pred = model(image)
            pred = torch.sigmoid(pred)
            mask = (pred > 0.6).float()
            
            return mask.cpu().detach()
    else:
        with torch.no_grad():
            val_Loss = 0
            val_Dicescore = 0
            model.eval()   
            for x, y in tqdm(valid_loader):
                x = x.type(torch.cuda.FloatTensor).to(device)
                y = y.type(torch.cuda.FloatTensor).to(device)

                predict = model(x)
                loss = criterion(predict, y)
                val_Loss += loss.item()
                
                predict = torch.sigmoid(predict)
                predict = (predict > 0.5).float() 

                dice_score = (2 * (y*predict).sum() + 1e-8)/((y+predict).sum() + 1e-8)
                try:
                    val_Dicescore += dice_score.cpu().item()
                except:
                    val_Dicescore += dice_score

            val_Loss /= len(valid_loader)
            val_Dicescore /= len(valid_loader)

        logger.info(f"Test Loss: {val_Loss}  - Dice Score: {val_Dicescore}")