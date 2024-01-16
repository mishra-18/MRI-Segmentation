import torch
from tqdm import tqdm
import torch.nn as nn
from src import logger
from typing import *
import warnings
warnings.filterwarnings('ignore')
def model_fit(
    epochs: int,
    model: nn.Module,
    device: Any,
    train_loader: Any,
    valid_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    PATH: str
):
    """

     Args:
         epochs (int): # of epochs
         model (nn.Module): model for training
         device (Union[int, str]): number or name of device
         train_loader (Any): pytorch loader for trainset
         valid_loader (Any): pytorch loader for testset
         criterion (nn.Module): loss critiria
         optimizer (nn.Module): optimizer for model training
         path (str): path for saving model


    Example:
    >>>     train(
    >>>     epochs=25,
    >>>     model=model,
    >>>     device=0/'cuda'/'cpu',
    >>>     train_loader=train_loader,
    >>>     valid_loader=valid_loader,
    >>>     criterion=fn_loss,
    >>>     optimizer=optimizer)
    """


    best_DICESCORE = 0
    model.to(device)
    summary = {
        'train_loss' : [],
        'train_dice' : [],
        'valid_loss' : [],
        'valid_dice' : []
    }
    for epoch in range(epochs):
        logger.info(f"EPOCH {epoch}/{epochs}")
        train_Loss = 0
        train_Dicescore = 0
        model.train()
        for x, y in tqdm(train_loader):
            x = x.type(torch.FloatTensor).to(device)
            y = y.type(torch.FloatTensor).to(device)
            
            predict = model(x)
            loss =  criterion(predict, y)
            train_Loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predict = torch.sigmoid(predict)
            predict = (predict > 0.5).float() 
            
            dice_score = (2 * (y*predict).sum() + 1e-8)/((y+predict).sum() + 1e-8)

            try:
                train_Dicescore += dice_score.cpu().item()
            except:
                train_Dicescore += dice_score
            
        train_Loss /= len(train_loader)
        train_Dicescore /= len(train_loader)
    


        with torch.no_grad():
            val_Loss = 0
            val_Dicescore = 0
            model.eval()   
            for x, y in tqdm(valid_loader):
                x = x.type(torch.FloatTensor).to(device)
                y = y.type(torch.FloatTensor).to(device)

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


        logger.info(f"Loss: {train_Loss}  - Dice Score: {train_Dicescore} - Validation Loss: {val_Loss} - Validation Dice Score: {val_Dicescore}")

        if val_Dicescore > best_DICESCORE:
            best_DICESCORE  = val_Dicescore
            torch.save(model, PATH)

        summary['train_loss'] = train_Loss
        summary['train_dice'] = train_Dicescore
        summary['valid_loss'] = val_Loss 
        summary['valid_dice'] = val_Dicescore


    return summary