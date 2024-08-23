import os
import torch
from datetime import date
from utils.utils import createFolder

def save_model(model, save_dir:str, name:str=None, best:bool = True) -> None:   
    if best == False:
        today = str(date.today().strftime("%Y-%m-%d"))
        save_dir = os.path.join(save_dir, today)
        createFolder(save_dir)

    dir = os.path.join(save_dir, name)
    torch.save(model, dir)

def load_model(name:str, model_path:str):
    full_name = f'{name}-best_model.tch'
    dir = os.path.join(model_path, full_name)
    model = torch.load(dir)
    return model