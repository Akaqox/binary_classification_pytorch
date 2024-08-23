import sys
sys.dont_write_bytecode = True
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from datetime import datetime
import gc
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset.dataset import CatDogDataset
from train.train import Train
# from utils.utils import tensor2image
from model.models import selectModel

learning_rate = 0.00001
batch_size = 16
dropout=0.2
EPOCH = 10
num_workers = 4

model_arch_name = "Resnet18"
hyper_p =[learning_rate, batch_size, dropout, model_arch_name]

dataset_path = 'dataset/cat_dog'

model= selectModel(model_arch_name)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('results/reports/catdog_trainer_{}'.format(timestamp))


if __name__ == "__main__":                                                     

    #Create dataset classes

    train = CatDogDataset(image_dir=dataset_path, sub='train')
    val = CatDogDataset(image_dir=dataset_path, sub='val')

    #Create Dataloaders
    trainLoader = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
    valLoader = DataLoader(val, batch_size=batch_size, num_workers=num_workers)

    #test images in batch

    # for images, labels in valLoader:
    #     plt.imshow(tensor2image(images[10, :, : ,:]))

    #memory allocation
    del train
    del val
    gc.collect


    # model_arch_name = ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet121']
    # for model_arch_name in model_arch_name:
    #     model= selectModel(model_arch_name)
    #     hyper_p =[learning_rate, batch_size, dropout, model_arch_name]
    #     criterion = torch.nn.BCELoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model = Train(trainLoader, valLoader, hyper_p, model=model, criterion=criterion, optimizer=optimizer, EPOCH=EPOCH)
    train_model.fit()
    train_model.plot()
    


