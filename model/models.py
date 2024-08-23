import torch
from torchvision import models
import torch.nn as nn


    #TODO
    #Dosya ismi degisecek ===> models.py CHECK
    #DISARDAN GELCEK INPUT ILE MODEL SECIMI CHECK
    #Resnet34,resnet50,densenet161.....etc CHECK
    #512 must be dynnamic CHECK

class Resnet18(torch.nn.Module):
    def __init__(self, dropout, name):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.name = name
        self.dropout = dropout
        
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, int(num_features/8)),
        nn.Dropout(dropout),
        nn.Linear(int(num_features/8), 1) 
        )

    def forward(self, x): 
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
    
class Resnet34(torch.nn.Module):
    def __init__(self, dropout, name):
        super(Resnet34, self).__init__()
        self.model = models.resnet34(pretrained = True)
        self.dropout = dropout
        self.name = name
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, int(num_features/8)),
        nn.Dropout(dropout),
        nn.Linear(int(num_features/8), 1) 
        )

    def forward(self, x): 
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
    
class Resnet50(torch.nn.Module):
    def __init__(self, dropout, name):
        super(Resnet50, self,).__init__()
        self.model = models.resnet50(pretrained = True)
        self.dropout = dropout
        self.name = name
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, int(num_features/8)),
        nn.Dropout(dropout),
        nn.Linear(int(num_features/8), 1) 
        )

    def forward(self, x): 
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
    
class Resnet101(torch.nn.Module):
    def __init__(self, dropout, name):
        super(Resnet101, self).__init__()
        self.model = models.resnet101(pretrained = True)
        self.dropout = dropout
        self.name = name
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, int(num_features/8)),
        nn.Dropout(dropout),
        nn.Linear(int(num_features/8), 1) 
        )

    def forward(self, x): 
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
    
    
class Densenet121(torch.nn.Module):
    def __init__(self, dropout, name):
        super(Densenet121, self).__init__()
        self.model = models.densenet121(pretrained = True)
        self.dropout = dropout
        self.name = name
        num_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, int(num_features/8)),
        nn.Dropout(dropout),
        nn.Linear(int(num_features/8), 1) 
        )

    def forward(self, x): 
        x = self.model(x)
        x = torch.sigmoid(x)
        return x    
    
def selectModel(model_name="Resnet18", dropout=0.2):

    if model_name == "Resnet18":
        return Resnet18(dropout = dropout, name=model_name)
    
    elif model_name == "Resnet34":
        return Resnet34(dropout = dropout, name=model_name)
    
    elif model_name == "Resnet50":
        return Resnet50(dropout = dropout, name=model_name)
    
    elif model_name == "Resnet101":
        return Resnet101(dropout = dropout, name=model_name)
    
    elif model_name == "Densenet121":
        return Densenet121(dropout = dropout, name=model_name)
    else:
        print(f"Model {model_name} is not supported. Resnet-18 returned")
        return Resnet18(dropout = dropout, name=model_name)