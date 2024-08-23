import sys
sys.dont_write_bytecode = True

import os
import torch
from torch.utils.data import DataLoader
from utils.model_utils import load_model
from utils.metrics import metrics
from dataset.dataset import CatDogDataset
from utils.utils import tensor2image
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
dataset_path = 'dataset/cat_dog'
model_dir = 'results/models'
model_arc_name = 'Densenet121'
batch_size = 32
num_workers = 4


class evaluation():
    def __init__(self, model_dir, test_loader, name = "Resnet18") -> None:
        self.model_name = name
        self.model = load_model(name, model_dir)
        self.test_loader = [(images.to(device), labels.to(device).view(-1, 1)) for images, labels in test_loader]
        self.tp = 0
        self.tn = 0
        self.fp = 0 
        self.fn = 0

    def evaluate(self):

        self.model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images, labels
                outputs = self.model(images)
                preds = outputs > 0.5
                self.save_results(preds, labels)

        return self.tp, self.tn, self.fp, self.fn
    

    def save_results(self, preds:torch.tensor, labels:torch.tensor) -> None:
        true = preds == labels
        self.tp += torch.sum(true[labels == 1]).item()
        self.tn += torch.sum(true[labels == 0]).item()
        false = preds != labels
        
        # for images, labels in self.test_loader:
        #     for j in range(batch_size):
        #         if false[j].item() == 1:
        #             tensor2image(images[j, :, : ,:])

        self.fp += torch.sum(false[labels == 1]).item()
        self.fn += torch.sum(false[labels == 0]).item()



# def visualize_predictions(model, dataset, num_images=6):

#     model.eval()

#     model.to(device)
    
#     fig, axes = plt.subplots(2, 3, figsize=(12, 12))
#     axes = axes.flatten()
    
#     sampled_indices = random.sample(range(len(dataset)), num_images)
    
#     for ax, idx in zip(axes, sampled_indices):
#         image, true_label = dataset[idx]
#         image = image.to(device).unsqueeze(0)
        
#         with torch.no_grad():
#             output = model(image)
        
#         pred_label = (output > 0.5).item()
        
#         image = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
#         image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#         image = np.clip(image, 0, 1)
        
#         ax.imshow(image)
#         ax.axis('off')
#         ax.set_title(f'True: {"Cat" if true_label == 1 else "Dog"}\nPred: {"Cat" if pred_label == 1 else "Dog"}')
        
#     plt.show()

if __name__ == "__main__":
    test = CatDogDataset(image_dir=dataset_path, sub='test')
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers)

    tester = evaluation(model_dir, test_loader, name=model_arc_name)

    results = tester.evaluate()
    metric = metrics(results)
    metric.quick()

