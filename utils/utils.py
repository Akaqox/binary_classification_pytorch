import numpy as np
import os
import torch
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, tolerance=2, min_delta=0.10):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        #self.model = model.to(DEVICE)

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def createFolder(dir:str) -> None:
    if(not os.path.isdir(dir)):
        os.makedirs(dir)

def tensor2image(tensor) -> np.ndarray:
    tensor = torch.unsqueeze(tensor.cpu(), 0)
    tensor = tensor.numpy()[0]
    tensor = np.transpose(tensor, (1, 2, 0))
    image= tensor/np.amax(tensor)
    plt.imshow(image)
    plt.show()
    print(type(image))
    return image

def train_report(save_dir, unique_name, hyper_p, train_p):

    report_filename = f'{unique_name[:8]}_classification-report.txt'
    with open(os.path.join(save_dir, report_filename), 'a') as report_file:
        report_file.write("\n\n-------- New Train --------\n")
        report_file.write(f"Model name is: {unique_name}\n")
        report_file.write(f"Epoch number: {train_p[1]}\n")
        report_file.write(f"Best validation loss: {train_p[1]}\n")
        report_file.write(f"Learning Rate: {hyper_p[0]}\n")
        report_file.write(f"Batch_size: {hyper_p[1]}\n")
        report_file.write(f"Dropout: {hyper_p[2]}\n")


def test_report(save_dir, unique_name, metrics):
    report_filename = f'{unique_name}-report.txt'
    with open(os.path.join(save_dir, report_filename), 'a') as report_file:
        report_file.write(f"Test Loss :{metrics[4]}\n")
        report_file.write(f"Test Accuracy :{metrics[0]}\n")
        report_file.write(f"Test Precision :{metrics[1]}\n")
        report_file.write(f"Test Sensitivity :{metrics[2]}\n")
        report_file.write(f"Test Specificity :{metrics[3]}\n")
        report_file.write(f"Test F1 Score :{metrics[4]}\n")

