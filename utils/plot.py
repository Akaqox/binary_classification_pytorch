import os
from utils.utils import createFolder
from datetime import date
import matplotlib.pyplot as plt

plt_dir = 'results/plots'        

def plot_loss(unique_name:str, train_loss:list, val_loss:list)-> None:

    plt.plot(train_loss, color='g', label='Training')
    plt.plot(val_loss, color='r', label='Validation')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()

    today = str(date.today().strftime("%Y-%m-%d"))
    save_dir = os.path.join(plt_dir, today)
    createFolder(save_dir)
    save_dir = os.path.join(save_dir, 'Loss_plt_' + unique_name[:-4])
    plt.savefig(save_dir)
    plt.show()


def plot_acc(unique_name:str, train_acc:list, val_acc:list)-> None:

    plt.plot(train_acc, color='g', label='Training')
    plt.plot(val_acc, color='r', label='Validation')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()

    today = str(date.today().strftime("%Y-%m-%d"))
    save_dir = os.path.join(plt_dir, today)
    createFolder(save_dir)
    save_dir = os.path.join(save_dir, 'Acc_plt_' + unique_name[:-4])
    plt.savefig(save_dir)
    plt.show()