import random
import shutil
import glob
import os
from sklearn.model_selection import train_test_split


train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
dir = "dataset/cat_dog"
folders = ['train', 'val', 'test']



# TODO
# Clean code Check
# dosya ismi split_dataset etc Check
# orneklere bakilsin kod okunurlugu cok dusuk Check
# train test split oranlari dinamik olsun Check
# assertler bir testtir yorum satiri ekle ne oldugunu anlamam icin yeni fonskiyon Check
# sklearn kutuphanesine goz atilabilir Check


def DeleteCreateFolders()-> None:
    #Delete and Create folders
    for folder in folders:
        save_folder = os.path.join(dir, folder)
        try:
            shutil.rmtree(save_folder)
        except OSError:
            print('Previous dataset not found')
        os.makedirs(save_folder)


def isPathValid(dir:str)-> bool:
    result = False
    if os.path.isdir(dir):
        result = True
        return result
    else:    
        assert result, "Dataset cannot found"


def isLabeled(labels:list[str])->bool:
    result = False

    #Look for previous builded folders and not include them as labels
    for folder in folders:
        if folder in labels:
            labels.remove(folder)

    if((len(labels))!= 0):
        result = True
        return result
    else:
        assert (len(labels)) != 0, "Dataset labels cannot found or already builded"


def split_directories(dir:str, labels:list, folders:list[str])-> None:

    '''
        builds dataset directories distributes data randomly with needed ratio
        
        Args:
        -----
            dir : str
                root directory of dataset
            labels : list
                it consists of sub-directories(which means labels) of dataset

        Returns:
        --------
            None
        '''
    
    DeleteCreateFolders()
    for label in labels:
        local_dir = os.path.join(dir, label)
        file_list = []

        #images on one label
        file_list = glob.glob(local_dir + '/*')
        random.shuffle(file_list)

        train, test = train_test_split(file_list, test_size=val_ratio+test_ratio)
        val, test = train_test_split(test, test_size=test_ratio/(val_ratio+test_ratio))
        
        paths_all = [train, val, test]

        #create folders and copy and send all data where needs
        for (folder, paths) in zip(folders, paths_all):
            save_folder = os.path.join(dir, folder)
            for file_path in paths:
                shutil.copy(file_path, save_folder)

if __name__ == "__main__":
    #Test the validity of path
    if isPathValid(dir):
        labels = os.listdir(dir)

    #Test subdirectories as labels    
    if isLabeled(labels):
        split_directories(dir, labels, folders)


