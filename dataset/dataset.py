import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



preprocess = transforms.Compose([
transforms.Resize((512,512)),
#transforms.CenterCrop(256),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

folders = ['train', 'val', 'test']


class CatDogDataset(Dataset):
    """
    Custom Dataset

        """
    def __init__(self, image_dir, sub=''):
        """
        Arguments:
            image dir: string
                indicates where is the dataset
            sub: string
                subdirectory name at default there is no subdirectory
        """
        
        self.sub=sub
        self.image_dir = os.path.join(image_dir, sub)
        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(self.image_dir):
            self.image_paths.append(os.path.join(self.image_dir, img_name))
            self.labels.append(0 if img_name[:3] == 'dog' else 1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(image)
        label = float(self.labels[idx])
        return input_tensor, label
    

