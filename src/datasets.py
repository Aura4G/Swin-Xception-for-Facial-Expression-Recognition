from torch.utils.data import Dataset
from PIL import Image
import os

class FERDataset(Dataset): #Use torch.utils.data's Dataset and Dataloader classes for preprocessing
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform #torchvision.transforms
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        #Finding file paths and loading all images and labels
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self): #len function overload to output the number of images in a directory
        return len(self.image_paths)

    def __getitem__(self, idx): #getitem function overload to return an image and its corresponding label from its index
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label