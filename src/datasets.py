import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import v2
import os

# Transformations that are applied the RAF-DB training and validation sets
transform_train = v2.Compose([
    v2.RandomAffine(degrees=10, scale=(0.8, 1.0), translate=(0.1, 0.1), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
    v2.Resize(size=(224, 224), antialias=True), 
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations, all of which are necessary for correct inference, to be applied on RAF-DB and FER2013 test sets
transform_test = v2.Compose([
    v2.Resize(size=(232, 232), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    
def load_datasets():
    """
    Sources dataset paths and creates batched data loaders for each relevant dataset.

    Returns:
        train_loader (DataLoader): The 80% split of the RAF-DB Training set the model uses to train on once per epoch
        val_loader (DataLoader): The 20% split of the RAF-DB Training set the model uses to validate on once per epoch
        test_raf_loader (DataLoader): The RAF-DB Test dataset
        test_fer_loader (DataLoader): The FER2013 Test dataset
    """

    full_raf_train = FERDataset(os.path.abspath("datasets/RAFDB/DATASET/train"), transform_train)
    raf_test = FERDataset(os.path.abspath("datasets/RAFDB/DATASET/test"), transform_test)
    fer_test = FERDataset(os.path.abspath("datasets/FER2013/test"), transform_test)

    train_size = int(0.8 * len(full_raf_train))
    val_size = len(full_raf_train) - train_size

    raf_train, raf_val = random_split(full_raf_train, [train_size, val_size])

    train_loader = DataLoader(raf_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(raf_val, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_raf_loader = DataLoader(raf_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_fer_loader = DataLoader(fer_test, batch_size=32,shuffle=False, num_workers=4, pin_memory=True)

    print(f"RAF-DB Training set images: {len(raf_train)}")
    print(f"RAF-DB Validation set images: {len(raf_val)}")
    print(f"RAF-DB Test set images: {len(raf_test)}")
    print(f"FER2013 Test set images: {len(fer_test)}")

    return train_loader, val_loader, test_raf_loader, test_fer_loader