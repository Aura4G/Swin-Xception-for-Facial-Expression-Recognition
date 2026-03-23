import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from imblearn.oversampling import SMOTE
import cv2


### FEATURE EXTRACTION FUNCTIONS FOR STAGE 2 OF TRAINING ###

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)

            x = model.patch_embed(images)

            for block in model.layer1:
                x = block(x)
            x = model.merge1(x)

            for block in model.layer2:
                x = block(x)
            x = model.merge2(x)

            for block in model.layer3:
                x = block(x)
            x = model.merge3(x)
            
            for block in model.layer4:
                x = block(x)

            x = model.norm(x)
            x = x.transpose(1, 2)
            x = model.avgpool1d(x)
            x = torch.flatten(x, 1)

            all_features.append(x.cpu().numpy())
            all_labels.append(labels.numpy())

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        return features, labels

def apply_smote(features, labels, random_state=42):
    print("Original class distribution:")
    print(Counter(labels))
    
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    balanced_features, balanced_labels = smote.fit_resample(features, labels)

    print("Balanced class distribution:")
    print(Counter(balanced_labels))

    return balanced_features, balanced_labels


### GRAD CAM FUNCTIONS ###

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def crop_to_square(img):
    h, w = img.shape[:2]
    
    if w > h:
        start_x = (w - h) // 2
        return img[:, start_x : start_x + h]
    elif h > w:
        start_y = (h - w) // 2
        return img[start_y : start_y + w, :]
    else:
        return img
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_crop(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        side = max(w, h)
        center_x, center_y = x + w//2, y + h//2

        nx1 = max(0, center_x - side//2)
        ny1 = max(0, center_y - side//2)
        nx2 = min(image.shape[1], nx1 + side)
        ny2 = min(image.shape[0], ny1 + side)

        return image[ny1:ny2, nx1:nx2]

    return image

def produce_grad_cam_images_from_dataset():
    pass