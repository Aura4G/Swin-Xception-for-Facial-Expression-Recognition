import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional
from PIL import Image
from torchvision.transforms import transforms

### GLOBAL VARIABLES ###

DATA_DIR = "datasets/RAFDB/DATASET/train/"
OUTPUT_DIR = "./cam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

categories = sorted(os.listdir(DATA_DIR))


data = {"activations": None, "gradients": None}

# Haar Cascade Classifier, utilised to detect faces in non-standardised images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Torchvision transforms to normalise and scale image inputs
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

### FEATURE EXTRACTION AND SYNTHESIS FOR STAGE 2 OF TRAINING ###

def extract_features(model, dataloader, device):
    """
    Freezes the passed model's and completes single pass of a dataset's images through the feature-extractor backbone, and compiles
    the resulting features (along with their corresponding labels) into data structures.

    Args:
        model (nn.Module): The Swin-Xception model that performs the feature extraction.
        dataloader (DataLoader): The batch-processed dataset used for the feature extraction.
        device (torch.device): Defines which device the model's feature extraction is being ran on.

    Returns:
        features (List): Data structure containing all of the extracted features for each class.
        labels (List): The data structure of labels that parallel the features data structure.
    """

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
    """
    Counts total feature support per-class and applies SMOTE to all classes except the majority to create uniform support.

    Args:
        features (List): A list of features extracted from the frozen Swin-Xception backbone.
        labels (List): A list of labels corresponding to the features list
        random_state (int): Sets the random seed for the SMOTE algorithm. Default: 42, to match all other random seed instantiations.

    Returns:
        balanced_features (List): A list of features, having been supplied synthetic samples of minority classes
        balanced_labels (List): A list of labels, parallel to the feature list.
    """

    # Count every feature extracted by class
    print("Original class distribution:")
    print(Counter(labels))
    
    smote = SMOTE(random_state=random_state, k_neighbors=5) # Uses 5 nearest neighbours to interpolate new samples
    balanced_features, balanced_labels = smote.fit_resample(features, labels) # smote.fit_resample returns the lists of balanced features and labels.

    print("Balanced class distribution:")
    print(Counter(balanced_labels))

    return balanced_features, balanced_labels


### GRAD CAM FUNCTIONS ###

def crop_to_square(img):
    """
    Crops non-standardised images (images that are not natively from a dataset) to a square
    
    Args:
        img: Variably-sized image input

    Returns:
        img: Image, where height = width
    """

    h, w = img.shape[:2]
    
    if w > h:
        start_x = (w - h) // 2
        return img[:, start_x : start_x + h]
    elif h > w:
        start_y = (h - w) // 2
        return img[start_y : start_y + w, :]
    else:
        return img

def get_face_crop(image):
    """ Takes a non-standardised image and uses a Haar cascade classifier to locate a face and crop the image to the face. """

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

def save_activations(module, input, output):
    """
    Hook function to capture the output activations of a registered module.
    Stores the output tensor in the external `data` dict under 'activations'.

    Args:
        module (nn.Module): The module the hook is registered to.
        input (tuple): Inputs to the module (unused).
        output (torch.Tensor): Output activations of the module.
    """
    data["activations"] = output

def save_gradients(module, grad_input, grad_output):
    """
    Hook function to capture the gradient of the loss with respect to a 
    registered module's output. Stores the gradient in the external `data` 
    dict under 'gradients'.

    Args:
        module (nn.Module): The module the hook is registered to.
        grad_input (tuple): Gradients with respect to the module's inputs (unused).
        grad_output (tuple[torch.Tensor]): Gradients with respect to the 
                                           module's output; first element is stored.
    """
    data["gradients"] = grad_output[0]

def compute_heatmap(model, img_tensor):
    """
    Computes a Grad-CAM heatmap for the most confident predicted class.
    Performs a forward and backward pass to collect activations and gradients
    via registered hooks, then computes a weighted activation map.

    Args:
        model (nn.Module): The model to explain. Must have save_activations and 
                           save_gradients hooks registered prior to calling.
        img_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).

    Returns:
        tuple:
            - cam (np.ndarray): Heatmap of shape (H', W') where H' and W' are 
                                the spatial dimensions of the activation layer.
            - prediction (int): Index of the predicted class.
    """

    model.zero_grad()

    output = model(img_tensor)
    _, prediction = output.max(1)
    

    output[0, prediction].backward()
    

    acts = data["activations"]
    grads = data["gradients"]
    

    weights = torch.mean(grads, dim=1, keepdim=True)

    cam = torch.sum(weights * acts, dim=-1)

    b, l = cam.shape
    side = int(l**0.5) 
    cam = cam.view(b, side, side)
    
    cam = torch.relu(cam)
    cam = cam.detach().cpu().squeeze().numpy()
    
    return cam, prediction.item()

def upsample_heatmap(heatmap, image):
    """
    Upsamples and overlays a Grad-CAM heatmap onto the original image.
    Handles both torch.Tensor and np.ndarray inputs for both arguments,
    normalises the heatmap to [0, 255], resizes it to match the image 
    resolution, and blends them using a 70/30 heatmap/image weighting.

    Args:
        heatmap (torch.Tensor | np.ndarray): Raw CAM output of shape (H', W'), 
                                             where H' and W' are the spatial 
                                             dimensions of the activation layer.
        image (torch.Tensor | np.ndarray): Original input image. If a Tensor, 
                                           expected shape is (1, C, H, W) or 
                                           (C, H, W) in RGB format.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with the heatmap overlaid 
                    on the original image.
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze()
        if image.dim() == 3:
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.numpy()

    if torch.is_tensor(heatmap):
        heatmap = torch.maximum(heatmap, torch.tensor(0.0))
        heatmap = heatmap.numpy()
    else:
        heatmap = np.maximum(heatmap, 0)

    m, M = heatmap.min(), heatmap.max()
    if M - m > 0:
        heatmap_norm = 255 * ((heatmap - m) / (M - m))
    else:
        heatmap_norm = np.zeros_like(heatmap)
        
    heatmap_norm = np.uint8(heatmap_norm)
    
    heatmap_resized = cv2.resize(heatmap_norm, (image.shape[1], image.shape[0]))
    
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    if image.max() <= 1.0:
        image = np.uint8(image * 255)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    combined_img = cv2.addWeighted(heatmap_color, 0.7, image_bgr, 0.3, 0)
    
    return cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

def display_images(combined_image, image):
    """
    Displays the Grad-CAM heatmap overlay and the original image side by side.

    Args:
        combined_image (np.ndarray): Heatmap overlay image of shape (H, W, 3), 
                                     as returned by upsample_heatmap.
        image (np.ndarray): Original input image of shape (H, W, 3).
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(combined_image)
    axes[0].set_title("Heatmap")
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].set_title("Original Image")
    axes[1].axis('off')
    plt.show()

def produce_grad_cam_images_from_dataset(model, device):
    """ 
    Takes the first image of each folder in the dataset, predicts its class, and outputs a grad-cam image to a folder.
    
    Args:
        model (nn.Module): The Swin-Xception model being utilised to produce grad-CAM images
        device (torch.device): The device the model is utilising
    """

    model.eval()

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(save_activations)
    target_layer.register_full_backward_hook(save_gradients)

    for category in categories:
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue

        img_name = os.listdir(category_path)[0]
        img_path = os.path.join(category_path, img_name)

        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        input_image_norm = np.float32(rgb_img) / 255
        input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)
        
        heatmap, pred_idx = compute_heatmap(model, input_tensor)
        predicted_label = categories[pred_idx]
        is_correct = (predicted_label == category)

        cam_image = upsample_heatmap(heatmap, rgb_img)
        display_images(cam_image, rgb_img)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(OUTPUT_DIR, f"{category}_image_predicted_as_{predicted_label}.jpg")
        cv2.imwrite(save_path, cam_image)

        print(f"Processed {category}, Predicted: {predicted_label}")

def produce_grad_cam_image(model, img_path, device):
    """ 
    Locates a local image, preprocesses it, predicts its class, and outputs a grad-cam image to a folder.
    
    Args:
        model (nn.Module): The Swin-Xception model being utilised to produce grad-CAM images
        img_path (string): The local path of the image to be classified and Explained via Grad-CAM
        device (torch.device): The device the model is utilising
    """

    model.eval()

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(save_activations)
    target_layer.register_full_backward_hook(save_gradients)

    img = cv2.imread(img_path)
    img_face = get_face_crop(img)

    rgb_img = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
    rgb_img = crop_to_square(rgb_img)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    input_image_norm = np.float32(rgb_img) / 255
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    heatmap, pred_idx = compute_heatmap(model, input_tensor)
    predicted_label = categories[pred_idx]

    cam_image = upsample_heatmap(heatmap, rgb_img)
    display_images(cam_image, rgb_img)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"predicted_{predicted_label}.jpg", cam_image)
    print(f"Predicted: {predicted_label}")


### Outputting Metrics ###

def get_predictions(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Obtains all of the model's predictions and ground truth labels of every image in a given dataset.

    Args:
        model (nn.Module): The Swin-Xception model making predictions.
        dataloader (DataLoader): The batched dataset the model is running predictions on.
        device (torch.device): The device the model is utilising.

    Returns:
        all_preds (np.array) : Numpy array of all the model's predicted labels.
        all_labels (np.array) : Numpy array of all of the ground truth labels.
    """

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """
    Plot and optionally save confusion matrices.

    Plots a confusion matrix counting actual predictions, and a confusion matrix normalised for percentages.

    Args:
        y_true (List): List of ground truth labels.
        y_pred (List): List of model-predicted labels.
        class_names (List[string]): List of class names.
        title (string): The title of the graphs.
        save_path (string): The file path to save visualised confusion matrices to. Default: None (results in not being saved).
    """

    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{title} - Raw Counts')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'{title} - Percentages (%)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

def print_detailed_metrics(y_true, y_pred, class_names, dataset_name):
    """
    Outputs:
    - The confusion matrices
    - Accuracy
    - Precision
    - Recall
    - F1-Score

    Args:
        y_true (List): List of ground truth labels.
        y_pred (List): List of model-predicted labels.
        class_names (List[string]): List of class names.
        dataset_name: The dataset being evaluated.
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Detailed Metrics")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    print("\nPer-Class Accuracy:")
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"{class_name:12s}: {acc:6.2f}%")
    
    # Overall accuracy
    overall_acc = (y_true == y_pred).sum() / len(y_true) * 100
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

def calculate_uar_war(y_true, y_pred, class_names):
    """
    Calculate UAR and WAR metrics
    
    UAR (Unweighted Average Recall): Mean of per-class recalls (treats all classes equally)

    WAR (Weighted Average Recall): Weighted mean of per-class recalls (weighted by class frequency)

    Args:
        y_true (List): List of ground truth labels.
        y_pred (List): List of model-predicted labels.
        class_names (List[string]): List of class names.

    Returns:
        uar (float): Unweighted Average Recall
        war (float): Weighted Average Recall
        per_class_recall (float)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class recall (same as per-class accuracy for single-label classification)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    
    # UAR: Simple average of all class recalls
    uar = np.mean(per_class_recall) * 100
    
    # WAR: Weighted average by class support
    class_support = cm.sum(axis=1)
    war = np.average(per_class_recall, weights=class_support) * 100
    
    return uar, war, per_class_recall

def print_detailed_metrics_with_uar_war(y_true, y_pred, class_names, dataset_name):
    """
    Prints detailed classification metrics, including UAR and WAR

    Args:
        y_true (List): List of ground truth labels.
        y_pred (List): List of model-predicted labels.
        class_names (List[string]): List of class names.
        dataset_name (string): The dataset being evaluated.
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Detailed Metrics")
    print(f"{'='*60}")
    
    # Standard classification report
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Calculate UAR and WAR
    uar, war, per_class_recall = calculate_uar_war(y_true, y_pred, class_names)
    
    print("\n" + "-"*60)
    print("Per-Class Recall (%):")
    print("-"*60)
    for class_name, recall in zip(class_names, per_class_recall):
        print(f"{class_name:12s}: {recall*100:6.2f}%")
    
    print("\n" + "-"*60)
    print("Summary Metrics:")
    print("-"*60)
    overall_acc = (y_true == y_pred).sum() / len(y_true) * 100
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"WAR (Weighted Average Recall): {war:.2f}%")
    print(f"UAR (Unweighted Average Recall): {uar:.2f}%")
    print("-"*60)


### t-SNE ###

def visualise_tsne(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: Optional[list[str]] = None,
    n_pca_components: int = 50,
    tsne_perplexity: float = 30.0,
    tsne_max_iter: int = 1000,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    title: str = "t-SNE of SwinXception GAP Features",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract post-GAP features from SwinXception and visualise with t-SNE.

    The extraction hook attaches after self.avgpool1d so features are
    768-d vectors — semantically rich, spatially summarised, and compact
    enough for reliable t-SNE embedding.

    Pipeline:  raw tokens (B, 49, 768)
                → LayerNorm → AdaptiveAvgPool1d → (B, 768)
                → PCA (50-d)  → t-SNE (2-d)  → scatter plot

    Args:
        model (nn.Module): SwinXception instance (eval mode recommended).
        dataloader (DataLoader): Yields (images, labels) batches.
        class_names (List[string]): Optional list mapping label indices to strings.
        n_pca_components (int): PCA dimensionality before t-SNE (set 0 to skip).
        tsne_perplexity (float): t-SNE perplexity; typically 5–50.
        tsne_max_iter (int): t-SNE optimisation iterations.
        device (torch.device): 'cuda', 'cpu', or auto-detected if None.
        save_path (string): If given, saves the figure to this path.
        title (string): Plot title.

    Returns:
        embeddings (List): (N, 2) t-SNE coordinates.
        labels (List[int]): (N,) ground-truth class indices.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Register a forward hook on avgpool1d to capture GAP output      
    gap_features: list[torch.Tensor] = []

    def _hook(module, input, output):
        # output shape: (B, 768, 1)  →  squeeze to (B, 768)
        gap_features.append(output.squeeze(-1).detach().cpu())

    hook_handle = model.avgpool1d.register_forward_hook(_hook)

    # Forward pass — no gradients needed                              
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)          # triggers the hook
            all_labels.append(labels.cpu())

    hook_handle.remove()

    features = torch.cat(gap_features, dim=0).numpy()   # (N, 768)
    labels   = torch.cat(all_labels,   dim=0).numpy()   # (N,)

    print(f"Extracted features: {features.shape}  |  samples: {len(labels)}")

    # ------------------------------------------------------------------ #
    # 3.  Optional PCA pre-reduction                                      #
    # ------------------------------------------------------------------ #
    if n_pca_components > 0 and features.shape[1] > n_pca_components:
        print(f"PCA: {features.shape[1]}-d → {n_pca_components}-d …")
        pca = PCA(n_components=n_pca_components, random_state=42)
        features = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance retained: {explained:.1%}")

    # ------------------------------------------------------------------ #
    # 4.  t-SNE embedding                                                 #
    # ------------------------------------------------------------------ #
    print(f"t-SNE: {features.shape[1]}-d → 2-d  "
          f"(perplexity={tsne_perplexity}, n_iter={tsne_max_iter}) …")

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        max_iter=tsne_max_iter,
        random_state=42,
        init="pca",          # more stable than random init
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(features)   # (N, 2)
    print("Done.")

    # ------------------------------------------------------------------ #
    # 5.  Plot                                                            #
    # ------------------------------------------------------------------ #
    n_classes = len(np.unique(labels))
    cmap      = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    colours   = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colours[cls_idx]],
            s=12,
            alpha=0.75,
            linewidths=0,
            label=class_names[cls_idx] if class_names else str(cls_idx),
        )

    legend = ax.legend(
        loc="best",
        framealpha=0.2,
        edgecolor="white",
        labelcolor="white",
        fontsize=9,
    )
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return embeddings, labels