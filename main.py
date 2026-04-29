from src import utils
from src import engine
from src import swinxception
from src import datasets

import torch
import torch.nn as nn

import os
import argparse

### STAGE ONE ###
def end_to_end_training(epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Stage One: End-to-End Training.

    This encapsulates the first stage of the training pipeline, in which the model is initialised
    (or loaded from a checkpoint if training was interrupted), and is trained in its entirety for a set number of epochs.
    Training and Validation are both run once per epoch, to give way to early-stopping conditions. At the end of training,
    The model is evaluated on the host (RAF-DB) and holdout (FER2013) test sets.

    Args:
        epochs (int): The maximum number of epochs the model is being trained for. Default: 100.
        device (torch.device): Either "cuda" or "cpu", depending on availability.

    Returns:
        model (nn.Module): The End-to-End trained and evaluated Swin-Xception model. This model can either
                            be utilised for inference as is, or utilised for SMOTE retraining (Stages 2 and 3 of the training pipeline).
    """

    train_loader, val_loader, test_raf_loader, test_fer_loader = datasets.load_datasets()
    model, criterion, optimiser, scheduler, start_epoch = engine.build_swinxception_model(epochs, device)
    model = engine.training_loop(model, train_loader, val_loader, criterion, optimiser, scheduler, device, start_epoch, epochs)
    
    raf_loss, raf_acc = engine.validate(model, test_raf_loader, criterion)
    fer_loss, fer_acc = engine.validate(model, test_fer_loader, criterion)

    print(f"RAF-DB Testing  | Loss: {raf_loss:.4f} | Accuracy: {raf_acc:.2f}%")
    print(f"FER2013 Testing | Loss: {fer_loss:.4f} | Accuracy: {fer_acc:.2f}%")

    return model

### STAGE TWO ###
def apply_smote_to_dataset(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Stage Two: Feature Extraction for SMOTE Retraining.

    This encapsulates the second stage of the training pipeline; the now end-to-end trained model is frozen for the
    purpose of feature extraction into two parallel data structures of deep features and labels. The features and labels
    are then used to determine the class distribution and apply "not majority" SMOTE to achieve a uniform distribution of features
    and labels.

    Args:
        device (torch.device): Either "cuda" or "cpu" depending on availability.

    Returns:
        (list): A list of the uniformly distributed features, including synthetically sampled features.
        (list): A list, parallel to the balanced features list, composed of the labels corresponding to all of the features.
    """

    model = engine.load_swinxception_model("swin_xception_baseline.pth")

    train_loader, _, _, _ = datasets.load_datasets()

    features, labels = utils.extract_features(model, train_loader, device)

    return utils.apply_smote(features, labels)

### STAGE THREE ###
def mlp_head_retraining(balanced_features, balanced_labels, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Stage Three: MLP Head Retraining on SMOTE'd dataset

    This encapsulates the third and final stage of the training pipeline; The synthetically oversampled
    features and labels culminate in their own dataset, which is used by the model to retrain its Linear
    Projection layer (The MLP Head). The rest of the model is frozen during the retraining process.

    Args:
        balanced_features (list): A list of the uniformly distributed features, including synthetically sampled features.
        balanced_labels (list): A list, parallel to the balanced features list, composed of the labels corresponding to all of the features.
        device (torch.device): Either "cuda" or "cpu" depending on availability.

    Returns:
        model (nn.Module): The retrained model. This model is ready for inference, with no further additions
        to be made.
    """

    model = engine.load_swinxception_model("swin_xception_baseline.pth")

    model = engine.retrain_mlp_head(model, balanced_features, balanced_labels, device)

    _, _, test_raf_loader, test_fer_loader = datasets.load_datasets()

    criterion = nn.CrossEntropyLoss()

    raf_loss, raf_acc = engine.validate(model, test_raf_loader, criterion)
    fer_loss, fer_acc = engine.validate(model, test_fer_loader, criterion)

    print(f"RAF-DB Testing  | Loss: {raf_loss:.4f} | Accuracy: {raf_acc:.2f}%")
    print(f"FER2013 Testing | Loss: {fer_loss:.4f} | Accuracy: {fer_acc:.2f}%")

    return model


### FULL TRAINING PIPELINE ###
def complete_training_pipeline(stage_one_epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Incorporates all three stages of the training pipeline into a single function:
    - End-to-End Training
    - Feature Extraction and SMOTE
    - MLP Head Retraining

    Args:
        stage_one_epochs (int): Maximum number of epochs to run
        device (torch.device): Either "cuda" or "cpu" depending on availability.

    Returns:
        swinxception_base (nn.Module): The End-to-End trained Swin-Xception model, with no SMOTE applied to it
        swinxception_final (nn.Module): The Swin-Xception model after SMOTE has been applied to the dataset, and
                                    the model's MLP head has been retrained on the balanced dataset.
    """

    # Stage One
    swinxception_base = end_to_end_training(stage_one_epochs, device)

    # Stage Two
    balanced_features, balanced_labels = apply_smote_to_dataset(device)

    # Stage Three
    swinxception_final = mlp_head_retraining(balanced_features, balanced_labels, device)

    return swinxception_base, swinxception_final


### REPORT METRICS ###
def report_all_metrics(model_path="swin_xception_final.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Prints every classification-based metric scored by the model:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - WAR & UAR

    This function also prints per-class recall, and produces the Confusion matrices and t-SNE graphs to an image folder.

    Args:
        model_path (string): The local path to the state dictionary of the SwinXception model being utilised. Default: swin_xception_final.pth
        device (torch.device): Either "cuda" or "cpu" depending on availability.
    """

    os.makedirs("image_figures", exist_ok=True)

    model = engine.load_swinxception_model(model_path)

    _, val_loader, test_raf_loader, test_fer_loader = datasets.load_datasets()

    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    print("\n" + "="*60)
    print("Generating Confusion Matrices and Metrics")
    print("="*60)

    # RAF-DB Test Set
    print("\nEvaluating RAF-DB test set...")
    raf_preds, raf_labels = utils.get_predictions(model, test_raf_loader, device)
    utils.plot_confusion_matrix(raf_labels, raf_preds, emotions, 
                        'RAF-DB Test Set',
                        save_path='image_figures/rafdb_confusion_matrix.png')
    utils.print_detailed_metrics_with_uar_war(raf_labels, raf_preds, emotions, 'RAF-DB')

    # FER2013 Test Set
    print("\nEvaluating FER2013 test set...")
    fer_preds, fer_labels = utils.get_predictions(model, test_fer_loader, device)
    utils.plot_confusion_matrix(fer_labels, fer_preds, emotions, 
                        'FER2013 Test Set', 
                        save_path='image_figures/fer2013_confusion_matrix.png')
    utils.print_detailed_metrics_with_uar_war(fer_labels, fer_preds, emotions, 'FER2013')

    print("\n" + "="*60)
    print("Minority Class Performance (Disgust & Fear)")
    print("="*60)

    minority_classes = ['disgust', 'fear']
    minority_indices = [emotions.index(cls) for cls in minority_classes]

    print(f"\n{'Class':<12} {'Dataset':<10} {'Recall':<10} {'Samples':<10}")
    print("-"*60)

    for cls_name, cls_idx in zip(minority_classes, minority_indices):
        
        # RAF-DB
        raf_mask = raf_labels == cls_idx
        if raf_mask.sum() > 0:
            raf_recall = (raf_preds[raf_mask] == raf_labels[raf_mask]).sum() / raf_mask.sum() * 100
            print(f"{cls_name.capitalize():<12} {'RAF-DB':<10} {raf_recall:>8.2f}% {raf_mask.sum():>9}")

        # FER2013
        fer_mask = fer_labels == cls_idx
        if fer_mask.sum() > 0:
            fer_recall = (fer_preds[fer_mask] == fer_labels[fer_mask]).sum() / fer_mask.sum() * 100
            print(f"{cls_name.capitalize():<12} {'FER2013':<10} {fer_recall:>8.2f}% {fer_mask.sum():>9}")
        
        print()

    embeddings, labels = utils.visualise_tsne(model=model, dataloader=val_loader, class_names=emotions, save_path="image_figures/tsne.png")


### Grad-CAM ###
def produce_grad_cam_image(model_path="swin_xception_final.pth", img_path=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Loads a Swin-Xception model from a state dictionary, and utilises the loaded model to predict and produce a Grad-CAM image
    of a local image file.

    Args:
        model_path (string): The local path to the state dictionary of the SwinXception model being utilised. Default: swin_xception_final.pth
        img_path (string): The local path of the image to be classified and Explained via Grad-CAM
        device (torch.device): Either "cuda" or "cpu" depending on availability.
    """
    if img_path is not None:
        model = engine.load_swinxception_model(model_path)
        
        utils.produce_grad_cam_image(model, img_path, device)

def produce_grad_cam_images_from_set(model_path="swin_xception_final.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Loads a Swin-Xception model from a state dictionary, and utilises the loaded model to predict and produce Grad-CAM images of
    the first image of each class in the RAF-DB dataset.

    model_path (string): The local path to the state dictionary of the SwinXception model being utilised. Default: swin_xception_final.pth
    device (torch.device): Either "cuda" or "cpu" depending on availability.
    """
    
    model = engine.load_swinxception_model(model_path)

    utils.produce_grad_cam_images_from_dataset(model, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinXception Training and Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage One
    train_parser = subparsers.add_parser("train", help="Run end-to-end training (Stage 1)")
    train_parser.add_argument("--epochs", type=int, default=100)

    # Stages Two & Three (SMOTE + MLP retraining)
    subparsers.add_parser("smote-retrain", help="Run SMOTE feature extraction and MLP head retraining (Stages 2 & 3)")

    # Full pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the complete training pipeline (Stages 1-3)")
    pipeline_parser.add_argument("--epochs", type=int, default=100)

    # Metrics
    metrics_parser = subparsers.add_parser("metrics", help="Report all classification metrics")
    metrics_parser.add_argument("--model-path", type=str, default="swin_xception_final.pth")

    # Grad-CAM (single image)
    gradcam_parser = subparsers.add_parser("gradcam", help="Produce Grad-CAM for a single image")
    gradcam_parser.add_argument("--model-path", type=str, default="swin_xception_final.pth")
    gradcam_parser.add_argument("--img-path", type=str, required=True)

    # Grad-CAM (from dataset)
    gradcam_set_parser = subparsers.add_parser("gradcam-set", help="Produce Grad-CAM images from RAF-DB dataset")
    gradcam_set_parser.add_argument("--model-path", type=str, default="swin_xception_final.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        end_to_end_training(epochs=args.epochs, device=device)

    elif args.command == "smote-retrain":
        balanced_features, balanced_labels = apply_smote_to_dataset(device=device)
        mlp_head_retraining(balanced_features, balanced_labels, device=device)

    elif args.command == "pipeline":
        complete_training_pipeline(stage_one_epochs=args.epochs, device=device)

    elif args.command == "metrics":
        report_all_metrics(model_path=args.model_path, device=device)

    elif args.command == "gradcam":
        produce_grad_cam_image(model_path=args.model_path, img_path=args.img_path, device=device)

    elif args.command == "gradcam-set":
        produce_grad_cam_images_from_set(model_path=args.model_path, device=device)