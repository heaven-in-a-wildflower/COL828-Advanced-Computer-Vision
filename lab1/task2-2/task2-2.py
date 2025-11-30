#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import CLIPModel, CLIPProcessor

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder

class CLIPZeroShotClassifier:
    """Zero-shot CLIP classifier for Pacemaker dataset using Hugging Face Transformers"""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.full_clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.full_clip_model.eval()

        # Templates for prompt ensembling (Pacemaker-specific)
        self.templates = [
            "an X-ray of a {} pacemaker",
            "a radiograph showing a {} pacemaker",
            "an X-ray image of {} pacemaker",
            "a medical X-ray of a {} pacemaker",
            "radiological image of {} pacemaker",
            "X-ray showing {} pacemaker",
            "medical imaging of {} pacemaker",
            "cardiac X-ray of {} pacemaker",
            "a photo of {} pacemaker in X-ray",
            "medical radiograph of {} pacemaker",
        ]

        self.text_features = None
        self.class_names = None

    def build_text_features(self, class_names, batch_size: int = 32):
        """Build ensembled text features using templates"""
        self.class_names = class_names
        print(f"Building text features for {len(class_names)} classes")
        print(f"Classes: {class_names}")

        # Create all prompts
        all_prompts = []
        for template in self.templates:
            for cname in class_names:
                clean_name = cname.replace("_", " ").lower()
                all_prompts.append(template.format(clean_name))

        print(f"Generated {len(all_prompts)} prompts using {len(self.templates)} templates")

        # Compute features in batches
        all_text_feats = []
        with torch.no_grad():
            for i in range(0, len(all_prompts), batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                text_inputs = self.processor(
                    text=batch_prompts, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                feats = self.full_clip_model.get_text_features(**text_inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_text_feats.append(feats.cpu())

        text_features = torch.cat(all_text_feats, dim=0)
        num_templates = len(self.templates)
        num_classes = len(class_names)

        text_features = text_features.view(num_templates, num_classes, -1)
        text_features = text_features.mean(dim=0)  # Average across templates
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.to(self.device)
        print(f"Final text_features shape: {self.text_features.shape}")

    def predict_batch(self, images):
        """Predict on a batch of images"""
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                if images.dim() != 4:
                    raise ValueError("If passing tensor images they must be shape (B, C, H, W)")
                image_inputs = {"pixel_values": images.to(self.device)}
            else:
                image_inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

            image_features = self.full_clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.full_clip_model.logit_scale.exp()
            logits = logit_scale * (image_features @ self.text_features.t())

            probabilities = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

        return logits, probabilities, predictions

    def evaluate(self, dataloader, label_encoder: LabelEncoder = None):
        """Evaluate the classifier over a dataloader"""
        print("Evaluating CLIP zero-shot classifier...")
        all_preds = []
        all_probs = []
        all_labels = []

        self.full_clip_model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Handle dict-based dataloader
                if isinstance(batch, dict):
                    images = batch['img']
                    labels = batch['label']
                else:
                    images, labels = batch[0], batch[1]

                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)

                logits, probs, preds = self.predict_batch(images)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Handle labels
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.cpu().numpy()
                else:
                    labels_np = np.array(labels)

                if label_encoder is not None:
                    if labels_np.dtype.type is np.str_ or labels_np.dtype == object:
                        labels_np = label_encoder.transform(labels_np)
                else:
                    if labels_np.dtype.type is np.str_ or labels_np.dtype == object:
                        raise ValueError("Labels are strings but no label_encoder provided.")

                all_labels.extend(labels_np)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        return metrics, all_preds, all_probs

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate evaluation metrics"""
        top1_accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        num_classes = len(np.unique(y_true))
        if num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            auc_roc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        else:
            if y_prob.shape[1] == 2:
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=range(2))
                auc_roc = roc_auc_score(y_true_bin, y_prob, average="macro")

        return {
            "top1_accuracy": top1_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "auc_roc": auc_roc,
        }


class CLIPDataset(Dataset):
    """Dataset for CLIP that returns dict with img, label, class_name"""
    def __init__(self, df, processor, label_encoder):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['filepath']
        image = Image.open(image_path).convert("RGB")

        # Process image for CLIP
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        # Get label
        label_str = row['label']
        label = self.label_encoder.transform([label_str])[0]

        return {
            'img': pixel_values,
            'label': label,
            'class_name': label_str
        }


def load_data_from_folders(data_root):
    """
    Load data from folder structure:
    data_root/
        Train/
            Class1/
                img1.jpg
                img2.jpg
            Class2/
                ...
        Test/
            Class1/
                ...

    Returns train_df, test_df with columns: filepath, label
    """
    train_data = []
    test_data = []

    train_dir = os.path.join(data_root, 'Train')
    test_dir = os.path.join(data_root, 'Test')

    # Load training data
    if os.path.exists(train_dir):
        print(f"Loading training data from {train_dir}")
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_name)
                        train_data.append({
                            'filepath': img_path,
                            'label': class_name
                        })

    # Load test data
    if os.path.exists(test_dir):
        print(f"Loading test data from {test_dir}")
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_name)
                        test_data.append({
                            'filepath': img_path,
                            'label': class_name
                        })

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print(f"\nDataset loaded:")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nClasses found: {sorted(train_df['label'].unique())}")
    print(f"\nTrain class distribution:\n{train_df['label'].value_counts()}")

    return train_df, test_df


def split_train_val(train_df, val_size=0.2, random_state=42):
    """Split training data into train and validation sets"""
    train_df, val_df = train_test_split(
        train_df, 
        test_size=val_size, 
        stratify=train_df['label'], 
        random_state=random_state
    )

    print(f"\nData split:")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    return train_df, val_df


def create_clip_zero_shot_classifier(device):
    """Create CLIP zero-shot classifier"""
    return CLIPZeroShotClassifier(model_name="openai/clip-vit-base-patch32", device=device)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset path - ADJUST THIS FOR YOUR PACEMAKER DATASET
    data_root = '/kaggle/input/pacemakers'  # Should contain Train/ and Test/ folders

    # Load data from folder structure
    train_df, test_df = load_data_from_folders(data_root)

    # Split train into train and validation
    train_df, val_df = split_train_val(train_df, val_size=0.2, random_state=42)

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'])

    print(f"\nEncoded classes: {label_encoder.classes_}")

    # Create CLIP classifier
    clip_classifier = create_clip_zero_shot_classifier(device)

    # Build text features
    class_names = label_encoder.classes_.tolist()
    clip_classifier.build_text_features(class_names)

    # Create datasets
    val_dataset = CLIPDataset(val_df, clip_classifier.processor, label_encoder)
    test_dataset = CLIPDataset(test_df, clip_classifier.processor, label_encoder)

    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    val_metrics, val_preds, val_probs = clip_classifier.evaluate(val_loader, label_encoder)
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_metrics, test_preds, test_probs = clip_classifier.evaluate(test_loader, label_encoder)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

