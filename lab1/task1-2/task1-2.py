#!/usr/bin/env python
# coding: utf-8

# In[14]:


from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(train_csv_path, test_csv_path, val_size=0.2, random_state=42):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Keep only required columns and standardize names
    train_df = train_df[['filenames', 'labels']].copy()
    train_df.columns = ['filenames', 'labels']

    print(f"Dataset shape: {train_df.shape}")
    print(f"Classes: {train_df['labels'].unique()}")
    print(f"Class distribution:\n{train_df['labels'].value_counts()}")

    train_df, val_df = train_test_split(
        train_df, test_size=val_size, 
        stratify=train_df['labels'], 
        random_state=random_state
    )

    print(f"\nData split:")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples") 
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df

class CLIPDataset(Dataset):
    def __init__(self, df, images_root, processor, image_col="filenames", label_col="labels"):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.processor = processor
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_root, str(row[self.image_col]))
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        label = row[self.label_col]
        return pixel_values, label


# In[15]:


import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder

class CLIPZeroShotClassifier:
    """Zero-shot CLIP classifier for Orthonet dataset using Hugging Face Transformers"""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading CLIP model: {model_name}")
        # Use the full CLIPModel (both vision + text) to keep dims consistent
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.full_clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.full_clip_model.eval()

        # Templates for prompt ensembling (Orthonet-specific)
        self.templates = [
            "an X-ray of a {} implant",
            "a radiograph showing a {} implant",
            "an X-ray image of {} implant",
            "a medical X-ray of a {} implant",
            "radiological image of {} implant",
            "X-ray showing {} implant",
            "medical imaging of {} implant",
            "orthopedic X-ray of {} implant",
            "a photo of {} implant in X-ray",
            "medical radiograph of {} implant",
        ]

        self.text_features = None  # Tensor (num_classes, feature_dim)
        self.class_names = None

    def build_text_features(self, class_names, batch_size: int = 32):
        """
        Build ensembled text features using self.templates.
        text_features will be a tensor of shape (num_classes, feature_dim) on self.device.
        """
        self.class_names = class_names
        print(f"Building text features for {len(class_names)} classes")
        print(f"Classes example: {class_names[:10]}")

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
                feats = self.full_clip_model.get_text_features(**text_inputs)  # (B, D)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_text_feats.append(feats.cpu())  # keep on CPU until concat to save GPU mem

        text_features = torch.cat(all_text_feats, dim=0)  # (num_templates * num_classes, D) on CPU
        # reshape to (num_templates, num_classes, D)
        num_templates = len(self.templates)
        num_classes = len(class_names)
        expected = num_templates * num_classes
        if text_features.shape[0] != expected:
            raise RuntimeError(f"Text features length mismatch: got {text_features.shape[0]}, expected {expected}")

        text_features = text_features.view(num_templates, num_classes, -1)  # (T, C, D)
        # average across templates
        text_features = text_features.mean(dim=0)  # (C, D)
        # normalize and move to device
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.to(self.device)  # (C, D) on device
        print(f"Final text_features shape: {self.text_features.shape}")

    def predict_batch(self, images):
        """
        images: either
            - a torch.Tensor of shape (B, C, H, W) that already contains pixel_values expected by CLIP
            - a list of PIL.Images or numpy arrays that the processor can handle
        Returns (logits, probabilities, predictions)
        """
        with torch.no_grad():
            # If images is a tensor (already preprocessed pixel_values)
            if isinstance(images, torch.Tensor):
                # Expect (B, C, H, W)
                if images.dim() != 4:
                    raise ValueError("If passing tensor images they must be shape (B, C, H, W)")
                image_inputs = {"pixel_values": images.to(self.device)}
            else:
                # Let processor handle preprocessing for PIL/numpy images or list of such images
                image_inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

            image_features = self.full_clip_model.get_image_features(**image_inputs)  # (B, D)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # text_features: (C, D) ; we want logits: (B, C)
            # logit_scale is a learnable parameter (initialized around log(1/0.07) ≈ 2.659), controls how sharp the similarity distribution is
            logit_scale = self.full_clip_model.logit_scale.exp()
            logits = logit_scale * (image_features @ self.text_features.t())  # (B, C) 

            probabilities = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

        return logits, probabilities, predictions

    def evaluate(self, dataloader, label_encoder: LabelEncoder = None):
        """
        Evaluate the classifier over a dataloader.
        Expects dataloader to yield (pixel_values_tensor, label) where pixel_values_tensor is (B, C, H, W)
        and label is either tensors or numpy arrays or lists. If labels are strings, label_encoder must be provided.
        """
        print("Evaluating CLIP zero-shot classifier...")
        all_preds = []
        all_probs = []
        all_labels = []

        self.full_clip_model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # batch assumed (images, labels)
                images, labels = batch[0], batch[1]
                # images may be a tensor already prepared by dataset
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                logits, probs, preds = self.predict_batch(images)

                # collect predictions/probs
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # handle labels: convert to numpy array; transform if strings and encoder provided
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.cpu().numpy()
                else:
                    labels_np = np.array(labels)

                if label_encoder is not None:
                    # If dtype is object (strings) or not integers, try transforming
                    if labels_np.dtype.type is np.str_ or labels_np.dtype == object:
                        labels_np = label_encoder.transform(labels_np)
                else:
                    # if labels are strings and no encoder provided, try to convert to ints
                    if labels_np.dtype.type is np.str_ or labels_np.dtype == object:
                        raise ValueError("Labels are strings but no label_encoder provided to transform them to integers.")

                all_labels.extend(labels_np)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        return metrics, all_preds, all_probs

    def calculate_metrics(self, y_true, y_pred, y_prob):
        # Top-1 Accuracy
        top1_accuracy = accuracy_score(y_true, y_pred)
        # F1 macro & weighted
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        # AUC-ROC
        num_classes = len(np.unique(y_true))
        if num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            auc_roc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        else:
            # binary case: expect probabilities shape (N, 2)
            if y_prob.shape[1] == 2:
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # fallback: compute roc_auc_score with one-vs-rest binarized
                y_true_bin = label_binarize(y_true, classes=range(2))
                auc_roc = roc_auc_score(y_true_bin, y_prob, average="macro")

        return {
            "top1_accuracy": top1_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "auc_roc": auc_roc,
        }


def create_clip_zero_shot_classifier(device):
    return CLIPZeroShotClassifier(model_name="openai/clip-vit-base-patch32", device=device)


# In[16]:


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    train_csv_path = '/kaggle/input/orthonet-data/train.csv'
    test_csv_path = '/kaggle/input/orthonet-data/test.csv'
    images_root = '/kaggle/input/orthonet-data/orthonet data/orthonet data'  # Adjust if images are in a subfolder

    train_df, val_df, test_df = load_and_split_data(train_csv_path, test_csv_path)

    # Example placeholders — replace with real values
    train_df, val_df, test_df = load_and_split_data(train_csv_path, test_csv_path)
    label_encoder = LabelEncoder(); label_encoder.fit(train_df['labels'])

    clip_classifier = create_clip_zero_shot_classifier(device)
    class_names = label_encoder.classes_.tolist()
    clip_classifier.build_text_features(class_names)

    val_dataset = CLIPDataset(val_df, images_root, clip_classifier.processor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=None)

    val_metrics, val_preds, val_probs = clip_classifier.evaluate(val_loader, label_encoder)
    print(val_metrics)

