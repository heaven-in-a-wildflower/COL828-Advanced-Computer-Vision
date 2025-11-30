#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install git+https://github.com/openai/CLIP.git')


# In[12]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt

_tokenizer = _Tokenizer()

# ============================================================================
# Dataset and Data Loading
# ============================================================================

class OrthonetDataset(Dataset):
    """Orthonet dataset for CLIP-based models"""

    def __init__(self, df, images_root, preprocess=None, label_encoder=None):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.preprocess = preprocess
        self.label_encoder = label_encoder

        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(df['labels'])
        else:
            self.labels = self.label_encoder.transform(df['labels'])

        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filenames']
        img_path = os.path.join(self.images_root, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Preprocess
        if self.preprocess:
            image = self.preprocess(image)

        label = self.labels[idx]

        return {
            'img': image,
            'label': label
        }


def load_and_split_data(train_csv_path, test_csv_path, val_split=0.2, random_state=42):
    """Load and split data into train/val/test"""
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=val_split, random_state=random_state, 
        stratify=train_df['labels']
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df


# In[13]:


# ============================================================================
# Text Encoder (Shared across all methods)
# ============================================================================

class TextEncoder(nn.Module):
    """CLIP Text Encoder"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if compound_prompts_deeper is not None:
            # MaPLe: pass deeper prompts
            combined = [x, compound_prompts_deeper, 0]
            outputs = self.transformer(combined)
            x = outputs[0]
        else:
            x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the eot embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# In[15]:


# ============================================================================
# CoOp: Context Optimization
# ============================================================================

class CoOpPromptLearner(nn.Module):
    """CoOp Prompt Learner"""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None, class_specific=False):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # Initialize with given words
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            if class_specific:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'CoOp Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        device = clip_model.token_embedding.weight.device
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        return prompts


class CoOpModel(nn.Module):
    """CoOp Model"""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None):
        super().__init__()
        self.prompt_learner = CoOpPromptLearner(classnames, clip_model, n_ctx, ctx_init)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# In[16]:


# ============================================================================
# CoCoOp: Conditional Context Optimization
# ============================================================================

class CoCoOpPromptLearner(nn.Module):
    """CoCoOp Prompt Learner with Instance-Conditional Context"""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'CoCoOp Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # Meta network for conditional prompts
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        self.meta_net = self.meta_net.to(dtype=clip_model.dtype)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        device = next(clip_model.parameters()).device
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        ctx_vectors = ctx_vectors.to(device)
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Create prompts for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = torch.cat([prefix, ctx_i, suffix], dim=1)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts

class CoCoOpModel(nn.Module):
    """CoCoOp Model"""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None):
        super().__init__()
        print(f"Classname={classnames}")
        self.prompt_learner = CoCoOpPromptLearner(classnames, clip_model, n_ctx, ctx_init)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = self.logit_scale.exp() * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits


# In[1]:


# ============================================================================
# Training and Evaluation
# ============================================================================
train_accs = []
train_losses = []
def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    return epoch_loss, epoch_acc

val_accs = []
f1_macro_list = []
f1_weighted_list = []
auc_roc_list = []
def evaluate(model, dataloader, device, num_classes):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['img'].to(device)
            labels = batch['label']

            logits = model(images)
            probabilities = F.softmax(logits, dim=-1)

            all_predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    # Calculate metrics
    top1_accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

    # AUC-ROC
    if num_classes > 2:
        y_true_binary = label_binarize(all_labels, classes=range(num_classes))
        auc_roc = roc_auc_score(y_true_binary, all_probabilities, multi_class='ovr', average='macro')
    else:
        auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])

    # Update global lists
    val_accs.append(top1_accuracy*100)
    f1_macro_list.append(f1_macro)
    f1_weighted_list.append(f1_weighted)
    auc_roc_list.append(auc_roc)

    metrics = {
        'top1_accuracy': top1_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_roc': auc_roc
    }

    return metrics

def plot_training_metrics(train_losses, train_accs, val_accs, f1_macro_list, f1_weighted_list, auc_roc_list):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 12))

    # Train loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='red', marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracies
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='green', marker='o')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # F1 Scores
    plt.subplot(2, 2, 3)
    plt.plot(epochs, f1_macro_list, label='F1 Macro', color='purple', marker='o')
    plt.plot(epochs, f1_weighted_list, label='F1 Weighted', color='orange', marker='o')
    plt.title('F1 Scores per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()

    # AUC-ROC
    plt.subplot(2, 2, 4)
    plt.plot(epochs, auc_roc_list, label='AUC-ROC', color='brown', marker='o')
    plt.title('AUC-ROC per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("/kaggle/working/plot.png")

def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       num_classes, device, num_epochs=50, lr=0.002, model_type='default'):
    """Train and evaluate a model"""
    # For CoOp, CoCoOp, MaPLe - train only prompt_learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad = False
    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    best_val_acc = 0
    best_model_state = None
    save_path = "best_model.pth"  # filename for saving

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, num_classes)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_metrics['top1_accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics['top1_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['top1_accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print(f"✅ Best model saved at epoch {epoch + 1} with Val Acc: {best_val_acc:.4f}")

        scheduler.step()

    plot_training_metrics(
    train_losses, train_accs, val_accs,
    f1_macro_list, f1_weighted_list, auc_roc_list
    )

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, num_classes)

    return test_metrics


# In[23]:


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths (modify these according to your setup)
    train_csv_path = '/kaggle/input/orthonet-data/train.csv'
    test_csv_path = '/kaggle/input/orthonet-data/test.csv'
    images_root = '/kaggle/input/orthonet-data/orthonet data/orthonet data'

    # Load data
    train_df, val_df, test_df = load_and_split_data(train_csv_path, test_csv_path)

    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['labels'])
    classnames = label_encoder.classes_.tolist()
    num_classes = len(classnames)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {classnames}")

    # Load CLIP
    print("\nLoading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Create datasets
    train_dataset = OrthonetDataset(train_df, images_root, preprocess, label_encoder)
    val_dataset = OrthonetDataset(val_df, images_root, preprocess, label_encoder)
    test_dataset = OrthonetDataset(test_df, images_root, preprocess, label_encoder)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    results = {}

    # ========== CoOp ==========
    # print("\n" + "="*60)
    # print("Training CoOp")
    # print("="*60)
    # coop_model = CoOpModel(classnames, clip_model, n_ctx=16).to(device)
    # coop_metrics = train_and_evaluate(
    #     coop_model, train_loader, val_loader, test_loader, 
    #     num_classes, device, num_epochs=50, lr=0.002
    # )
    # results['CoOp'] = coop_metrics

    # # # ========== CoCoOp ==========
    print("\n" + "="*60)
    print("Training CoCoOp")
    print("="*60)
    cocoop_model = CoCoOpModel(classnames, clip_model, n_ctx=16).to(device)
    cocoop_metrics = train_and_evaluate(
        cocoop_model, train_loader, val_loader, test_loader, 
        num_classes, device, num_epochs=50, lr=0.001
    )
    results['CoCoOp'] = cocoop_metrics

    # # Check which parameters will be updated
    # trainable_params = [name for name, param in cocoop_model.named_parameters() if param.requires_grad]
    # print(f"Trainable parameters: {trainable_params}")

    # ========== Print Results ==========
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)

    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    return results


if __name__ == "__main__":
    results = main()


# In[26]:


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths (modify these according to your setup)
    train_csv_path = '/kaggle/input/orthonet-data/train.csv'
    test_csv_path = '/kaggle/input/orthonet-data/test.csv'
    images_root = '/kaggle/input/orthonet-data/orthonet data/orthonet data'

    # Load data
    train_df, val_df, test_df = load_and_split_data(train_csv_path, test_csv_path)

    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['labels'])
    classnames = label_encoder.classes_.tolist()
    num_classes = len(classnames)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {classnames}")

    # Load CLIP
    print("\nLoading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Create datasets
    train_dataset = OrthonetDataset(train_df, images_root, preprocess, label_encoder)
    val_dataset = OrthonetDataset(val_df, images_root, preprocess, label_encoder)
    test_dataset = OrthonetDataset(test_df, images_root, preprocess, label_encoder)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    results = {}

    # ========== CoOp ==========
    print("\n" + "="*60)
    print("Training CoOp")
    print("="*60)
    coop_model = CoOpModel(classnames, clip_model, n_ctx=16).to(device)
    coop_metrics = train_and_evaluate(
        coop_model, train_loader, val_loader, test_loader, 
        num_classes, device, num_epochs=50, lr=0.001
    )
    results['CoOp'] = coop_metrics

    # # # ========== CoCoOp ==========
    # print("\n" + "="*60)
    # print("Training CoCoOp")
    # print("="*60)
    # cocoop_model = CoCoOpModel(classnames, clip_model, n_ctx=16).to(device)
    # cocoop_metrics = train_and_evaluate(
    #     cocoop_model, train_loader, val_loader, test_loader, 
    #     num_classes, device, num_epochs=50, lr=0.001
    # )
    # results['CoCoOp'] = cocoop_metrics

    # # Check which parameters will be updated
    # trainable_params = [name for name, param in cocoop_model.named_parameters() if param.requires_grad]
    # print(f"Trainable parameters: {trainable_params}")

    # ========== Print Results ==========
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)

    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    return results


if __name__ == "__main__":
    results = main()

