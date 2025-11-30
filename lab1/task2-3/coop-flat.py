#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install git+https://github.com/openai/CLIP.git')


# In[2]:


import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ============================================================================
# Dataset
# ============================================================================
class PacemakerDataset(Dataset):
    """Pacemaker dataset for CLIP-based models"""

    def __init__(self, root_dir, preprocess=None, label_encoder=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.label_encoder = label_encoder

        # Collect image paths + class folders
        self.samples = []
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Split company name (before ' - ') and full class name
            if " - " in class_folder:
                company, _ = class_folder.split(" - ", 1)
            else:
                company = class_folder

            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({
                        "filepath": os.path.join(class_path, fname),
                        "class_name": class_folder,
                        "company": company
                    })

        # Encode labels - always use fine-grained class names
        all_labels = [s["class_name"] for s in self.samples]

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(all_labels)
        else:
            self.labels = self.label_encoder.transform(all_labels)

        # Attach labels to samples
        for i, s in enumerate(self.samples):
            s["label"] = self.labels[i]

        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["filepath"]).convert("RGB")

        if self.preprocess:
            image = self.preprocess(image)

        return {
            "img": image,
            "label": sample["label"],
            "company": sample["company"],
            "class_name": sample["class_name"]
        }


def create_pacemaker_datasets(root_dir, preprocess=None, val_split=0.2, random_state=42):
    """Create train/val/test datasets for fine-grained classification"""
    train_root = os.path.join(root_dir, "Train")
    test_root = os.path.join(root_dir, "Test")

    full_train_dataset = PacemakerDataset(train_root, preprocess=preprocess)
    samples = full_train_dataset.samples
    labels = [s["label"] for s in samples]

    train_idx, val_idx = train_test_split(
        range(len(samples)),
        test_size=val_split,
        stratify=labels,
        random_state=random_state
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_dataset = PacemakerDataset(train_root, preprocess=preprocess, 
                                     label_encoder=full_train_dataset.label_encoder)
    val_dataset = PacemakerDataset(train_root, preprocess=preprocess, 
                                   label_encoder=full_train_dataset.label_encoder)

    train_dataset.samples = train_samples
    train_dataset.labels = [s["label"] for s in train_samples]
    val_dataset.samples = val_samples
    val_dataset.labels = [s["label"] for s in val_samples]

    test_dataset = PacemakerDataset(test_root, preprocess=preprocess, 
                                    label_encoder=full_train_dataset.label_encoder)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# Text Encoder (shared by CoOp and CoCoOp)
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

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the eot embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# ============================================================================
# CoOp: Context Optimization
# ============================================================================
class CoOpPromptLearner(nn.Module):
    """CoOp Prompt Learner with SHARED context"""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Shared context: shape (n_ctx, ctx_dim)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'CoOp Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")
        print(f"Using SHARED prompts across all {n_cls} classes")

        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, ctx_dim)

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
        ctx = self.ctx  # (n_ctx, ctx_dim)
        # Expand to all classes
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)

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


# ============================================================================
# CoCoOp: Conditional Context Optimization
# ============================================================================
class CoCoOpPromptLearner(nn.Module):
    """CoCoOp Prompt Learner with SHARED instance-conditional context"""

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
            # Shared context: shape (n_ctx, ctx_dim)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'CoCoOp Initial context: "{prompt_prefix}"')
        print(f"Number of context words: {n_ctx}")
        print(f"Using SHARED conditional prompts across all {n_cls} classes")

        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, ctx_dim)

        # Meta network for conditional prompts
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if dtype == torch.float16:
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        device = next(clip_model.parameters()).device
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Create prompts for all classes per image
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


# ============================================================================
# Training and Evaluation (FIXED)
# ============================================================================
def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch - FIXED to compute logits once"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Compute logits ONCE
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        # Use the SAME logits for accuracy (no recomputation, no mode switching)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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

    # Top-3 accuracy
    if num_classes >= 3:
        top3_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=3)
    else:
        top3_accuracy = top1_accuracy

    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # AUC-ROC
    try:
        if num_classes > 2:
            y_true_binary = label_binarize(all_labels, classes=range(num_classes))
            auc_roc = roc_auc_score(y_true_binary, all_probabilities, multi_class='ovr', average='macro')
        else:
            auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
    except ValueError:
        auc_roc = 0.0

    metrics = {
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_roc': auc_roc
    }

    return metrics


def plot_training_curves(history, model_name='', save_path='training_curves.png'):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Training Curves (45 Classes)', fontsize=16, fontweight='bold')

    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'g-o', label='Train', linewidth=2, markersize=4)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Top-1 Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['val_top3'], 'purple', linewidth=2, markersize=4)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Top-3 Accuracy')
    axes[0, 2].set_title('Validation Top-3 Accuracy')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['f1_macro'], 'purple', label='Macro', linewidth=2, markersize=4)
    axes[1, 0].plot(epochs, history['f1_weighted'], 'orange', label='Weighted', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['auc_roc'], 'brown', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC-ROC')
    axes[1, 1].set_title('AUC-ROC Score')
    axes[1, 1].grid(True, alpha=0.3)

    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.show()


# ============================================================================
# Main Training Pipeline
# ============================================================================
def train_model(model_type, root_dir, clip_model, preprocess, device):
    """Train a single model on 45 fine-grained classes"""

    print("\n" + "="*70)
    print(f"{model_type} - Fine-grained Classification (45 classes)")
    print("="*70)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_pacemaker_datasets(
        root_dir, preprocess=preprocess
    )

    classnames = train_dataset.label_encoder.classes_.tolist()
    num_classes = len(classnames)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Sample classes: {classnames[:5]}...")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    if model_type == 'CoOp':
        model = CoOpModel(classnames, clip_model, n_ctx=10)
    else:  # CoCoOp
        model = CoCoOpModel(classnames, clip_model, n_ctx=10)

    model.to(device)

    # Freeze all except prompt learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=1e-4
    )

    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [],
        'val_top3': [], 'f1_macro': [], 'f1_weighted': [], 'auc_roc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_metrics = evaluate(model, val_loader, device, num_classes)
        history['val_acc'].append(val_metrics['top1_accuracy'])
        history['val_top3'].append(val_metrics['top3_accuracy'])
        history['f1_macro'].append(val_metrics['f1_macro'])
        history['f1_weighted'].append(val_metrics['f1_weighted'])
        history['auc_roc'].append(val_metrics['auc_roc'])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Top-1: {val_metrics['top1_accuracy']:.4f}, Val Top-3: {val_metrics['top3_accuracy']:.4f}")
        print(f"F1 Macro: {val_metrics['f1_macro']:.4f}, AUC-ROC: {val_metrics['auc_roc']:.4f}")

        if val_metrics['top1_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['top1_accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"✓ New best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    # Load best model
    model.load_state_dict(best_model_state)

    # Plot
    save_path = f'/kaggle/working/{model_type}_training_curves.png'
    plot_training_curves(history, model_name=model_type, save_path=save_path)

    # Test evaluation
    print("\n" + "="*70)
    print(f"{model_type} - Test Evaluation")
    print("="*70)

    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"\nTest Results:")
    print(f"  Top-1 Accuracy: {test_metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f}")
    print(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")

    return model, test_metrics, history


# ============================================================================
# Main Function
# ============================================================================
def main(model_type='CoOp'):
    """
    model_type: 'CoOp' or 'CoCoOp'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training model: {model_type}")

    root_dir = "/kaggle/input/pacemakers"

    # Load CLIP
    print("\nLoading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    # Train model on 45 classes directly
    model, test_metrics, history = train_model(
        model_type=model_type,
        root_dir=root_dir,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device
    )

    return {
        'model': model,
        'test_metrics': test_metrics,
        'history': history
    }


# ============================================================================
# Run experiments
# ============================================================================
if __name__ == "__main__":
    # Run CoOp
    # print("\n" + "="*70)
    # print("STARTING CoOp SINGLE-STAGE TRAINING (45 Classes)")
    # print("="*70)
    # coop_results = main(model_type='CoOp')

    # # Run CoCoOp
    print("\n" + "="*70)
    print("STARTING CoCoOp SINGLE-STAGE TRAINING (45 Classes)")
    print("="*70)
    cocoop_results = main(model_type='CoCoOp')

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - CoOp vs CoCoOp (45 Classes)")
    print("="*70)

    # print("\n### CoOp Results ###")
    # for k, v in coop_results['test_metrics'].items():
    #     print(f"  {k}: {v:.4f}")

    print("\n### CoCoOp Results ###")
    for k, v in cocoop_results['test_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)

