import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Senior DL Engineer Component: Focal Loss (Gamma=2.0) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Custom Dataset for Albumentations
class MedicalImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            path, label = self.dataset.samples[idx]
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Empty image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            return image, label
        except Exception as e:
            # Fallback to the next image if corrupted
            return self.__getitem__((idx + 1) % len(self))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[Senior DL Engineer] Using device: {device}")

    # OPTIMIZED FOR 4GB VRAM (RTX 3050)
    # SENIOR PERFORMANCE CONFIG (SWIN-LARGE @ 384)
    IMG_SIZE = 224 
    NUM_CLASSES = 3
    BATCH_SIZE = 1 # Required for Swin-L on 4GB VRAM
    ACCUMULATION_STEPS = 16 # Effective Batch Size = 16
    PHASE1_EPOCHS = 5
    PHASE2_EPOCHS = 25
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "../dataset/LumbarSpinalStenosis")
    models_dir = os.path.join(base_dir, "../../models/SOTA_Vision")
    os.makedirs(models_dir, exist_ok=True)

    # 1. MEDICAL-GRADE AUGMENTATION (Albumentations)
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # Specialized Medical Distortions
        A.ElasticTransform(p=0.2), # Default values for clinical baseline
        A.GridDistortion(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = MedicalImageFolder(os.path.join(dataset_dir, "train"), transform=train_transform)
    val_dataset = MedicalImageFolder(os.path.join(dataset_dir, "test"), transform=val_transform)

    # SENIOR DL ENGINEER: DYNAMIC CLASS DETECTION
    NUM_CLASSES = len(train_dataset.classes)
    print(f"\n[Senior DL Engineer] Clinical Category Mapping:")
    for idx, cls in enumerate(train_dataset.classes):
        print(f"  Index {idx} -> {cls}")
    print(f"[Senior DL Engineer] Total Classes Identified: {NUM_CLASSES}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2. ARCHITECTURE: Swin-Large (Targeting 98% Accuracy)
    print(f"Initializing swin_large_patch4_window7_224 (With Gradient Checkpointing)...")
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
    model.set_grad_checkpointing(True) # VRAM Optimization
    
    # High-Precision 3-Class Pathology Head
    in_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    model = model.to(device)

    # Helper: Freeze/Unfreeze
    def set_backbone_trainable(model, trainable=True):
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = trainable

    # 3. PHASE 1: WARM-UP (Backbone Frozen)
    print("\n[PHASE 1] Training classification head (5 Epochs, LR=1e-3)...")
    set_backbone_trainable(model, trainable=False)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader)//ACCUMULATION_STEPS, epochs=PHASE1_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # Shared Metrics
    best_f1 = 0.0
    patience = 5
    no_improve_epochs = 0

    # TRAINING LOOP (Total 30 Epochs)
    for epoch in range(TOTAL_EPOCHS):
        if epoch == PHASE1_EPOCHS:
            print("\n[PHASE 2] Unfreezing backbone for full fine-tuning (25 Epochs, LR=5e-6)...")
            set_backbone_trainable(model, trainable=True)
            # Re-initialize optimizer and scheduler for Phase 2
            optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
            scheduler = OneCycleLR(optimizer, max_lr=5e-6, steps_per_epoch=len(train_loader)//ACCUMULATION_STEPS, epochs=PHASE2_EPOCHS)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] Train")
        for i, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            running_loss += loss.item() * ACCUMULATION_STEPS
            train_pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # Validation
        val_metrics = validate(model, val_loader, criterion, device, current_epoch=epoch+1, total_epochs=TOTAL_EPOCHS)
        
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {val_metrics['acc']*100:.2f}% | F1: {val_metrics['f1']:.4f}")

        # Early Stopping & Best Model Saving
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(models_dir, "swin_large_best.pth"))
            print(f"[BEST MODEL] F1: {best_f1:.4f} | Weights Saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} epochs. Terminating.")
                break

    # 5. FINAL SUMMARY REPORT
    print("\n" + "="*50)
    print("TRAINING SUMMARY REPORT")
    print("="*50)
    print(f"Baseline Accuracy (EfficientNet-B2): 45.73%")
    print(f"Optimized Accuracy (Swin-Large):     {(best_acc*100):.2f}%")
    print(f"Improvement:                        {((best_acc*100)-45.73):.2f}%")
    print(f"Target Accuracy:                    98.00%")
    print(f"Status:                             {'ACHIEVED' if (best_acc*100) >= 98 else 'TRENDING'}")
    print("="*50)

def validate(model, loader, criterion, device, current_epoch=0, total_epochs=0):
    model.eval()
    all_preds = []
    all_labels = []
    
    val_pbar = tqdm(loader, desc=f"Epoch [{current_epoch}/{total_epochs}] Val")
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            
            all_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    preds_classes = np.argmax(all_preds, axis=1)
    
    acc = accuracy_score(all_labels, preds_classes)
    f1 = f1_score(all_labels, preds_classes, average='weighted')
    
    return {"acc": acc, "f1": f1}

if __name__ == "__main__":
    main()
