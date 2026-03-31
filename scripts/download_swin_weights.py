import os
import torch
import torch.nn as nn
import timm

def download_and_save_swin():
    print("Initiating Swin Transformer (Large) weight download...")
    
    # Configuration
    model_name = 'swin_large_patch4_window12_384'
    save_dir = './models/SOTA_Vision/'
    save_path = os.path.join(save_dir, 'swin_large_sciatica.pth')
    
    # Ensure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Load model with SOTA weights from timm
    print(f"Fetching pretrained weights for {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    
    # model.head usually contains global_pool and fc
    print("Adjusting model head for 3-class pathology output...")
    # Using timm's native classification head if possible or manual replacement with pooling
    in_features = model.head.fc.in_features
    # We replace only the internal FC part of the head, keeping the global pooling
    model.head.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3) 
    )
    
    # Save the state_dict
    torch.save(model.state_dict(), save_path)
    print(f"SOTA weights successfully saved to: {save_path}")
    
    # File size check
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Total Disk Usage: {file_size:.2f} MB")

if __name__ == "__main__":
    download_and_save_swin()
