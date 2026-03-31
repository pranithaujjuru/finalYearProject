import torch
import torch.nn as nn
import timm
import os
import time

def zero_inference_test():
    print("Starting Zero-Inference Test for Swin-L...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path relative to project root
    model_path = "./models/SOTA_Vision/swin_large_sciatica.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    # Load architecture
    print("Loading Swin-L model architecture...")
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=False)
    in_features = model.head.fc.in_features
    # Matching the head in vision_agent.py
    model.head.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 3) 
    )
    
    # Load weights
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Memory check
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"VRAM Allocated: {memory_allocated:.2f} MB")
    
    # Dummy Inference
    print("Running dummy inference (384x384 tensor)...")
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        
    print(f"Inference complete. Time: {(end_time - start_time)*1000:.2f} ms")
    
    # Verify shape
    if output.shape == (1, 3):
        print("Verification SUCCESS: Output shape is (1, 3) [Herniation, Stenosis, No Finding]")
    else:
        print(f"Verification FAILED: Unexpected output shape {output.shape}")
        
    # Final memory check
    if torch.cuda.is_available():
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"VRAM Reserved: {memory_reserved:.2f} MB")

if __name__ == "__main__":
    zero_inference_test()
