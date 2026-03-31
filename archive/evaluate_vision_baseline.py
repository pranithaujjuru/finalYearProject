import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "ai_training/dataset/LumbarSpinalStenosis")
    model_path = os.path.join(base_dir, "models/vision_agent_v1.pth")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_dir):
        print("Test directory missing.")
        return

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loaded {len(test_dataset)} test images.")

    # Load Model
    model = models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3) # 3 classes
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model weights.")
    else:
        print("No existing model found. Using untrained weights.")

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 10 == 0:
                print(f"Processed {total} images...")
            
            # For demonstration in this session, evaluate on first 640 images (20 batches)
            if total >= 640:
                break

    accuracy = 100 * correct / total
    print(f"\n--- Results ---")
    print(f"Samples Evaluated: {total}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()
