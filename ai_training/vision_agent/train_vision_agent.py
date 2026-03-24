import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

def main():
    # Hardware Acceleration: Detect and use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Define paths robustly based on this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "../dataset/LumbarSpinalStenosis")
    models_dir = os.path.join(base_dir, "../../models")
    os.makedirs(models_dir, exist_ok=True)

    # Data Preprocessing: Resize, convert to RGB (standard in ImageFolder), normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "test")

    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Dataset directories not found in '{dataset_dir}'.")
        print("Please ensure the 'train' and 'test' folders exist.")
        return

    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Memory-Safe Data Loading
    # num_workers=0 is safer on Windows to avoid multiprocessing issues with DataLoader
    batch_size = 32
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model Architecture: Load EfficientNetB0
    print("Initializing model...")
    # use weights parameter as trained parameter is deprecated
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head for 3 classes
    num_classes = 3 # Herniated Disc, Thecal Sac Compression, No Stenosis
    # EfficientNet classifier is a Sequential block, we replace the final Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters that require gradients (the classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # Training Loop
    num_epochs = 5
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).float().sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct_train / total_train

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).float().sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    # Save Artifact
    model_save_path = os.path.join(models_dir, "vision_agent_v1.pth")
    print(f"Saving model weights to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Training complete and model saved successfully.")

if __name__ == "__main__":
    main()
