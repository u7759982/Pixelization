import os
import random
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# directories for the dataset
data_dir = "image1w"
real_dir = os.path.join(data_dir, "real")
unreal_dir = os.path.join(data_dir, "unreal")

# create directory structure
def create_directory_structure():
    # create main directories
    os.makedirs("data/train/real", exist_ok=True)
    os.makedirs("data/train/unreal", exist_ok=True)
    os.makedirs("data/validation/real", exist_ok=True)
    os.makedirs("data/validation/unreal", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)  # separate test directory
    
    print("Directory structure created successfully.")

# dataset splitting function
def split_dataset(real_dir, unreal_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # list of the images
    real_images = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    unreal_images = [f for f in os.listdir(unreal_dir) if os.path.isfile(os.path.join(unreal_dir, f))]
    
    # shuffle    images
    random.shuffle(real_images)
    random.shuffle(unreal_images)
    
    # split real images
    real_train_end = int(len(real_images) * train_ratio)
    real_val_end = real_train_end + int(len(real_images) * val_ratio)
    
    real_train = real_images[:real_train_end]
    real_val = real_images[real_train_end:real_val_end]
    real_test = real_images[real_val_end:]
    
    # split unreal images
    unreal_train_end = int(len(unreal_images) * train_ratio)
    unreal_val_end = unreal_train_end + int(len(unreal_images) * val_ratio)
    
    unreal_train = unreal_images[:unreal_train_end]
    unreal_val = unreal_images[unreal_train_end:unreal_val_end]
    unreal_test = unreal_images[unreal_val_end:]
    
    # copy files to respective directories
    # training set
    for img in real_train:
        shutil.copy(os.path.join(real_dir, img), os.path.join("data/train/real", img))
    
    for img in unreal_train:
        shutil.copy(os.path.join(unreal_dir, img), os.path.join("data/train/unreal", img))
    
    # validation set
    for img in real_val:
        shutil.copy(os.path.join(real_dir, img), os.path.join("data/validation/real", img))
    
    for img in unreal_val:
        shutil.copy(os.path.join(unreal_dir, img), os.path.join("data/validation/unreal", img))
    
    # test set with name prefix
    for img in real_test:
        shutil.copy(os.path.join(real_dir, img), os.path.join("data/test", f"real_{img}"))
    
    for img in unreal_test:
        shutil.copy(os.path.join(unreal_dir, img), os.path.join("data/test", f"unreal_{img}"))
    
    print(f"Dataset split complete:")
    print(f"Training: {len(real_train)} real, {len(unreal_train)} unreal")
    print(f"Validation: {len(real_val)} real, {len(unreal_val)} unreal")
    print(f"Test: {len(real_test)} real, {len(unreal_test)} unreal")
    
    # saparate test images
    with open("test_images.txt", "w") as f:
        for img in real_test:
            f.write(f"real_{img},0\n")  # 0 for real
        for img in unreal_test:
            f.write(f"unreal_{img},1\n")  # 1 for unreal

# data augmentation and normalization
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# load data
def load_data(batch_size=32):
    train_transform, val_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(root="data/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root="data/validation", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.class_to_idx

# build the model using transfer learning
def build_model(num_classes=2):
    # load resnet50 model
    model = models.resnet50(pretrained=True)
    
    # freeze most of the layers
    for param in model.parameters():
        param.requires_grad = False
    
    # replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model

# train the model
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f}")
        
        # learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # save the best model locally
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'class_to_idx': train_loader.dataset.class_to_idx
            }, best_model_path)
            print(f"Model saved at epoch {epoch+1} with validation loss: {epoch_val_loss:.4f}")
    
    # plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model

# evaluate the model on test set
def evaluate_model():
    # load the best model
    model = build_model()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # define test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # read test images from the text file
    test_images = []
    with open("test_images.txt", "r") as f:
        for line in f:
            filename, label = line.strip().split(',')
            test_images.append((filename, int(label)))
    
    # evaluate the model
    predictions = []
    true_labels = []
    
    for filename, label in test_images:
        img_path = os.path.join("data/test", filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            
        predictions.append(predicted.item())
        true_labels.append(label)
    
    # calculate accuracy
    accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # generate classification report
    cr = classification_report(true_labels, predictions, target_names=['Real', 'Unreal'])
    print("Classification Report:")
    print(cr)
    
    # save evaluation results
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(cr)

# save the model for inference
def save_model_for_inference():
    model = build_model()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # save the model for inference
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    torch.save({
        'model': model,
        'idx_to_class': idx_to_class,
        'input_size': (224, 224),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }, 'realunreal_classifier.pt')
    
    print("Model saved for inference as 'realunreal_classifier.pt'")
    
#   main function to run the entire process
def main():
    print("Starting the Real vs Unreal image classifier training process...")
    
    # create directory structure
    create_directory_structure()
    
    # split the dataset
    split_dataset(real_dir, unreal_dir)
    
    # load data
    train_loader, val_loader, class_to_idx = load_data()
    print(f"Class mapping: {class_to_idx}")
    
    # build and train the model
    model = build_model()
    train_model(model, train_loader, val_loader, num_epochs=15)
    
    #evaluate the model
    evaluate_model()
    
    # save the model for inference
    save_model_for_inference()
    
    print("Complete! The model has been trained, evaluated, and saved.")

if __name__ == "__main__":
    main()