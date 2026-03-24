import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
learning_rate = 1e-3
dataset_path = "dataset"


def get_datasets():
    img_size = 128 #224
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, "validation"), transform=val_transform)

    class_names = train_dataset.classes

    def print_distribution(dataset, name):
        labels = [label for _, label in dataset.samples]
        counts = np.bincount(labels)
        print(f"\n{name} set per-class distribution:")
        for cls, count in zip(class_names, counts):
            print(f"{cls}: {count}")

    print_distribution(train_dataset, "Training")
    print_distribution(val_dataset, "Validation")

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        len(class_names),
        class_names
    )


def create_efficientnet_b3_model(num_classes):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, class_names, max_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 7

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / total

        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(acc)

        print(f"Epoch {epoch}/{max_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {acc:.2f}% | "
              f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

        scheduler.step(avg_val_loss)

        save_confusion_matrix(all_labels, all_preds, class_names, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ New best model saved! (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            break

    return model, train_losses, val_losses, val_accuracies


def save_confusion_matrix(y_true, y_pred, class_names, epoch):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix Epoch {epoch}:\n{cm}\n")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_path = f"confusion_matrix_epoch_{epoch}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_training_history(train_losses, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, num_classes, class_names = get_datasets()
    model = create_efficientnet_b3_model(num_classes).to(device)

    print(f"\nTraining EfficientNet-B3 on {num_classes} classes:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, class_names, max_epochs=25
    )

    plot_training_history(train_losses, val_losses, val_accuracies)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")