# ================= IMPORTS =================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================= DEVICE =================
device = "cpu"

# ================= CLASS NAMES =================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ================= DATA =================
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# ================= MODEL =================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAIN + TEST (AUTO 5 TIMES) =================
TOTAL_EPOCHS = 5
epoch_accuracies = []

for epoch in range(1, TOTAL_EPOCHS + 1):

    # ---------- TRAIN ----------
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ---------- TEST (100 images) ----------
    model.eval()
    correct = 0
    total = 100
    shown_images = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            if i == total:
                break

            image, label = image.to(device), label.to(device)
            output = model(image)
            predicted = torch.argmax(output, dim=1)

            if predicted == label:
                correct += 1

            if len(shown_images) < 10:
                shown_images.append((image.cpu(), predicted.item()))

    accuracy = (correct / total) * 100
    epoch_accuracies.append(accuracy)

    # ---------- PRINT ----------
    print(f"\nEpoch {epoch} Accuracy: {accuracy:.2f}%")

    # ---------- SHOW IMAGES ----------
    plt.figure(figsize=(10, 5))
    for idx, (img, pred) in enumerate(shown_images):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[pred])
        plt.axis("off")

    plt.suptitle(f"Epoch {epoch} Accuracy: {accuracy:.2f}%")
    plt.show()

# ================= FINAL RESULT =================
print("\n============== FINAL SUMMARY ==============")
for i, acc in enumerate(epoch_accuracies, start=1):
    print(f"Epoch {i} Accuracy: {acc:.2f}%")

print(f"\nFINAL Accuracy after {TOTAL_EPOCHS} epochs: {epoch_accuracies[-1]:.2f}%")
