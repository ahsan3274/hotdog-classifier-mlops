import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.pyfunc

BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE).float()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
PATIENCE = 5
best_val_acc = 0
patience_counter = 0

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


train_losses, train_accuracies = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE).float(), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    y_true, y_pred = evaluate(model, test_loader)
    val_acc = accuracy_score(y_true, y_pred)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break


class WrappedResNet18(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        import torch.nn as nn
        import torchvision.models as models
        weights_path = context.artifacts["model_weights"]
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        self.model = model

    def predict(self, context, model_input):
        import torch
        import numpy as np
        arr = np.array(model_input, dtype=np.float32)
        tensor = torch.tensor(arr).view(-1, 3, 224, 224)
        with torch.no_grad():
            output = self.model(tensor)
            return torch.argmax(output, dim=1).numpy()


torch.save(model.state_dict(), "best_model.pt")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hotdog-classifier-resnet18-v3")

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WrappedResNet18(),
        artifacts={"model_weights": "best_model.pt"}
    )
    print("✅ PyFunc model logged and ready to serve.")

