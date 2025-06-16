import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np



BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"  # expects data/train and data/test


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


base_model = torchvision.models.mobilenet_v2(pretrained=True)
for param in base_model.features.parameters():
    param.requires_grad = False  # optional: freeze feature extractor

base_model.classifier[1] = nn.Linear(base_model.last_channel, NUM_CLASSES)
base_model = base_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=1e-3)

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hotdog-classifier-mobilenet-v2")


with mlflow.start_run() as run:
    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    for epoch in range(EPOCHS):
        base_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        
        val_acc = evaluate(base_model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Final evaluation
    final_val_acc = evaluate(base_model, test_loader)
    mlflow.log_metric("final_val_accuracy", final_val_acc)
    
    torch.save(base_model.state_dict(), "mobilenetv2_state.pt")
    print("✅ Torch model state_dict saved.")

    class MobileNetV2MLflowWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            import torchvision
            import torch.nn as nn
            # Rebuild the model and load the state_dict
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
            state_dict = torch.load(context.artifacts["torch_model_state"], map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(self.device)
            self.model = model

        def predict(self, context, model_input):
            import torch
            # Accept both numpy arrays and DataFrames
            if hasattr(model_input, "values"):
                x = model_input.values
            else:
                x = model_input
            # Ensure float32 and torch tensor
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            # Add batch dimension if single image
            if x.ndim == 3:
                x = x.unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(x)
                _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

    artifacts = {"torch_model_state": "mobilenetv2_state.pt"}
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=MobileNetV2MLflowWrapper(),
        artifacts=artifacts,
    )
    print("✅ Model logged to MLflow with pyfunc wrapper.")

run_id = run.info.run_id
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")






