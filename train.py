import os
import cv2
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models.signature import infer_signature
import seaborn as sns
import joblib

# Config
DATA_DIR = "data"
TRAIN_HOTDOG = os.path.join(DATA_DIR, "train", "hot_dog")
TRAIN_NOT = os.path.join(DATA_DIR, "train", "not_hot_dog")
TEST_HOTDOG = os.path.join(DATA_DIR, "test", "hot_dog")
TEST_NOT = os.path.join(DATA_DIR, "test", "not_hot_dog")
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}


def load_images(folder, label):
    features, labels = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = hog(gray, **HOG_PARAMS)
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)


X_train_hot, y_train_hot = load_images(TRAIN_HOTDOG, 1)
X_train_not, y_train_not = load_images(TRAIN_NOT, 0)
X_test_hot, y_test_hot = load_images(TEST_HOTDOG, 1)
X_test_not, y_test_not = load_images(TEST_NOT, 0)

X_train = np.concatenate([X_train_hot, X_train_not])
y_train = np.concatenate([y_train_hot, y_train_not])
X_test = np.concatenate([X_test_hot, X_test_not])
y_test = np.concatenate([y_test_hot, y_test_not])


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Test accuracy: {accuracy:.4f}")


# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hotdog-classifier-log-reg-hog")

# Evaluate and compute metrics
preds = clf.predict(X_test)
val_acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
signature = infer_signature(X_test, preds)

# Confusion matrix plot
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Not Hotdog', 'Hotdog'], yticklabels=['Not Hotdog', 'Hotdog'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("conf_matrix.png")

# Log to MLflow
with mlflow.start_run() as run:
    mlflow.log_param("model", "HOG + LogisticRegression")
    mlflow.log_param("hog_orientations", HOG_PARAMS["orientations"])
    mlflow.log_param("image_size", IMAGE_SIZE)

    mlflow.log_metric("val_acc", val_acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact("conf_matrix.png")

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        signature=signature
    )

    print("✅ Model logged to run ID:", run.info.run_id)


joblib.dump(clf, "models/logreg_model.pkl")