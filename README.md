# Hotdog Classifier MLOps 🚀🌭

A practical, end-to-end machine learning pipeline—**from classic ML to deep learning**—showcasing how to take models from a notebook to a deployable web app using [Streamlit](https://streamlit.io/), [MLflow](https://mlflow.org/), and PyTorch/Scikit-learn.

---

## 🔗 [Try the App Live!](https://hotdog-classifier-mlops-unptu4sulukgp9rekyetef.streamlit.app/)

---

## 📚 Project Overview

- **Challenge:** Can you tell if a food image is a hotdog or not? (inspired by _Silicon Valley_ TV show)
- **Goal:** Demo a real MLOps workflow: training, experiment tracking, model packaging, and app deployment—all reproducibly and open-source.
- **Models included:**
  - **HOG + Logistic Regression** (scikit-learn)
  - **MobileNetV2** (PyTorch, transfer learning)
  - **ResNet18** (PyTorch, transfer learning + early stopping)

---

## 🗂️ Repo Structure

<details>
<summary>Click to expand</summary>

```
hotdog-classifier-mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── train_logreg.py                  # Classic ML: HOG + LogisticRegression
├── train_mobilenet.py               # MobileNetV2 (PyTorch)
├── train_resnet.py                  # ResNet18 (PyTorch)
├── streamlit_app.py                 # Streamlit inference web app (for deployment)
├── streamlit_multi_model_hotdog.py  # Local microservice REST demo
├── models/                          # Trained weights (.pt, .pkl)
│   ├── best_model.pt
│   ├── mobilenetv2_state.pt
│   └── logreg_model.pkl
├── data/                          # hotdog images (optional)
├── notebooks/                       
```
</details>

---

## 🚦 Quickstart

### 1. Clone the repo
```
git clone git@github.com:ahsan3274/hotdog-classifier-mlops.git
cd hotdog-classifier-mlops
```

### 2. Create and activate a virtual environment (recommended)
```
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download and prepare the dataset
- Download the dataset from Kaggle: https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
- Unzip so your project folder contains:

```
data/
  train/
    hot_dog/
    not_hot_dog/
  test/
    hot_dog/
    not_hot_dog/
```

- Place the `data/` directory at the root of this repository.

> **Note:** Dataset is not included in this repo for copyright and size reasons.

### 5. Run the Streamlit app
```
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏋️‍♂️ Training the Models (Optional)

Want to retrain or experiment?
- Classic ML:  
  ```
  python train.py
  ```
- MobileNetV2:  
  ```
  python train_v2.py
  ```
- ResNet18:  
  ```
  python train_v3.py
  ```

Trained weights will be saved in the `/models` folder.

---

## ⚙️ MLOps Features

- MLflow for experiment tracking, logging, and reproducibility
- Streamlit for fast, interactive deployment
- Multiple model support with easy model switching
- Reproducible preprocessing (no more “works on my notebook”!)

---

## 📝 Dataset

This project uses the open dataset "Hot Dog - Not Hot Dog" from Kaggle.

**How to use:**

1. Download from https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
2. Unzip and place as:

```
data/
  train/
    hot_dog/
    not_hot_dog/
  test/
    hot_dog/
    not_hot_dog/
```

3. Place the `data/` directory in the root of this project.

---

## 🖥️ Streamlit App Files Explained

This project provides two main Streamlit apps:

### 1. streamlit_app.py   🌐 Recommended for Deployment

- Purpose: Main inference app for deployment on Streamlit Cloud or locally.
- How it works: Loads all models and allows model selection via dropdown.
- Run with:
  ```
  streamlit run streamlit_app.py
  ```

### 2. streamlit_multi_model_hotdog.py   🧪 Advanced Local Use

- Purpose: Local frontend for models served as separate microservices.
- Requires each model to run on its own port as a backend server.
- Run with:
  ```
  streamlit run streamlit_multi_model_hotdog.py
  ```

---

**Summary Table:**

| File                             | For Cloud Deployment? | For Local/Advanced Use? | Model Selection | Inference Logic        |
|----------------------------------|------------------------|--------------------------|------------------|-------------------------|
| streamlit_app.py                 | ✅                     | ✅                       | Dropdown         | In-app (single process) |
| streamlit_multi_model_hotdog.py  | ❌                     | ✅                       | Dropdown         | REST API to model ports |

---

## 🙋 FAQ

**Q: Why are there .pt and .pkl files in `/models`?**  
A: `.pt` = PyTorch weights; `.pkl` = scikit-learn logistic regression model.

**Q: How do I add my own images?**  
A: Use the file uploader in the Streamlit app.

**Q: Can I retrain or add new models?**  
A: Absolutely! Fork, edit, and submit PRs.

---

## 📢 For Students

- Fork this repo and try new architectures or augmentations.
- Use MLflow to compare your experiments.
- Ask questions or share results on Discord: `nasquamesse3274`
- Contributions welcome — open a pull request!

---

## 👨‍💻 Credits

- Ahsan Tariq: https://huggingface.co/spaces/ahsan3274/personal-portfolio  
- LinkedIn: https://www.linkedin.com/in/ahsan-32-tariq/  
- Email: ahsanntariq@protonmail.com  
- Workshop: 0xMLOps, Young-AI Leaders Linz JKU Linz, June 2025

---

## ⭐️ License

Open source for learning, research, and fun.

---

_Built for the next generation of MLOps builders!_
