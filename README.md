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
├── train_logreg.py        # Classic ML: HOG + LogisticRegression
├── train_mobilenet.py     # MobileNetV2 (PyTorch)
├── train_resnet.py        # ResNet18 (PyTorch)
├── streamlit_app.py       # Streamlit inference web app
├── models/                # Trained weights (.pt, .pkl)
│   ├── best_model.pt
│   ├── mobilenetv2_state.pt
│   └── logreg_model.pkl
├── images/                # Demo/example images (optional)
├── notebooks/             # EDA or exploratory notebooks (optional)
```
</details>

---

## 🚦 Quickstart

### 1. **Clone the repo**
```bash
git clone https://github.com/yourusername/hotdog-classifier-mlops.git
cd hotdog-classifier-mlops
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Download and prepare the dataset**
- [Download the dataset from Kaggle](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) (Kaggle account required)
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

> **Dataset is not included in this repo for copyright and size reasons.**
> Please respect [Kaggle's Terms of Use](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog/rules).

### 4. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```
- Then open [localhost:8501](http://localhost:8501) in your browser.

### 5. **Try out the models**
- Upload a food image (`.jpg`/`.png`)
- Select a model from the dropdown
- See if it's a 🌭 or 🚫!

---

## 🏋️‍♂️ Training the Models (Optional)

Want to retrain or experiment?
- **Classic ML:**  
  ```bash
  python train_logreg.py
  ```
- **MobileNetV2:**  
  ```bash
  python train_mobilenet.py
  ```
- **ResNet18:**  
  ```bash
  python train_resnet.py
  ```
- Trained weights will be saved in the `/models` folder.

---

## ⚙️ MLOps Features

- **MLflow** for experiment tracking, logging, and reproducibility
- **Streamlit** for fast, interactive deployment
- **Multiple model support** with easy model switching
- **Reproducible preprocessing** (no more “works on my notebook”!)

---

## 📝 Dataset

This project uses the open dataset [Hot Dog - Not Hot Dog](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) from Kaggle.

**How to use:**

1. [Download from Kaggle](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) (you need a Kaggle account)
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

> **Dataset is not included for copyright and size reasons.**

---

## 🙋 FAQ

**Q: Why are there .pt and .pkl files in `/models`?**  
A: `.pt` = PyTorch weights; `.pkl` = scikit-learn (logistic regression) model.

**Q: How do I add my own images?**  
A: Use the file uploader in the Streamlit app.

**Q: Can I retrain or add new models?**  
A: Absolutely! Fork, edit, and submit PRs.

---

## 📢 For Students

- Fork this repo and try new architectures or augmentations.
- Use MLflow to compare your experiments.
- Ask questions or share results at [your Discord/Slack/group]!
- Contributions welcome—open a pull request for improvements.

---

## 👨‍💻 Credits

- [Your Name] ([your LinkedIn](#) / [your email](#))
- Workshop/Event: 0xMLOps, JKU Linz, June 2025

---

## ⭐️ License

Open source for learning, research, and fun.  
See `LICENSE` file for details.

---

_Built for the next generation of MLOps builders!_
