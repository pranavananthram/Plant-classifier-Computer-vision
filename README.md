# Plant Leaf Disease Classification

A deep learning project that classifies plant leaf images into disease categories using a fine-tuned MobileNet model. The project includes data preprocessing, training, evaluation, Dockerized deployment, and automated CI/CD through GitHub Actions to AWS ECS.

---

## Dataset

This project uses the **New Plant Diseases Dataset (Augmented)** from Kaggle.

Dataset structure:

```
/kaggle/input/new-plant-diseases-dataset/
│── New Plant Diseases Dataset(Augmented)/
│── new plant diseases dataset(augmented)/
│── test/
```

---

## Features

* MobileNetV2-based leaf disease classifier
* Custom augmentation pipeline
* Training & validation with metrics
* Saves final model as `model.pth`
* Gradio UI for easy image-based inference
* Fully Dockerized application
* Automated deployment to AWS ECS using GitHub Actions

---

## Model Architecture

* **Backbone:** MobileNetV2
* **Input Size:** 224×224
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Data Augmentation:** Random crops, flips, rotations, color jitter, normalization

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

```bash
python train.py
```

This script will:
✔ Load and preprocess dataset
✔ Train MobileNetV2
✔ Save model as `model.pth`

---

## Running the Gradio App

```bash
python app.py
```

The app will be available at:

```
http://localhost:7860
```

---

## Docker Deployment

### Build image

```bash
docker build -t plant-leaf-app .
```

### Run container

```bash
docker run -p 7860:7860 plant-leaf-app
```

---

## AWS Deployment (ECS + GitHub Actions CI/CD)

This project includes a complete CI/CD pipeline using **GitHub Actions** to:

1. Build the Docker image
2. Log in to Amazon ECR
3. Push the image to ECR
4. Deploy the new image to an **ECS Fargate** service
5. Automatically refresh the running task

The workflow file (`.github/workflows/deploy.yml`) handles:

* ECR login
* Docker build & push
* ECS task definition render/update
* Automatic rollout of the new version

This enables **hands-free continuous deployment** on every push to `main`.

---

## Results

You can optionally add:

* Accuracy/loss graphs
* Confusion matrix
* Sample predictions

---

## Project Structure

```
.
├── train.py
├── app.py
├── model.pth
├── Dockerfile
├── requirements.txt
├── README.md
├── utils/
└── .github/
    └── workflows/
        └── deploy.yml
```

---

