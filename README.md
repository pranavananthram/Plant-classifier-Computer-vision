# Plant Leaf Disease Classification

A deep learning project that classifies plant leaf images into disease categories using a fine-tuned MobileNet model. The project includes data preprocessing, training, evaluation, and deployment using a Docker-packaged Gradio app.

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

* MobileNetV2-based classifier
* Custom augmentation pipeline
* Training & validation loops with accuracy/loss tracking
* Model saving in `.pth` format
* Gradio web interface for predictions
* Dockerized for easy deployment on AWS / Cloud / Local

---

## Model Architecture

* **Backbone:** MobileNetV2
* **Input Size:** 224×224
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Augmentations:**

  * RandomResizedCrop
  * RandomHorizontalFlip
  * RandomRotation
  * ColorJitter
  * Normalization

---

## 🛠 Installation

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

```python
python train.py
```

This script will:
✔ Load dataset
✔ Train MobileNetV2
✔ Save `model.pth`

---

## Running the Gradio App

```bash
python app.py
```

The app will launch a web UI where you can upload leaf images and get predictions.

---

## Docker Deployment

Build the image:

```bash
docker build -t plant-leaf-app .
```

Run the container:

```bash
docker run -p 7860:7860 plant-leaf-app
```

Your Gradio app will now be accessible at:

```
http://localhost:7860
```

---

## Cloud Deployment 

* Push the Docker image to **ECR**
* Deploy via **AWS Fargate**, **EC2**, or **Lightsail Container Service**
* Ensure security group allows port **7860**

---

## Results

Include training accuracy, validation curves, confusion matrix, etc. (optional)

---

## Project Structure

```
.
├── train.py
├── app.py
├── model.pth
├── requirements.txt
├── Dockerfile
├── README.md
└── utils/
```

---

## Contributions

Pull requests are welcome! For major changes, please open an issue first.

---

