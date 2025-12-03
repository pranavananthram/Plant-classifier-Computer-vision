import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ----- DEVICE -----
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- MODEL -----
num_classes = 38  # <-- replace with your number of classes

# Load MobileNetV2 and replace the classifier
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Load your trained weights
model.load_state_dict(torch.load("mobilenetv2_plant.pth", map_location=device))
model.to(device)
model.eval()

# ----- IMAGE TRANSFORMS -----
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# ----- CLASS NAMES -----
# Replace this list with the actual classes from your dataset in order
class_names = [
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Potato___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Strawberry___Leaf_scorch",
    "Peach___healthy",
    "Apple___Apple_scab",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Apple___Black_rot",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Apple___Cedar_apple_rust",
    "Tomato___Target_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Strawberry___healthy",
    "Apple___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Esca_(Black_Measles)",
    "Raspberry___healthy",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___healthy"
]

# ----- PREDICTION FUNCTION -----
def predict(img):
    img = Image.fromarray(img)
    x = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # Return dict of class probabilities
    return {class_names[p]: float(probs[0][p]) for p in range(len(class_names))}

# ----- GRADIO APP -----
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5),
    title="Plant Leaf Classifier"
)

# Expose app to Docker
demo.launch(server_name="0.0.0.0", server_port=7860)
