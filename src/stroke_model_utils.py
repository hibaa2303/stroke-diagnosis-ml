import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import joblib
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BiLSTM Model
def load_model(model_path="./model_pkl/pca_ga_bilstm_model.pth", input_dim=256):
    class BiLSTMNet(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.fc = torch.nn.Linear(hidden_dim * 2, 2)

        def forward(self, x):
            x = x.unsqueeze(1)
            out, _ = self.lstm(x) 
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    model = BiLSTMNet(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Load CNN Models for Feature Extraction
def load_cnn_models():
    models_dict = {}

    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier = torch.nn.Sequential(*list(alexnet.classifier.children())[:-1])
    models_dict["alexnet"] = alexnet.to(DEVICE).eval()

    vgg = models.vgg19(pretrained=True)
    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
    models_dict["vgg"] = vgg.to(DEVICE).eval()

    inception = models.inception_v3(pretrained=True, aux_logits=True)
    inception.fc = torch.nn.Identity()
    inception.AuxLogits.fc = torch.nn.Identity()
    models_dict["inception"] = inception.to(DEVICE).eval()

    return models_dict

# Transform for model input
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

# Load PCA and GA index for inference
pca_model = joblib.load("./model_pkl/pca_model.pkl")
selected_indices = np.load("./model_pkl/ga_selected_indices.npy")

# Extract final features with PCA + GA
@torch.no_grad()
def extract_features(image_path, models_dict):
    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    features = []
    for model in models_dict.values():
        out = model(tensor)
        features.append(out.view(-1).cpu().numpy())
    raw_features = np.concatenate(features)
    pca_features = pca_model.transform([raw_features])[0]
    selected_features = pca_features[selected_indices]
    return selected_features