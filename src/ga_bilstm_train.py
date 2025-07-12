import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
import joblib

# Configuration
DATASET_DIR = "./dataset"
BATCH_SIZE = 16
EPOCHS = 50
KFOLD = 5
LEARNING_RATE = 0.0005
GA_SELECTED_FEATURES = 256
PCA_COMPONENTS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Preprocessing (grayscale + augmentation)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

# Dataset loading
def load_dataset():
    normal_dir = os.path.join(DATASET_DIR, "Normal")
    stroke_dir = os.path.join(DATASET_DIR, "Stroke")
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    stroke_images = [os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    normal_images = random.sample(normal_images, 950)
    stroke_images = stroke_images[:950]
    image_paths = normal_images + stroke_images
    labels = [0]*950 + [1]*950
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    return np.array(image_paths), np.array(labels)

# Feature extractors
def get_feature_extractors():
    models_dict = {}
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
    models_dict["alexnet"] = alexnet.to(DEVICE).eval()

    vgg = models.vgg19(pretrained=True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    models_dict["vgg"] = vgg.to(DEVICE).eval()

    inception = models.inception_v3(pretrained=True, aux_logits=True)
    inception.fc = nn.Identity()
    inception.AuxLogits.fc = nn.Identity()
    models_dict["inception"] = inception.to(DEVICE).eval()

    return models_dict

def extract_features(image_path, models_dict, train=True):
    transform = train_transform if train else val_transform
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    features = []
    with torch.no_grad():
        for model in models_dict.values():
            out = model(tensor)
            features.append(out.view(-1).cpu().numpy())
    return np.concatenate(features)

# BiLSTM
class BiLSTMNet(nn.Module):
    def __init__(self, input_dim=GA_SELECTED_FEATURES, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# GA with BiLSTM-based fitness
def genetic_feature_selector(X, y, num_selected=GA_SELECTED_FEATURES, generations=10, population_size=20):
    num_features = X.shape[1]
    population = np.random.randint(0, 2, size=(population_size, num_features))

    def fitness(individual):
        selected = individual.astype(bool)
        if selected.sum() == 0: return 0
        X_sel = X[:, selected]
        model = BiLSTMNet(input_dim=selected.sum()).float()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(5):
            inputs = torch.tensor(X_sel, dtype=torch.float32)
            targets = torch.tensor(y, dtype=torch.long)
            out = model(inputs)
            loss = loss_fn(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=1)
        return accuracy_score(y, preds.numpy())

    for _ in range(generations):
        scores = [fitness(ind) for ind in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:population_size // 2]]
        while len(population) < population_size:
            p1, p2 = random.sample(population[:10], 2)
            child = np.where(np.random.rand(num_features) > 0.5, p1, p2)
            mutation = np.random.rand(num_features) < 0.01
            child[mutation] = 1 - child[mutation]
            population.append(child)
    best = population[0]
    return np.where(best == 1)[0][:num_selected]

# Dataset
class StrokeDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.long)

# Training
def train():
    image_paths, labels = load_dataset()
    models_dict = get_feature_extractors()
    print("Extracting features...")
    raw_features = np.array([extract_features(p, models_dict, train=True) for p in tqdm(image_paths)])

    print("Applying PCA before GA...")
    pca = PCA(n_components=PCA_COMPONENTS)
    features = pca.fit_transform(raw_features)

    print("Selecting features using GA...")
    selected_indices = genetic_feature_selector(features, labels)
    features = features[:, selected_indices]
    joblib.dump(pca, "pca_model.pkl")
    np.save("ga_selected_indices.npy", selected_indices)

    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_loader = DataLoader(StrokeDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(StrokeDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = BiLSTMNet().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                preds = torch.argmax(model(X), dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Fold {fold+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        all_scores.append((acc, prec, rec, f1))

    all_scores = np.array(all_scores)
    print("\n=== Final Average Scores ===")
    print(f"Accuracy: {all_scores[:,0].mean():.4f}")
    print(f"Precision: {all_scores[:,1].mean():.4f}")
    print(f"Recall: {all_scores[:,2].mean():.4f}")
    print(f"F1 Score: {all_scores[:,3].mean():.4f}")
    torch.save(model.state_dict(), "pca_ga_bilstm_model.pth")
    print("Model saved as pca_ga_bilstm_model.pth")

if __name__ == "__main__":
    train() 