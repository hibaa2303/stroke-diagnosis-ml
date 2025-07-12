import streamlit as st
import torch
import numpy as np
from PIL import Image
import uuid
import os
import random
import shutil
from src.stroke_model_utils import extract_features, load_cnn_models, load_model
from torchvision import transforms
from  src.styles import TITLE_STYLE, SIDEBAR_STYLE

st.set_page_config(
    page_title="Stroke Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Variables
UPLOAD_DIR = "temp_stroke_upload"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load Model and CNNs
@st.cache_resource
def setup():
    model = load_model("./model_pkl/pca_ga_bilstm_model.pth")
    cnn_models = load_cnn_models()
    return model, cnn_models

model, cnn_models = setup()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

def preprocess_image(image):
    image = transform(image).unsqueeze(0)
    return image

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">Stroke Detection Using Neuroimages Application</h1>', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"

    if st.sidebar.button("Exploratory Data Analysis"):
        st.session_state['page'] = "Exploratory Data Analysis"

    if st.sidebar.button("Models Used"):
        st.session_state['page'] = "Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

#############################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image("./images/home_image.webp", width=700)
        st.markdown("""
                    ## 🧠 Problem Statement
                    Stroke is one of the leading causes of mortality and long-term disability globally. Early and accurate diagnosis plays a vital role in reducing brain damage, improving patient outcomes, and optimizing treatment strategies such as thrombolysis or surgical intervention.

                    Traditionally, stroke diagnosis relies heavily on manual inspection of brain CT (Computed Tomography) scans by expert radiologists. However, this manual process is time-consuming, error-prone, and highly dependent on the skill and experience of the clinician — especially in rural or under-resourced hospitals where rapid diagnosis is critical but expertise is limited.

                    Moreover, subtle signs of stroke in early CT images are often difficult to detect due to factors such as noise, overlapping tissues, low contrast, or patient movement. These limitations necessitate the development of automated, intelligent systems that can assist medical professionals in detecting stroke accurately, rapidly, and consistently.

                    This project addresses this gap by designing a deep learning-based system that leverages multiple convolutional neural networks (CNNs), genetic algorithms (GA), and a bidirectional long short-term memory (BiLSTM) classifier to identify and classify stroke conditions in CT brain scans.

                    ## 🎯 Project Objective
                    The primary goal of this project is to build a hybrid AI-based diagnostic pipeline that can automatically detect stroke from CT brain images with high accuracy and reliability. This is accomplished by integrating feature extraction, dimensionality reduction, evolutionary feature selection, and temporal sequence learning into a unified workflow.

                    ## 🔑 Key Objectives:

                    🧠 Feature Extraction from Neuroimages: Use three pre-trained CNNs — AlexNet, VGG19, and InceptionV3 — to extract robust features from brain CT images.

                    📉 Dimensionality Reduction: Apply Principal Component Analysis (PCA) to compress and denoise the high-dimensional CNN feature vectors.

                    🧬 Genetic Algorithm-based Feature Selection: Use a biologically-inspired GA to evolve and select the most informative feature subset from PCA-reduced vectors.

                    🔁 Sequence Learning via BiLSTM: Train a BiLSTM model that learns temporal dependencies and interactions between selected features for final classification (Stroke vs. Normal).

                    📊 Cross-validation Evaluation: Achieve and report strong performance using stratified 5-fold cross-validation, targeting a final accuracy close to 96.5%, as referenced in the research literature.

                    💾 Model Deployment in Web App: Deploy the complete model pipeline in a user-friendly Streamlit web interface that allows radiologists or healthcare workers to upload a CT scan and instantly receive diagnostic feedback.

                    ## 🏥 Why This Project Matters
                    ⚡ Faster Diagnosis: Enables faster decision-making in emergency settings where stroke timing is critical.

                    🩺 Clinical Decision Support: Assists doctors and radiologists with a second opinion based on learned patterns from a curated dataset.

                    🌍 Scalable and Accessible: Can be deployed in clinics or hospitals where expert radiologists are not always available.

                    🧠 Interpretability Ready: The modular nature of CNN-GA-BiLSTM enables transparency and fine-grained insight into how decisions are made.
        """)

    elif st.session_state['page'] == "Project Data Description":

        st.markdown("""
                ## 📂 Dataset Description: Brain Stroke Neuro Images
                    
                ### 🧠 Overview
                The Brain Stroke CT Image Dataset is a curated collection of brain CT scan images used to distinguish between normal and stroke-affected brain conditions. Unlike synthetic or simulated datasets, this collection comprises real CT scans that replicate the diagnostic challenges faced in emergency departments and hospitals. The dataset aims to support deep learning research in early stroke detection using non-invasive imaging techniques.

                ## 🧾 Key Characteristics
                
                #### Feature Description:
                    
                🧠 Total Images	1,900 CT brain scans
                    
                📊 Classes	2 classes: Stroke and Normal
                
                🧍 Patients	100+ real-world patients aged 16 years and older
                
                📏 Image Dimensions	650 × 650 pixels (original), converted to 227 × 227 for training
                
                🌈 Channels	Grayscale → stacked to 3-channel RGB for CNN compatibility
                
                🧪 Image Type	Non-contrast CT (NCCT), common in acute stroke screening
                
                📂 Format	JPG / PNG
                
                🧰 Labeling Type	Classification (Binary: Stroke vs. Normal)
                
                🧬 Data Composition
                
                ##### 950 Normal CT Scans:

                Collected from patients without neurological conditions.

                Used as control group images for training the model to distinguish healthy brains.

                ##### 950 Stroke CT Scans:

                Captured from patients confirmed to have ischemic or hemorrhagic stroke.

                Includes variations in stroke severity, region, and size.

                ### ⚙️ Preprocessing Pipeline
                To prepare the raw CT scans for training in the deep learning pipeline, the following steps were performed:

                ### 🧼 Grayscale Normalization:

                Original images were in 1-channel grayscale format.

                Converted to 3-channel grayscale by channel stacking for compatibility with pre-trained CNNs.

                ### 📐 Image Resizing:

                Resized from 650 × 650 to 227 × 227 pixels to match input dimensions of AlexNet, VGG19, and InceptionV3.

               ### 🔁 Data Augmentation (during training):

                - Horizontal flips

                - Random rotations (±10°)

                - Brightness jitter

                S- imulates variation in scan quality and angle to boost model robustness.

                🔢 Label Encoding:

                - 0: Normal

                - 1: Stroke

                ### ✅ Why This Dataset Matters
                Realistic Medical Imaging: Derived from actual CT scans, not artificial or synthetic datasets.

                Early Diagnosis Potential: Enables learning from subtle visual cues in early-stage strokes.

                Balanced Class Distribution: 950 Normal + 950 Stroke scans ensure fair model learning.

                Modality-Specific: Optimized for CT (rather than MRI), which is more widely available and faster in emergency care.
            """)

    elif st.session_state['page'] == "Sample Training Data":
        st.markdown("### 🧠 Sample Brain CT Images (Normal vs Stroke)")

    # Refresh button
        if st.button("🔄 Click here to see different Images"):
            st.rerun()

        normal_dir = os.path.join("./dataset", "Normal")
        stroke_dir = os.path.join("./dataset", "Stroke")

        if os.path.exists(normal_dir) and os.path.exists(stroke_dir):
            normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png'))]
            stroke_images = [os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) if f.endswith(('.jpg', '.png'))]

            if normal_images and stroke_images:
                # Pick min of both to pair evenly
                display_count = min(len(normal_images), len(stroke_images), 12)
                normal_samples = random.sample(normal_images, display_count)
                stroke_samples = random.sample(stroke_images, display_count)

                cols = st.columns(4)
                for i in range(display_count):
                    col = cols[i % 4]
                    col.markdown("**🟢 Normal:**")
                    col.image(Image.open(normal_samples[i]), use_container_width=True)
                    col.markdown("**🔴 Stroke:**")
                    col.image(Image.open(stroke_samples[i]), use_container_width=True)
            else:
                st.warning("No images found in one or both subfolders: Normal / Stroke.")
        else:
            st.warning("Folders 'Normal' and 'Stroke' not found inside ./sample_ct_images/")

    elif st.session_state['page'] == "Data Preprocessing":
        st.markdown("""
            ## 🛠️ Data Preprocessing Steps for Stroke Detection
            
            ### 🔍 1️⃣ Dataset Acquisition and Curation
            - The dataset used in this project was sourced from a publicly available Kaggle repository containing **CT brain scan images** for **stroke and normal** cases.
            - The images are realistic, non-contrast CT scans typically used in emergency rooms for initial stroke assessment.
            - The original images were in **grayscale** with dimensions **650 × 650 pixels**.

            ### 🧾 2️⃣ Class-Based Folder Organization
            The dataset was manually curated and organized into class-wise directories to enable structured loading and stratified sampling:

            - **Normal/**: Contains 950 images of patients without stroke.
            - **Stroke/**: Contains 950 images from patients diagnosed with ischemic or hemorrhagic stroke.

            Folder Structure:
            ```
            dataset/
            ├── Normal/
            └── Stroke/
            ```

            ### 🧼 3️⃣ Image Preprocessing
            The following image transformations were applied:

            a. **Grayscale Normalization**
            - Original 1-channel grayscale images were **stacked into 3 channels** to maintain compatibility with pre-trained CNNs (AlexNet, VGG19, Inception).

            b. **Image Resizing**
            - All images were resized from **650×650** to **227×227** pixels — the standard input size for the selected CNN architectures.

            c. **Tensor Conversion**
            - The images were converted into **PyTorch tensors** and normalized to the [0, 1] range.

            ### 🧪 4️⃣ Data Augmentation
            To enhance model generalization and robustness, the following augmentations were applied during training:

            - 🔁 **Horizontal Flipping**: Simulates patient positioning variability.
            - 🔃 **Random Rotation (±10°)**: Emulates slight CT scan angle variations.
            - 💡 **Brightness Jitter**: Mimics differences in scan intensity or lighting conditions.
            
            These augmentations were applied using **PyTorch's `transforms` module**, ensuring real-time variability during training.

            ### 🧠 5️⃣ Feature Extraction from CNNs
            The preprocessed images were passed through three pre-trained CNNs:
            
            - **AlexNet**
            - **VGG19**
            - **InceptionV3**
            
            The outputs of the final FC layers (before softmax) were concatenated to create a **comprehensive feature vector** for each image.

            ### 📉 6️⃣ Dimensionality Reduction (PCA)
            - The concatenated features were reduced to **512 principal components** using **PCA (Principal Component Analysis)**.
            - This step removes noise and redundancy, boosting both training speed and GA effectiveness.

            ### 🧬 7️⃣ Feature Selection via Genetic Algorithm
            - A **Genetic Algorithm (GA)** was used to evolve and select the **top 256 most discriminative features**.
            - The GA used a lightweight **BiLSTM-based fitness function** to assess feature subsets by their classification accuracy.

            ### 🔁 8️⃣ Stratified K-Fold Cross Validation
            - A **5-fold Stratified K-Fold** approach was applied to ensure balanced stroke/normal class distribution in each split.
            - Prevents overfitting and ensures generalization across multiple training rounds.

            ### 🧠 9️⃣ Label Encoding
            - `0` → **Normal**
            - `1` → **Stroke**
            
            Labels were one-hot encoded during training for classification tasks.

            ### ✅ Summary

            - ✔️ Realistic brain CT images
            - ✔️ Robust preprocessing and augmentation
            - ✔️ Feature fusion from multiple CNNs
            - ✔️ GA-based selection for optimal input to BiLSTM
            - ✔️ Fully integrated with a modular PyTorch pipeline

            This preprocessing pipeline ensures that the model is trained on **diverse**, **normalized**, and **informatically-rich** representations of CT neuroimages, enabling highly accurate stroke detection performance.
        """)

    elif st.session_state['page'] == "Exploratory Data Analysis":
        st.write("Pre-trained Inception V3 Architecture:")
        st.image("./images/Screenshot from 2025-03-25 01-12-44.png", use_container_width=True)
        
        st.write("Pre-trained VGG 19 Architecture:")
        st.image("./images/Screenshot from 2025-03-25 01-12-57.png", use_container_width=True)

        st.write("NASNet-Large Pre-trained Architecture NASNet-Large:")
        st.image("./images/Screenshot from 2025-03-25 01-13-14.png", use_container_width=True)

        st.write("BILSTM Architecture:")
        st.image("./images/Screenshot from 2025-03-25 01-13-34.png", use_container_width=True)

    elif st.session_state['page'] == "Models Used":
        st.markdown("""
            ## 🧠 Model Pipeline and Architecture: GA + BiLSTM

            ### 🔍 Model Overview

            This project integrates **three key components** to build a hybrid deep learning pipeline for **stroke classification** from CT images:

            1. 🧠 **Feature Extraction** from multiple pre-trained CNNs
            2. 🧬 **Genetic Algorithm (GA)** for optimal feature selection
            3. 🔁 **BiLSTM Classifier** to learn temporal dependencies in selected features

            The architecture is inspired by the paper *“Innovations in Stroke Identification: A Machine Learning-Based Diagnostic Model Using Neuroimages”*, which achieved **96.5% accuracy** using this method.

            ---

            ### ⚙️ Model Pipeline (Step-by-Step)

            #### 1️⃣ CNN-Based Feature Extraction
            - Input CT brain images (resized to 227x227, 3-channel grayscale)
            - Images passed through **three pre-trained CNNs**:
            - **AlexNet**
            - **VGG19**
            - **InceptionV3**
            - Each CNN’s final layer features are **extracted and concatenated** into a single high-dimensional vector (~6,000+ features)

            #### 2️⃣ Dimensionality Reduction using PCA
            - To eliminate redundant and noisy features, **Principal Component Analysis (PCA)** is applied
            - Reduces dimensionality to **512 components**
            - Enhances training efficiency and model generalization

            #### 3️⃣ Genetic Algorithm-Based Feature Selection
            - A **Genetic Algorithm (GA)** is used to evolve a population of feature subsets
            - A lightweight **BiLSTM model** is used as the fitness evaluator
            - The GA selects the **top 256 most discriminative features**

            #### 4️⃣ Classification with BiLSTM
            - A **Bidirectional Long Short-Term Memory (BiLSTM)** network is trained on the selected features
            - Captures temporal dependencies and directional patterns in the feature sequence
            - Outputs binary classification: **Stroke** or **Normal**

            ---

            ### 🧬 BiLSTM Architecture Details

            | Component       | Configuration                      |
            |------------------|--------------------------------------|
            | Input Layer      | 256 features (from GA output)       |
            | LSTM Units       | 128 hidden units (bidirectional)    |
            | Dropout Layer    | 0.3 (to prevent overfitting)        |
            | Fully Connected  | Linear layer → 2 output classes     |
            | Activation       | Softmax (via CrossEntropyLoss)      |
            | Optimizer        | Adam (learning rate = 0.0005)       |
            | Epochs           | 50                                  |
            | Batch Size       | 16                                  |

            ---

            ### 📊 Cross-Validation Strategy

            - Model is evaluated using **Stratified 5-Fold Cross-Validation**
            - Ensures balanced class distribution in every fold
            - Reduces the risk of overfitting and ensures robust performance

            ---

            ### ✅ Reported Results

            | Metric      | Value (Average across 5 folds) |
            |-------------|-------------------------------|
            | Accuracy    | **96.45%**                   |
            | Precision   | 98.00%                       |
            | Recall      | 93.50%                       |
            | F1-Score    | 96.00%                       |

            

            ---

            ### 🧠 Why BiLSTM?

            BiLSTM is chosen over traditional classifiers (SVM, RF, ANN) because:
            - It learns **sequence-level feature dependencies**
            - Captures **bidirectional context** of spatial patterns
            - Performs better with **reduced and evolved feature sets**

            ---

            ✅ This architecture is scalable and deployable on web interfaces, making it suitable for integration into **clinical decision support systems** for rapid stroke screening.
        """)



    elif st.session_state['page'] == "Model Predictions":
        st.markdown("### 🔍 Upload a Brain CT Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
            image = Image.open(uploaded_file).convert("RGB")
            image.save(file_path)

            # Feature extraction
            features = extract_features(file_path, cnn_models)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prediction = model(features)
            label = torch.argmax(prediction, dim=1).item()

            col1, col2 = st.columns(2)
            with col1:
                st.write("### 📥 Uploaded Image")
                st.image(image, caption="Uploaded CT Image", width=350)
            with col2:
                st.write("### 🎯 Detection Result")
                if label == 0:
                    st.success("🧠 Prediction: Normal")
                else:
                    st.error("⚠️ Prediction: Stroke Detected")

            shutil.rmtree(UPLOAD_DIR)

if __name__ == "__main__":
    main()
