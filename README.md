Stroke Diagnosis using Machine Learning

Description:

A machine learning-based system for predicting strokes from neuroimaging data and patient health indicators. This project aims to assist early stroke detection to improve medical response time and accuracy.

Project Objectives:

* Detect stroke presence using image and structured data.
* Apply preprocessing, feature engineering, and model training.
* Evaluate models and interpret results with visualizations.


Technologies Used:

* Python 3.10
* Scikit-learn
* TensorFlow / PyTorch 
* NumPy, Pandas
* Matplotlib, Seaborn

Folder Structure:

stroke-diagnosis-ml/
\|-- data/                -> Raw or preprocessed data
\|-- models/              -> Trained model files
\|-- src/                 -> Core Python scripts
\|-- utils/               -> Helper functions/utilities
\|-- requirements.txt     -> Python dependencies
\|-- README.md            -> Project overview

Setup Instructions:

1. Clone the Repository
   git clone [https://github.com/hibaa2303/stroke-diagnosis-ml.git](https://github.com/hibaa2303/stroke-diagnosis-ml.git)
   cd stroke-diagnosis-ml

2. Create a Virtual Environment
   python3 -m venv venv
   source venv/bin/activate   (on macOS/Linux)

3. Install Dependencies
   pip install -r requirements.txt



Model Overview:

* Uses patient health features like age, glucose, BMI, etc.
* Preprocessing includes missing value handling and normalization
* Models used: Logistic Regression, Random Forest, SVM
* Evaluation metrics: Accuracy, Precision, Recall, F1-score


