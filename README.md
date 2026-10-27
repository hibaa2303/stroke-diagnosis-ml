# 🧠 Stroke Diagnosis Using Machine Learning

A machine learning project focused on stroke diagnosis using neuroimaging data and patient health indicators. It explores data preprocessing, model development, and evaluation through machine learning techniques.

## ✨ Features

- Data preprocessing and feature engineering
- Machine learning model development and evaluation
- Analysis of stroke-related data
- Streamlit-based application interface

## 🛠️ Technologies Used

- Python 3.11
- NumPy and Pandas
- Scikit-learn
- PyTorch and Torchvision
- Matplotlib and Seaborn
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

- `app.py` — Streamlit application
- `src/` — Source code and helper modules
- `model_pkl/` — Saved model files
- `dataset/` — Dataset files, if included
- `images/` — Project images
- `requirements.txt` — Python dependencies

## 🚀 Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/hibaa2303/stroke-diagnosis-ml.git
cd stroke-diagnosis-ml
```

### 2. Create a virtual environment

Python 3.11 is recommended.

```bash
python3.11 -m venv venv311
```

### 3. Activate the environment

**macOS / Linux**

```bash
source venv311/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal to access the application.

## 📝 Notes

- Ensure the required model files are available in `model_pkl/`.
- Dataset availability may depend on the files included in the repository.
- Dependency compatibility may vary by operating system.

## 👩‍💻 Author

**Hiba Aayesha**

GitHub: [hibaa2303](https://github.com/hibaa2303)
