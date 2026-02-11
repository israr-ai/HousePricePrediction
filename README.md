# 🏠 House Price Prediction Web App (Flask + ML)
![Uploading Screenshot 2026-02-11 162624.png…]()
![Uploading Screenshot 2026-02-11 162712.png…]()

This is a Machine Learning project that predicts house prices based on user input.

## 🚀 Features
- Predict house price using trained ML model
- Simple and clean UI
- Flask-based deployment
- Model trained using Scikit-learn

## 📂 Project Structure
HousePricePrediction/
│── app.py
│── train_model.py
│── model.pkl
│── scaler.pkl
│── requirements.txt
│── templates/
│ └── index.html
│── static/
└── style.css


## 🧠 ML Algorithm Used
- Linear Regression

## ⚙️ Installation Steps

### 1️⃣ Create Virtual Environment
```bash
python -m venv env
```
### 2️⃣ Activate Environment
# Windows (PowerShell)
```
.\env\Scripts\activate

windows(CMD)..
env\Scripts\activate

linux/Mac...
source env/bin/activate
```

## 3️⃣ Install Requirements
```
pip install -r requirements.txt

```

📌 Train the Model

Make sure dataset file is available in project folder:

Housing.csv


Now run:

python train_model.py


After training, it will generate:

model.pkl

scaler.pkl

▶️ Run Flask App
python app.py


Then open browser:

http://127.0.0.1:5000/

📝 Inputs

The user provides the following inputs:

Area (sq ft)

Bedrooms

Bathrooms

Stories

Parking

Main Road (Yes/No)

Air Conditioning (Yes/No)

Furnishing Status

🎯 Output

Predicted House Price (₹)

📊 Visualization

The web app includes a chart:

Area vs Price Chart using Chart.js

📦 Requirements

Main Libraries Used:

Flask

Pandas

NumPy

Scikit-learn

Joblib

👨‍💻 Author

Israr Shaikh
📌 GitHub: https://github.com/israr-ai


⭐ Future Improvements

Add feature importance chart

Add multiple ML models (RandomForest, XGBoost)

Improve UI design

Deploy on Render / Railway / HuggingFace Spaces
