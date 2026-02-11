import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dataset_loader import get_dataset_path
import os
import pandas as pd

dataset_path = get_dataset_path()
print("Dataset Path:", dataset_path)

# Example: dataset files check
print(os.listdir(dataset_path))

# Example: CSV load (file name change karna pad sakta hai)
df = pd.read_csv(os.path.join(dataset_path, "Housing.csv"))
print(df.head())




# ----------------------------
# Convert Yes/No columns into 1/0
# ----------------------------
yes_no_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# ----------------------------
# Convert furnishingstatus into numeric using One-Hot Encoding
# ----------------------------
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test_scaled)

# ----------------------------
# Evaluation
# ----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------------------------
# Save Model and Scaler
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")   # important for Flask

print("Saved model.pkl, scaler.pkl and columns.pkl")