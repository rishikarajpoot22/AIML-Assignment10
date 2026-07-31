import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv("heart.csv")

print(data.head())

print(data.isnull().sum())


# Features and target

X = data.drop("target", axis=1)
y = data["target"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X_train, y_train)


# Accuracy

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Accuracy:", accuracy)


# Save model

joblib.dump(
    model,
    "model.pkl"
)

print("Model saved successfully")