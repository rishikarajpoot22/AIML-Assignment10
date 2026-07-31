from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)


model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Heart Disease Prediction API"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array(
        list(data.values())
    ).reshape(1,-1)


    prediction = model.predict(features)


    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"


    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)