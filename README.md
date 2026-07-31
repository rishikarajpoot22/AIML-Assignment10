# Assignment-10: End-to-End Machine Learning Model Deployment using GitHub and Render

## Objective

The objective of this assignment is to develop an end-to-end machine learning application for predicting heart disease risk. The project includes data preprocessing, model training, model serialization, REST API development using Flask, and deployment using Render cloud platform.

---

## Dataset

**Heart Disease Prediction Dataset**

Kaggle Dataset Link:

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains clinical parameters of patients and is used to predict whether a patient is at risk of heart disease.

---

## Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- Gunicorn

---

## Methodology

The project was implemented using the following steps:

1. Loaded the heart disease dataset using Pandas.
2. Performed data exploration and checked missing values.
3. Separated input features and target variable.
4. Split the dataset into 80% training and 20% testing data.
5. Trained a Random Forest Classification model.
6. Evaluated the model using accuracy score.
7. Saved the trained model using Joblib.
8. Developed a REST API using Flask.
9. Deployed the Flask application using Render.

---

## Machine Learning Model

Algorithm Used:

**Random Forest Classifier**

The model was trained using patient clinical features to predict heart disease.

Model Evaluation:

- Accuracy Score: (Add your accuracy here)

---

## API Development

A Flask REST API was created to serve predictions.

### API Endpoint
