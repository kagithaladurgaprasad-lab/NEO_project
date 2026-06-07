# NEO_project
# ☄️ Near-Earth Object (NEO) Hazard Prediction System

## Overview

This project is an end-to-end Machine Learning application that predicts whether a Near-Earth Object (NEO) is potentially hazardous to Earth. The system leverages supervised machine learning algorithms, hyperparameter optimization with Optuna, experiment tracking using MLflow, and an interactive Streamlit web application for real-time predictions.

The project demonstrates a complete MLOps workflow, including data preprocessing, model training, model comparison, experiment tracking, model persistence, and deployment.

---

## Features

* Automated data preprocessing pipeline
* Multiple machine learning algorithms

  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Support Vector Machine (SVM)
  * Gaussian Naive Bayes
  * Random Forest
  * Gradient Boosting
* Hyperparameter tuning using Optuna
* Experiment tracking with MLflow
* Model performance comparison and ranking
* Model serialization using Joblib
* Interactive Streamlit dashboard
* Real-time asteroid hazard prediction

---

## Dataset

The project uses NASA Near-Earth Object (NEO) data containing asteroid characteristics such as:

* Estimated Minimum Diameter
* Estimated Maximum Diameter
* Relative Velocity
* Miss Distance
* Absolute Magnitude
* Orbiting Body
* Sentry Object Status

### Target Variable

* `hazardous`

  * 1 → Hazardous Asteroid
  * 0 → Non-Hazardous Asteroid

---

## Technology Stack

### Programming Language

* Python

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-Learn

### Hyperparameter Optimization

* Optuna

### Experiment Tracking

* MLflow

### Model Persistence

* Joblib

### Web Application

* Streamlit

---

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Train-Test Split
6. Hyperparameter Optimization using Optuna
7. Cross Validation
8. Model Training
9. Model Evaluation
10. Model Selection
11. Model Saving
12. Experiment Tracking with MLflow
13. Streamlit Deployment

---

## Model Evaluation Metrics

The following metrics are used to evaluate model performance:

* Cross Validation Accuracy
* Training Accuracy
* Testing Accuracy
* Training Time
* Prediction Time
* Model Size

---

## Project Structure

```text
project/
│
├── app.py
├── neo.csv
├── model.pkl
├── requirements.txt
├── notebooks/
├── models/
├── mlruns/
└── README.md
```

---

## Running the Project

### Clone Repository

```bash
git clone <repository-url>
cd project
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Environment

```bash
myenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch MLflow

```bash
mlflow ui
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

---

## Streamlit Application

The Streamlit dashboard allows users to:

* Enter asteroid characteristics
* Analyze hazard levels in real time
* Receive hazard classification predictions
* Visualize prediction outcomes through an interactive interface

---

## Future Enhancements

* Docker Deployment
* CI/CD Pipeline Integration
* Cloud Deployment (AWS/Azure/GCP)
* Model Monitoring
* Automated Retraining
* Feature Importance Visualization
* Explainable AI (SHAP)

---

## Author

Durga Prasad

Machine Learning | Data Science | MLOps | Generative AI Enthusiast

---

## License

This project is intended for educational and research purposes.
