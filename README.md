# Student Dropout Prediction with MLflow and Databricks

Predict student outcomes (Graduate, Dropout, Enrolled) using enrollment and first-semester academic features.

## Overview

This project demonstrates an end-to-end ML workflow on Databricks:

- Data ingestion from Delta tables
- Feature engineering and preprocessing
- Model training and evaluation
- Experiment tracking with MLflow
- Batch inference pipeline and production-ready predictions

We compare two feature sets:

1. **Enrollment-only features** – Predict dropout risk at admission.
2. **Enrollment + First Semester features** – Improve predictions using early academic performance.

## Notebooks

1. **01_training_and_experiments.ipynb**  
   - EDA, feature preprocessing, and model training
   - Logistic Regression & Random Forest experiments
   - MLflow tracking of parameters, metrics, and registered models
   - Confusion matrices and learning curve analysis
   - Hyperparameter tuning using GridSearchCV

2. **02_inference_pipeline.ipynb**  
   - Batch inference using the MLflow Champion model
   - Delta table output for downstream dashboards
   - Logging of prediction counts and sample outputs

## Key Features

- MLflow experiment tracking and model registry
- Stratified train/test split to preserve class balance
- Pipelines with `StandardScaler` (numerical) and `OneHotEncoder` (categorical)
- Production-ready batch inference workflow
- Visualizations for class-level performance analysis

## Results Summary

| Model | Feature Set | Accuracy | Macro F1 |
|-------|------------|---------|----------|
| Logistic Regression | Enrollment | 0.590 | 0.554 |
| Random Forest       | Enrollment | 0.663 | 0.531 |
| Logistic Regression | First Semester | 0.714 | 0.672 |
| Random Forest       | First Semester | 0.745 | 0.635 |

**Insight:** First-semester academic data significantly improves dropout prediction. Logistic Regression provides more balanced predictions across classes, which is important for early-warning systems.

---

## About

This project was developed as a Databricks ML workflow demonstrating:

- How to track ML experiments and models with **MLflow**
- How to evaluate multiclass classification performance
- How to implement a **reproducible batch inference pipeline**
- Techniques for early-warning predictions in educational datasets
