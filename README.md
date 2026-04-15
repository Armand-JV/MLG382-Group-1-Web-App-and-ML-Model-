# Diabetes Risk Segmentation & Decision Support System

An end-to-end machine learning project that predicts diabetes risk, segments patients into meaningful lifestyle-based groups, and provides explainable insights through an interactive web dashboard.

This project was developed for the **BC Analytics** health-tech case study using the **Diabetes_and_Lifestyle_Dataset_** and follows the **CRISP-DM** framework.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Methodology](#project-methodology)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Explainability](#explainability)
- [Deployment](#deployment)
- [Deliverables](#deliverables)
- [Team Collaboration](#team-collaboration)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---
## How to run.
Use the any terminal (in VScode or not doesn't matter)
Make sure your directory is set to MLG382-Group-1-Web-App-and-ML-Model-\app> 

So run:

cd app

python app.py

Then go to the url: http://127.0.0.1:8050/
---

## Project Overview

Healthcare providers often need better tools to identify diabetes risk early, understand the lifestyle factors driving that risk, and support data-driven decisions during patient consultations. This project addresses that need by combining predictive modeling, clustering, and explainable AI into a single decision support system. The final result is an interactive dashboard that helps users explore risk categories, patient segments, and actionable recommendations. :contentReference[oaicite:1]{index=1}

---

## Problem Statement

The goal of this project is to build a system that can:

1. **Classify patients into diabetes risk categories**
2. **Identify key lifestyle and health factors influencing risk**
3. **Group patients into meaningful lifestyle-based segments**
4. **Present insights through a user-friendly dashboard**

---

## Objectives

- Predict the **diabetes stage** for a patient using machine learning
- Compare multiple classification models to identify the best performer
- Segment patients using **k-means clustering**
- Use **SHAP values** to explain both risk classification and segmentation
- Deploy the solution as an **interactive Dash web application**

---

## Dataset

This project uses the **Diabetes_and_Lifestyle_Dataset_** provided for the course. The target variable is **`diabetes_stage`**, and the project requires careful domain understanding of diabetes types and risk patterns before modeling. :contentReference[oaicite:2]{index=2}

> Note: The dataset should be cleaned, explored, and prepared before model training. Feature engineering and preprocessing may be applied depending on data quality.

---

## Project Methodology

This project follows the **CRISP-DM** framework:

### 1. Business Understanding
- Understand the healthcare problem
- Define the project goals
- Identify what insights are needed for decision support

### 2. Data Understanding
- Load and inspect the dataset
- Explore feature distributions
- Study the target variable `diabetes_stage`
- Identify missing values, outliers, and correlations

### 3. Data Preparation
- Clean and transform the dataset
- Encode categorical variables
- Scale numerical features where needed
- Split data for training and validation

### 4. Modeling
- Train classification models:
  - Decision Tree
  - Random Forest
  - XGBoost
- Build a clustering model using:
  - K-Means (`k=3`)

### 5. Evaluation
- Compare model performance using relevant metrics
- Assess cluster quality and interpretability
- Use SHAP to understand important features

### 6. Deployment
- Deploy the trained model(s) in a **Dash** application
- Enable interactive exploration of predictions and patient segments

---

## Tech Stack

### Data Analysis
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost
- K-Means Clustering

### Explainability
- SHAP

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Web Application
- Dash
- Dash Bootstrap Components

### Development & Collaboration
- Git
- GitHub

### Deployment
- Render

### Environment Management
- Virtualenv or Conda
- `requirements.txt`

---

## Features

- Diabetes risk prediction
- Lifestyle-based patient segmentation
- SHAP-based feature importance interpretation
- Interactive dashboard for exploration
- Model comparison for classification performance
- Clean project structure for GitHub submission
- Version-controlled workflow with meaningful commits

---

## Repository Structure

```bash
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── clustering.py
│   ├── explainability.py
│   └── utils.py
├── app/
│   ├── app.py
│   ├── layout.py
│   └── callbacks.py
├── models/
│   ├── classification_model.pkl
│   └── clustering_model.pkl
├── reports/
│   ├── figures/
│   └── technical_report.pdf
├── documentation/
│   ├── domain_understanding.docx
│   └──example.doxc
├── requirements.txt
├── README.md
└── .gitignore
