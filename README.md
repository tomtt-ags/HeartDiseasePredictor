# HeartDiseasePredictor
This project predicts whether a patient has heart disease based on medical attributes using machine learning models. It is based on the UCI Heart Disease dataset
.

Project Overview

  Loads and inspects the dataset.
  
  Performs exploratory data analysis (EDA) with visualizations and correlation heatmaps.
  
  Preprocesses features using pipelines:
  
  Numerical features: imputation + scaling.
  
  Categorical features: imputation + one-hot encoding.
  
  Splits the dataset into training and testing sets.
  
  Trains and evaluates multiple models:
  
  Logistic Regression
  
  Random Forest
  
  Support Vector Machine (SVM)
  
  K-Nearest Neighbors (KNN)
  
  Evaluates performance with metrics like accuracy, precision, recall, F1-score, and confusion matrices.
  
  Requirements

  Python 3.x

Libraries:

  pandas, numpy
  
  matplotlib, seaborn
  
  scikit-learn
  
  kagglehub
  
  Install dependencies with:
  
  pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
  
  Usage
  
  Download the dataset automatically via KaggleHub.
  
  Run the script to:
  
  Explore dataset characteristics.
  
  Train and evaluate machine learning models.
  
  View performance reports and confusion matrices.

Results

  Logistic Regression, Random Forest, SVM, and KNN are compared.
  
  Classification reports and confusion matrices highlight the performance of each model.
  
  SVM confusion matrix is visualized for deeper insights.
