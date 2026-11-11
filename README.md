Predicting Ten-Year Coronary Heart Disease Risk with Machine Learning

Project Overview

This project presents an end-to-end Machine Learning approach for predicting the risk of developing Coronary Heart Disease (CHD) within ten years.

The project uses data from the Framingham Heart Study, consisting of 4,238 records and 15 clinical and behavioral features. The main objective is to develop and evaluate predictive models that can estimate an individual’s ten-year risk of coronary heart disease based on demographic information, medical history, lifestyle factors, and clinical measurements.

Project Objectives

* Develop a complete data preprocessing and machine learning pipeline.
* Handle missing values and investigate outliers.
* Perform exploratory data analysis and data visualization.
* Apply feature engineering and dimensionality reduction techniques.
* Compare multiple machine learning algorithms using cross-validation.
* Evaluate model performance using appropriate classification metrics.
* Interpret the final model and discuss its limitations.

Dataset Description

The dataset contains 4,238 records and 16 columns, including 15 predictive features and one target variable (TenYearCHD).

The features include:

* Demographic: Age, Sex, Education.
* Behavioral: Smoking Status, Cigarettes Per Day.
* Medical History: Blood Pressure Medication, Previous Stroke, Hypertension, Diabetes.
* Clinical Measurements: Total Cholesterol, Systolic Blood Pressure, Diastolic Blood Pressure, BMI, Heart Rate, and Glucose.

Methodology

The project follows a seven-step machine learning workflow:

1. Missing Value Handling.
2. Exploratory Data Analysis and Visualization.
3. Data Distribution Analysis and Transformation.
4. Feature Engineering.
5. Dimensionality Reduction and Feature Selection.
6. Train/Test Splitting.
7. Machine Learning Model Development and Evaluation.

Special attention was given to preventing data leakage by fitting preprocessing operations exclusively on training data.

Machine Learning Algorithms

The following algorithms were implemented and compared:

* Logistic Regression
* Random Forest
* Extra Trees
* HistGradientBoosting
* Support Vector Machine (SVM)
* XGBoost

Ensemble learning techniques, including Stacking and Soft Voting, were also evaluated.

Hyperparameter tuning was performed using RandomizedSearchCV and cross-validation.

Final Model and Results

Logistic Regression was selected as the final model due to its simplicity, interpretability, and probability calibration.

The final model achieved the following results on the held-out test dataset:

Evaluation Metric	Result
ROC-AUC	0.699
95% Confidence Interval	0.658–0.743
PR-AUC	0.305
Recall (Sensitivity)	0.53
Precision	0.24
Specificity	0.70
Accuracy	0.67

Technologies Used

* Python 3.12
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

Project Team

Prepared by:

* Kawakib Salih Mahdi
* Wahad Raad Falih
* Zeina Khalid Jaber

Supervised by:
Prof. Dr. Ali Gamal

Institution: Almaaqal International University (AMIU)
Faculty: Faculty of Engineering
Department: AI Engineering Department

Limitations and Disclaimer

This project is developed for academic and educational purposes. The model has limitations, including class imbalance, missing clinical variables, limited test-set size, and the absence of external validation.

This model is not intended for clinical diagnosis or medical decision-making. Its predictions should not be considered a substitute for professional medical evaluation.

Conclusion

This project demonstrates a complete machine learning workflow for predicting ten-year coronary heart disease risk. The experimental results show that Logistic Regression provides a simple and interpretable approach, while also highlighting the limitations of predictive performance when using the available dataset and features.
