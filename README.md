❤️ Heart Disease Prediction using Machine Learning
📌 Project Overview

This project focuses on building a machine learning model to predict whether a person is likely to have heart disease based on medical attributes. The model is trained using Logistic Regression and implemented with NumPy, Pandas, and scikit-learn. The aim is to demonstrate the complete ML pipeline—from data collection and preprocessing to model evaluation and prediction.

🧠 Technologies & Libraries Used

Python

NumPy

Pandas

scikit-learn

Jupyter Notebook

📊 Dataset Description

The dataset (heart.csv) contains 1025 records and 14 columns, including one target variable.

Key Features:

Age

Sex

Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol (chol)

Fasting Blood Sugar (fbs)

Resting ECG (restecg)

Maximum Heart Rate Achieved (thalach)

Exercise-Induced Angina (exang)

ST Depression (oldpeak)

Slope of ST Segment (slope)

Number of Major Vessels (ca)

Thalassemia (thal)

Target Variable:

0 → No Heart Disease

1 → Heart Disease

⚙️ Data Preprocessing

Loaded the dataset using Pandas

Checked dataset shape, data types, and missing values

Performed statistical analysis using describe()

Verified that the dataset contains no null values

Split the data into features (X) and target (Y)

🔀 Train-Test Split

Training Data: 80%

Testing Data: 20%

Used stratify=Y to maintain class balance

Applied random_state=2 for reproducibility

🤖 Model Used

Logistic Regression

Chosen for its effectiveness in binary classification problems

Implemented using sklearn.linear_model.LogisticRegression

📈 Model Evaluation

The model was evaluated using accuracy score.

Training Accuracy: ~85.24%

Testing Accuracy: ~80.48%

This shows good generalization with minimal overfitting.

🔮 Predictive System

A predictive system was built that:

Accepts user input as medical parameters

Converts input into a NumPy array

Reshapes the data for a single prediction

Predicts whether the person has heart disease or not

Sample Output:

Person does not have a Heart Disease

The person has Heart Disease

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/heart-disease-prediction.git


Install required libraries

pip install numpy pandas scikit-learn


Run the Jupyter Notebook

jupyter notebook


Execute all cells to train the model and make predictions

📌 Future Improvements

Feature scaling using StandardScaler

Try other ML models (Random Forest, SVM, KNN)

Add precision, recall, F1-score, and confusion matrix

Deploy the model using Flask or Streamlit

👤 Author

Zaid Khan
B.Tech CSE (2nd Year)
Machine Learning Enthusiast
