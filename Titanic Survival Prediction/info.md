# Titanic Survival Prediction - README

## Project Overview

This project focuses on predicting the survival of passengers aboard the Titanic based on various features provided in the dataset. The goal is to use machine learning to build a binary classification model that predicts whether a passenger survived (1) or not (0).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries Used](#libraries-used)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Model Training](#model-training)
7. [Model Prediction](#model-prediction)
8. [Predictive System](#predictive-system)
9. [Conclusion](#conclusion)
10. [How to Run](#how-to-run)

---

## Introduction

This project is part of the Data Science domain, where the goal is to predict whether a passenger on the Titanic survived or not. The Titanic dataset contains information like age, gender, class, and other details for passengers. The aim is to train a machine learning model using this dataset to predict survival outcomes.

---

## Libraries Used

The following Python libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computing.
- `matplotlib`: For data visualization.
- `seaborn`: For statistical data visualization.
- `sklearn`: For machine learning model building and evaluation.

---

## Dataset Description

The Titanic dataset used here contains 891 rows and 12 columns:

1. `PassengerId`: Unique ID for each passenger.
2. `Survived`: Binary variable indicating survival (1 for survived, 0 for not survived).
3. `Pclass`: Passenger class (1st, 2nd, 3rd).
4. `Name`: Name of the passenger.
5. `Sex`: Gender of the passenger.
6. `Age`: Age of the passenger.
7. `SibSp`: Number of siblings/spouses aboard.
8. `Parch`: Number of parents/children aboard.
9. `Ticket`: Ticket number.
10. `Fare`: Fare paid.
11. `Cabin`: Cabin number.
12. `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Data Preprocessing

1. **Handling Missing Values**:
   - The dataset contains missing values in the `Age`, `Cabin`, and `Embarked` columns.
   - The `Age` column is dropped as it has too many missing values, and the `Cabin` column is mostly empty.
   - The `Embarked` column has only 2 missing values, which are handled appropriately.

2. **Encoding Categorical Features**:
   - The `Sex` column is encoded using `LabelEncoder` to convert the categorical data into numerical values (male = 1, female = 0).

---

## Exploratory Data Analysis (EDA)

1. **Survival Distribution**:
   - The dataset contains 549 non-survivors and 342 survivors.
   
2. **Survival by Passenger Class**:
   - A bar chart shows the count of passengers by class and survival status, revealing that passengers in higher classes were more likely to survive.
   
3. **Survival by Gender**:
   - A bar chart shows the survival rate by gender. Females had a higher survival rate compared to males.

---

## Model Training

### 1. Feature Selection:
The selected features for model training are:
- `Pclass` (Passenger Class)
- `Sex` (Gender)

### 2. Splitting the Data:
Using the `train_test_split()` method, the dataset is split into training and testing sets with an 80/20 split.

### 3. Logistic Regression Model:
A logistic regression model is trained using the training set, as logistic regression is appropriate for binary classification problems like this one.

---

## Model Prediction

The model's predictions are made using the `predict()` method on the test dataset. The predicted values (0 for non-survivors and 1 for survivors) are compared with the actual values to evaluate the performance of the model.

Example output of predictions:
```python
[0 0 0 1 1 0 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 0 1 1 0 ... ]
```

---

## Predictive System

A simple predictive system is implemented to predict whether a passenger survived or not, based on their `Pclass` and `Sex`. The system uses the trained logistic regression model to make predictions.

Example prediction:
```python
res = log.predict([[1, 0]])  # Example input where 1st class and female (Sex = 0)
```

If the result is `0`, the output will be:
```
Unfortunate! Not survived.
```
If the result is `1`, the output will be:
```
Survived.
```

---

## Conclusion

The logistic regression model built in this project can predict the survival of Titanic passengers with basic features like class and gender. The model can be further improved by using additional features and more complex models, but it provides a solid foundation for binary classification tasks.

---

## How to Run

1. **Clone the Repository**:
   Clone the repository or download the code.

2. **Install Required Libraries**:
   Install the necessary libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the Script**:
   Execute the code in a Jupyter notebook or any Python environment.

4. **Dataset**:
   Ensure the Titanic dataset (`Titanic-Dataset.csv`) is present in the working directory or specify the correct file path.

---

This completes the README for the Titanic Survival Prediction project. Happy coding!
