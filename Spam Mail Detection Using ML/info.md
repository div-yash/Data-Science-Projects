# Email Spam Detector using Python and Machine Learning

## Overview

This project aims to build an email spam detector using Python and machine learning techniques. By analyzing the text content of emails, the model is trained to classify emails as either "spam" or "ham" (non-spam). The project leverages machine learning algorithms, specifically Logistic Regression, to perform binary classification and achieve a high accuracy rate in detecting spam emails.

---

## Table of Contents

1. [Project Aim](#project-aim)
2. [Workflow](#workflow)
3. [Requirements](#requirements)
4. [Project Structure](#project-structure)
5. [Steps and Code Explanation](#steps-and-code-explanation)
    - Loading Data
    - Data Preprocessing
    - Train-Test Split
    - Feature Extraction
    - Training the Model
    - Model Evaluation
    - Predictive System
6. [Conclusion](#conclusion)

---

## Project Aim

The goal of this project is to use machine learning algorithms to detect spam emails. By processing and analyzing a dataset of emails, the model learns to distinguish between spam and ham emails, ultimately achieving an accuracy of over 96%.

---

## Workflow

1. **Mail Data**: The project begins by loading and exploring the dataset containing labeled email data (spam/ham).
2. **Data Preprocessing**: The data is cleaned by handling null values and encoding text labels into numerical values.
3. **Train-Test Split**: The data is split into training and testing sets to evaluate model performance.
4. **Feature Extraction**: Text data is transformed into numerical features using the TF-IDF Vectorizer.
5. **Logistic Regression Model**: The Logistic Regression algorithm is employed to classify emails into spam or ham.
6. **Trained Model**: The trained model is evaluated on both training and testing data to ensure reliable performance.

---

## Requirements

To successfully run this project, you will need the following Python libraries:

```bash
pip install pandas numpy scikit-learn
```

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning model and utilities.

---

## Project Structure

```bash
|-- Spam Email Detection.csv  # The dataset containing emails and labels
|-- spam_detector.py          # Python script containing the project code
```

---

## Steps and Code Explanation

### 1. Loading Data

The first step involves loading the email dataset into a pandas DataFrame. This dataset contains email text and labels (`spam` or `ham`).

```python
import pandas as pd
spam_ds = pd.read_csv('/content/Spam Email Detection.csv')
print(spam_ds.head())
```

### 2. Data Preprocessing

#### Handling Missing Values
The dataset contains columns with missing values. These are replaced with empty strings.

```python
mail_data = spam_ds.where(pd.notnull(spam_ds), '')
```

#### Label Encoding
Labels are encoded into numerical values:
- `ham` is encoded as 1.
- `spam` is encoded as 0.

```python
mail_data.loc[mail_data['v1'] == 'spam', 'v1'] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1'] = 1
```

### 3. Train-Test Split

The dataset is split into training and test sets, with 80% used for training and 20% for testing.

```python
from sklearn.model_selection import train_test_split

x = mail_data['v2']  # Email texts
y = mail_data['v1']  # Labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
```

### 4. Feature Extraction

We use **TF-IDF Vectorizer** to convert email texts into numerical features that can be fed into the Logistic Regression model.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
```

### 5. Training the Model

We train a **Logistic Regression** model on the feature vectors generated from the email texts.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_features, y_train)
```

### 6. Model Evaluation

We evaluate the model on both the training data and test data to check its accuracy.

```python
from sklearn.metrics import accuracy_score

# Prediction on training data
prediction_train = model.predict(x_train_features)
accuracy_train = accuracy_score(y_train, prediction_train)

# Prediction on test data
prediction_test = model.predict(x_test_features)
accuracy_test = accuracy_score(y_test, prediction_test)

print('Accuracy on training data:', accuracy_train)
print('Accuracy on test data:', accuracy_test)
```

**Result**: 
- Training accuracy: ~96%
- Test accuracy: ~96%

### 7. Predictive System

To predict whether a given email is spam or ham, the input email text is transformed using the TF-IDF vectorizer and passed through the trained model.

```python
input_mail = ["Don't worry. I guess he's busy"]
input_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

---

## Conclusion

This project demonstrates how machine learning can be applied to classify emails as spam or ham using logistic regression. The model achieves a high accuracy of 96% on both training and test data, making it a reliable spam detection system. 

