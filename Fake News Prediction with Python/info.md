Here's a detailed README file for your Fake News Detection project using Logistic Regression:

---

# Fake News Detection Using Logistic Regression

This project aims to detect whether a news article is real or fake based on its content. It utilizes a **Logistic Regression** model for binary classification, as the task involves classifying news into two categories: **real** or **fake**.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Dependencies](#dependencies)
5. [Code Explanation](#code-explanation)
    - [Data Pre-processing](#data-pre-processing)
    - [Stemming](#stemming)
    - [Converting Text to Numeric Data](#converting-text-to-numeric-data)
    - [Model Training](#model-training)
6. [How to Run the Project](#how-to-run-the-project)
7. [Conclusion](#conclusion)

---

## Overview
The Fake News Detection project uses machine learning to classify news articles as either real or fake. The model is trained using logistic regression, which is well-suited for binary classification tasks.

---

## Dataset
The dataset used in this project is a collection of news articles with the following features:

- **id**: A unique identifier for each news article.
- **title**: The title of the news article.
- **author**: The author of the news article.
- **text**: The body of the article (which might be incomplete).
- **label**: The target variable, where:
  - `0` represents **real news**.
  - `1` represents **fake news**.

---

## Project Workflow

1. **Data Collection**: Load the dataset from a CSV file containing labeled news articles.
2. **Data Pre-processing**: Clean the text data by removing unnecessary characters, stemming, and eliminating stopwords.
3. **Train-Test Split**: Split the dataset into training and testing sets.
4. **Feature Extraction**: Convert textual data into numerical features using **TF-IDF Vectorizer**.
5. **Model Training**: Train the **Logistic Regression** model on the training set.
6. **Prediction and Evaluation**: Use the trained model to predict whether news articles are fake or real and evaluate the performance using accuracy metrics.

---

## Dependencies

Before running the project, ensure that you have the following Python libraries installed:

```bash
pip install numpy pandas nltk scikit-learn
```

---

## Code Explanation

### 1. Data Pre-processing

#### Importing Dependencies
```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

- **NumPy** and **Pandas**: Used for data handling.
- **NLTK**: Used for text pre-processing, including removing stopwords and stemming.
- **Scikit-learn**: Used for machine learning model training and evaluation.

#### Load and Inspect Dataset
```python
news_dataset = pd.read_csv('/content/train.csv')
```

- The dataset is loaded into a Pandas DataFrame for easy manipulation.

#### Handle Missing Values
```python
news_dataset = news_dataset.fillna('')
```

- Replace any missing values in the dataset with empty strings to avoid errors in later stages.

#### Combine Author and Title
```python
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
```

- Combine the **author** and **title** columns into a new column called `content`, which will be used for the classification task.

### 2. Stemming

Stemming reduces words to their base or root form. For example, words like *actor*, *actress*, and *acting* will all be reduced to *act*.

```python
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

news_dataset['content'] = news_dataset['content'].apply(stemming)
```

- **Regular Expressions (re)**: Used to filter out non-alphabetical characters.
- **PorterStemmer**: Applied to convert each word to its root form.
- **Stopwords**: Common words like "the", "is", "and" are removed, as they don't contribute significantly to model accuracy.

### 3. Converting Text to Numeric Data

We use **TF-IDF Vectorizer** to convert the text into numeric data:

```python
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(news_dataset['content'].values)
```

- **TF-IDF (Term Frequency-Inverse Document Frequency)** assigns weights to words based on their frequency in the document and across documents. It helps the model identify important words in the dataset.

### 4. Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(x, news_dataset['label'], test_size=0.2, stratify=y, random_state=2)
```

- The dataset is split into training and testing sets in an 80-20 ratio, ensuring a balanced distribution of fake and real news using the `stratify` parameter.

### 5. Model Training

We use **Logistic Regression** for binary classification:

```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

- The model is trained on the training set using the **fit()** method.

### 6. Model Evaluation

To evaluate the model's performance, we calculate the accuracy score on both the training and testing datasets:

```python
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
```

- The accuracy score measures how well the model is performing in terms of predicting the correct labels for both real and fake news.

---

## How to Run the Project

1. Clone the repository or download the code files.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the code using Jupyter Notebook, Google Colab, or any Python environment.
4. Make sure the dataset (CSV file) is located in the correct path as specified in the code.
5. Execute the cells to see the model training and evaluation process in action.

---

## Conclusion

This project demonstrates a complete machine learning pipeline to detect fake news using a **Logistic Regression** model. The workflow includes:
- Text pre-processing (stemming, removing stopwords).
- Converting text into numerical form using **TF-IDF**.
- Training and evaluating a binary classification model.

---

