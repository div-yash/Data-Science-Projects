# Movie Rating Prediction with Python

## Overview

This project aims to build a model that predicts the rating of a movie based on various features such as **genre, director, actors, duration, and votes**. Using regression techniques, the goal is to create a model that analyzes historical movie data and estimates the rating given to a movie by users or critics.

By working on this project, you will gain hands-on experience in **data preprocessing**, **feature engineering**, **data visualization**, and **machine learning**. Additionally, it provides valuable insights into the factors that influence movie ratings.

## Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Data Visualization](#data-visualization)
6. [Feature Engineering](#feature-engineering)
7. [Model Building and Evaluation](#model-building-and-evaluation)
8. [Results](#results)
9. [How to Use](#how-to-use)
10. [Conclusion](#conclusion)
11. [License](#license)
12. [Contact](#contact)

---

## Project Description

In this project, we developed a machine learning model that predicts movie ratings based on several key factors such as:
- **Genre**
- **Director**
- **Actors**
- **Duration**
- **Votes**
  
We utilized **Linear Regression** to create the prediction model. The project includes:
- Cleaning and preprocessing the dataset.
- Exploratory data analysis (EDA) using visualization tools.
- Feature engineering to optimize input features for the model.
- Building and evaluating a machine learning model.

---

## Requirements

Before running the project, make sure the following libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

Alternatively, you can install all the required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Required Libraries
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` & `seaborn`: For data visualization.
- `plotly`: For interactive data visualizations.
- `scikit-learn`: For machine learning algorithms and metrics.

---

## Dataset

The dataset used in this project contains **15,509 records** and features such as:
- `Name`: The name of the movie.
- `Year`: The year of release.
- `Duration`: The runtime of the movie.
- `Genre`: The genre of the movie.
- `Rating`: IMDb rating.
- `Votes`: Number of votes the movie received.
- `Director`, `Actor 1`, `Actor 2`, `Actor 3`: Information about the director and leading actors.

The raw dataset is cleaned, preprocessed, and reduced to **5,659 rows** after handling missing values and duplicates.

---

## Data Preprocessing

### 1. Handling Missing Values
The dataset contained missing values across various columns. We dropped rows where critical data such as `Rating` or `Votes` were missing.

### 2. Data Cleaning
- Converted `Votes` and `Duration` to numerical formats.
- Cleaned `Year` by removing unnecessary characters like parentheses.
- Split the `Genre` column where multiple genres were listed and kept unique values.

### 3. Duplicates Removal
We identified and removed **6 duplicate rows** to ensure data integrity.

```python
movie_df.dropna(inplace=True)
movie_df.drop_duplicates(inplace=True)
```

---

## Data Visualization

Visualizing the data helped us understand the trends and patterns in movie ratings over the years.

### 1. Distribution of Ratings
We created a histogram to visualize the distribution of movie ratings across the dataset.

```python
rating=px.histogram(movie_df,x='Rating',histnorm='probability density',nbins=40)
rating.update_layout(title='Distribution of Rating')
rating.show()
```

### 2. Average Rating by Year
We explored the relationship between **Year** and **Rating** for the top genres using line plots.

```python
plot=px.line(year_avg_rating,x='Year',y='Rating',color='Genre')
plot.update_layout(title='Average Rating by Year for Top Genres')
plot.show()
```

---

## Feature Engineering

We created new features to improve model accuracy by grouping movies by **Genre**, **Director**, and **Actors** based on their average ratings. This allows the model to learn patterns from historical rating data.

```python
genre_mean_rating = movie_df.groupby('Genre')['Rating'].transform('mean')
movie_df['Genre_mean_rating']=genre_mean_rating

director_mean_rating=movie_df.groupby('Director')['Rating'].transform('mean')
movie_df['Director_encoded']=director_mean_rating

actor1_mean_rating=movie_df.groupby('Actor 1')['Rating'].transform('mean')
movie_df['Actor1_encoded']=actor1_mean_rating
```

---

## Model Building and Evaluation

### 1. Splitting Data
The dataset was split into training and testing sets:

```python
x=movie_df[['Year','Votes','Duration','Genre_mean_rating','Actor1_encoded','Actor2_encoded','Actor3_encoded']]
y=movie_df['Rating']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
```

### 2. Linear Regression Model
We built a **Linear Regression** model to predict the movie ratings.

```python
model = LinearRegression()
model.fit(x_train, y_train)
model_pred = model.predict(x_test)
```

### 3. Model Evaluation
We evaluated the model using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R2 Score**.

```python
print('Mean Squared error: ', mean_squared_error(y_test, model_pred))
print('Mean absolute error: ', mean_absolute_error(y_test, model_pred))
print('R2 score: ', r2_score(y_test, model_pred))
```

---

## Results

The **Linear Regression** model achieved the following performance metrics:
- **Mean Squared Error (MSE)**: 0.5403
- **Mean Absolute Error (MAE)**: 0.5377
- **R2 Score**: 0.714

These metrics indicate that the model is reasonably accurate at predicting movie ratings based on the provided features.

---

## How to Use

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install the Requirements**: Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Code**: Use a Python environment or Jupyter Notebook to run the code.
4. **Make Predictions**: Input your own movie data and use the trained model to predict movie ratings.

---

## Conclusion

This project demonstrates how data science techniques can be used to predict movie ratings based on features such as genre, director, and actors. By performing data preprocessing, feature engineering, and regression modeling, we successfully built a model with an **R2 score of 0.714**.

---

## License

This project is licensed under the MIT License.

---

## Contact

For further information or questions, feel free to contact:

- **Name**: Divya Prakash
- **Email**: [div13yash007@gmail.com]
