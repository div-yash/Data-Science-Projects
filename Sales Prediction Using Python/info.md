# Sales Prediction Using Python

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Aim](#project-aim)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Data Exploration and Visualization](#data-exploration-and-visualization)
   1. [Dataset Statistics](#dataset-statistics)
   2. [Pair Plot](#pair-plot)
   3. [Heatmap of Correlation](#heatmap-of-correlation)
6. [Building a Sales Prediction Model](#building-a-sales-prediction-model)
   1. [Data Preparation](#data-preparation)
   2. [Training the Model](#training-the-model)
   3. [Model Evaluation](#model-evaluation)
7. [Model Insights](#model-insights)
8. [Improvements and Next Steps](#improvements-and-next-steps)
9. [Conclusion](#conclusion)

---

## Project Overview
Sales prediction is a critical task for businesses that helps them anticipate customer demand and optimize their advertising strategies. In this project, we forecast product sales using machine learning techniques based on advertising expenditure data from various channels like TV, radio, and newspapers. The insights gathered will help businesses understand how their advertising spend influences sales, which is crucial for maximizing return on investment (ROI).

Sales prediction can provide valuable information such as:
- **Optimizing advertising budgets**: Identifying the most effective platforms for ads.
- **Understanding customer behavior**: Analyzing trends to forecast future sales.
- **Revenue forecasting**: Helping businesses plan for upcoming financial periods.

## Project Aim
The primary objective of this project is to develop a machine learning model that accurately predicts the number of units sold for a given product based on advertising expenditures across multiple platforms such as TV, radio, and newspapers. This prediction helps businesses in decision-making regarding future advertising investments.

**Key Factors in Sales Prediction**:
- **Advertising expenditure**: How much is spent on different media (TV, Radio, Newspapers).
- **Target audience**: Segmenting the audience based on location, preferences, etc.
- **Platform selection**: Choosing the right mix of advertising channels (e.g., focusing more on TV or digital platforms).

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **NumPy**: Provides support for mathematical operations on large, multi-dimensional arrays.
  - **Pandas**: Facilitates data manipulation and analysis, particularly for structured data.
  - **Matplotlib** and **Seaborn**: Used for data visualization to help understand patterns and correlations.
  - **scikit-learn**: A robust library used for building machine learning models such as Linear Regression.

## Dataset
The dataset used for this project contains information about advertising expenditures and corresponding sales. It has four columns:
1. **TV**: Amount spent on TV advertising.
2. **Radio**: Amount spent on radio advertising.
3. **Newspaper**: Amount spent on newspaper advertising.
4. **Sales**: The number of units sold.

Here’s a sample of the dataset:

| TV      | Radio   | Newspaper | Sales |
|---------|---------|-----------|-------|
| 230.1   | 37.8    | 69.2      | 22.1  |
| 44.5    | 39.3    | 45.1      | 10.4  |
| 17.2    | 45.9    | 69.3      | 12.0  |
| 151.5   | 41.3    | 58.5      | 16.5  |
| 180.8   | 10.8    | 58.4      | 17.9  |

The dataset consists of 200 rows. We will use this dataset to develop and evaluate our sales prediction model.

## Data Exploration and Visualization

### Dataset Statistics
To understand the general distribution and summary of the data, we begin by looking at basic statistical measures such as mean, standard deviation, and minimum/maximum values for each feature.

```python
sales_df.describe()
```

|        | TV      | Radio    | Newspaper | Sales   |
|--------|---------|----------|-----------|---------|
| count  | 200.0   | 200.0    | 200.0     | 200.0   |
| mean   | 147.04  | 23.26    | 30.55     | 15.13   |
| std    | 85.85   | 14.85    | 21.77     | 5.28    |
| min    | 0.7     | 0.0      | 0.3       | 1.6     |
| max    | 296.4   | 49.6     | 114.0     | 27.0    |

### Key Observations:
1. The highest advertising expenditure is on **TV** with a maximum of 296.4.
2. The minimum amount spent on **Radio** and **Newspaper** can be as low as 0.
3. The **Sales** range from 1.6 to 27 units, with an average of 15 units.
4. **TV advertising** expenditure has a strong impact on sales, as indicated by visual and statistical analysis.

### Pair Plot
A pair plot is used to visualize the relationships between each feature in the dataset, particularly between advertising spending and sales.

```python
sns.pairplot(sales_df)
```

This provides insights into the relationships between features, helping us determine if linear models are suitable.

### Heatmap of Correlation
We generate a correlation matrix to identify how each advertising channel correlates with sales.

```python
sns.heatmap(sales_df.corr(), annot=True)
plt.show()
```

The heatmap shows that **TV** advertising has a higher correlation with **Sales** than **Radio** and **Newspaper**, making it a strong candidate for model training.

---

## Building a Sales Prediction Model

### Data Preparation
We start by splitting the dataset into training and testing sets. In this project, we begin by only using **TV** expenditure as a feature, but later, we will expand the model to include other variables (Radio and Newspaper).

```python
from sklearn.model_selection import train_test_split

# Using TV feature for initial training
x_train, x_test, y_train, y_test = train_test_split(sales_df[['TV']], sales_df[['Sales']], test_size=0.3, random_state=0)
```

### Training the Model
We use a simple linear regression model to predict sales based on advertising expenditure on TV:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

### Model Evaluation
After training the model, we use the test data to generate predictions and evaluate the model's performance. We also visualize the fit between predicted sales and actual sales data.

```python
mod = model.predict(x_test)

plt.scatter(x_test, y_test)
plt.plot(x_test, model.intercept_ + model.coef_ * x_test, color='red')
plt.show()
```

The scatter plot shows the relationship between TV expenditure and sales, with the regression line indicating the predicted values.

### Model Coefficients and Intercept
The coefficients and intercept of the linear model help in understanding the effect of TV advertising on sales:

```python
model.coef_  # Coefficient for TV
model.intercept_  # Intercept of the model
```

This indicates that for each additional unit of currency spent on TV advertising, sales are expected to increase by `model.coef_` units.

---

## Model Insights
1. **Sales Correlation**: TV advertising has the highest correlation with sales, implying that increasing the TV advertising budget can result in a significant sales boost.
2. **Linear Regression Performance**: The linear regression model fits the data reasonably well for TV advertising but can be improved by including other features like radio and newspaper spending.
3. **Model Coefficients**: The coefficient of the linear regression model shows how much sales increase per unit of advertising expenditure.

---

## Improvements and Next Steps
While the initial model uses only TV advertising data, more improvements can be made by incorporating other variables and exploring advanced techniques:

1. **Expand the Model**:
   - Include `Radio` and `Newspaper` as features in the model to account for the combined impact of all advertising platforms.
   - Use multiple linear regression instead of simple linear regression.

2. **Hyperparameter Tuning**:
   - Tune the parameters of the model to improve performance and reduce prediction errors.

3. **Cross-Validation**:
   - Use k-fold cross-validation to ensure that the model generalizes well to unseen data.

4. **Try Different Models**:
   - Explore other regression models like Ridge, Lasso, or Decision Trees to see if they provide better predictions.

---

## Conclusion
This project demonstrates how linear regression can be applied to predict sales based on advertising expenditure. Our findings suggest that TV advertising has the most significant impact on sales, followed by radio and newspapers. By including all available features and fine-tuning the model, businesses can better forecast sales and optimize their advertising strategy, leading to better resource allocation and maximized ROI.

This is a foundational model, and expanding it with more features and techniques will further improve its accuracy and usefulness for real-world business applications.
