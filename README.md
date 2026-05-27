# Ames Housing Price Prediction

<p align="center">
  <img src="images/01-readme-main.png" alt="Ames Housing Price Prediction project banner" width="100%">
</p>

<p align="center">
  A machine learning regression project focused on predicting house sale prices using the Ames Housing dataset.
</p>

## 1. Project Description

This project is my implementation of a machine learning workflow for predicting house sale prices using the Ames Housing dataset.

The idea behind the project was not only to train a regression model, but also to understand the main factors that influence residential property prices. For that reason, I started with a detailed exploratory data analysis before moving into feature engineering, preprocessing, model comparison and final evaluation.

The target variable is **SalePrice**, and the model uses property-related features such as overall quality, living area, basement size, garage capacity, neighborhood, house age, remodeling information and other structural characteristics.

I organized the work into two main notebooks. The first notebook focuses on EDA, where I analyzed the target distribution, missing values, outliers and the relationship between the main features and sale price. The second notebook focuses on modelling, where I built a preprocessing pipeline, created new features, compared different regression models and tuned the best-performing one.

After comparing multiple models with cross-validation, Gradient Boosting was selected as the final model. It provided the best balance between predictive performance and business interpretation, since its most important features were aligned with what would be expected in the real estate market: overall quality, total property size, number of bathrooms, house age, garage capacity and quality-related variables.

The final model, metrics and test predictions were saved as artifacts so the results can be reviewed and the project can be extended later.

## 2. Project Stack

This project was developed using Python in Jupyter Notebook, with Visual Studio Code as the development environment and Git/GitHub for version control.

The main libraries used were:

- **Pandas and NumPy** for data manipulation and numerical operations.
- **Matplotlib and Seaborn** for exploratory data visualization.
- **Scikit-Learn** for preprocessing, pipelines, cross-validation, model training, hyperparameter tuning and evaluation.
- **Joblib** for saving the final trained model.

The regression models tested included Ridge Regression, ElasticNet, Random Forest, Extra Trees and Gradient Boosting Regressor.

## 3. Business Problem and Project Objective

### 3.1 What is the business problem?

In the real estate market, estimating the correct sale price of a property is a challenging task. A house price depends on many factors, such as location, overall quality, living area, age, garage capacity, basement features, number of bathrooms and other structural characteristics.

Incorrect pricing can create problems for different stakeholders. Sellers may lose money if a property is undervalued, while overpriced houses may stay on the market for longer. Buyers also need reliable price references to avoid overpaying, and real estate professionals can benefit from data-driven tools to support pricing decisions.

The business problem of this project is: **how can we estimate the final sale price of a house based on its characteristics?**

### 3.2 What is the context?

House pricing is usually influenced by a combination of physical, qualitative and location-related attributes. In this project, the Ames Housing dataset was used to analyze historical house sales and understand which features are most associated with **SalePrice**.

Some important factors considered in this context include:

- **Property quality:** overall material and finish quality.
- **Property size:** total square footage, living area and basement area.
- **Location:** neighborhood and surrounding characteristics.
- **Age and renovation:** year built and remodeling information.
- **Property functionality:** garage capacity, bathrooms, fireplace, basement and other amenities.

By using historical data, machine learning can help identify patterns that are not always obvious through manual analysis. The goal is not to replace a professional appraisal, but to create a model that can support better pricing analysis and provide a data-driven estimate of house value.

### 3.3 Which are the project objectives?

The main objectives of this project are:

- Understand the main factors associated with house sale prices.
- Perform exploratory data analysis to identify patterns, missing values, outliers and feature engineering opportunities.
- Build a supervised machine learning regression model to predict **SalePrice**.
- Compare different regression algorithms using cross-validation.
- Select and tune the best-performing model.
- Evaluate the final model on an untouched test set.
- Interpret the model results through feature importance.
- Save the final model, metrics and predictions as project artifacts.

### 3.4 Which are the project benefits?

This project can provide several benefits, such as:

- More consistent house price estimation.
- Better understanding of the main price drivers in the dataset.
- Support for buyers, sellers and real estate professionals during pricing analysis.
- Reduced reliance only on intuition or manual comparisons.
- A reproducible machine learning workflow that can be improved or deployed in the future.

As a result, the project creates a data-driven approach to estimate house prices and understand which property characteristics have the strongest influence on the final sale value.

### 3.5 Conclusion

When using the final model, the main objective is to generate a numerical price estimate for each property based on its characteristics.

For this type of problem, predicting a continuous value is more useful than assigning houses into broad price categories, because it gives a more practical estimate of the expected sale price. This can help support pricing decisions, compare similar properties and identify whether a house may be underpriced or overpriced compared to the patterns learned from historical data.

However, the model should be used as a decision-support tool, not as a definitive appraisal system. Real estate prices can also be affected by external factors that are not present in the dataset, such as market demand, interest rates, local economic conditions, school quality, crime rates and negotiation dynamics.

Even with these limitations, the project demonstrates how exploratory data analysis, feature engineering and machine learning can be combined to build a strong and interpretable regression workflow for house price prediction.

## 4. Solution Pipeline

The project followed a structured machine learning workflow inspired by the CRISP-DM framework.

The main steps were:

1. Define the business problem and project objective.
2. Load the Ames Housing dataset and get a general overview of the data.
3. Split the dataset into training and test sets before deeper analysis, reducing the risk of data leakage.
4. Perform exploratory data analysis on the training data.
5. Analyze missing values, outliers, numerical features and categorical variables.
6. Create new features based on property characteristics, such as house age, remodeling status, total property size and total number of bathrooms.
7. Build preprocessing pipelines for numerical and categorical variables.
8. Train and compare multiple regression models using cross-validation.
9. Tune the best-performing model.
10. Evaluate the final selected model on the untouched test set.
11. Interpret the model results using feature importance.
12. Save the final model, evaluation metrics and test predictions as project artifacts.

Each step is explained in detail inside the notebooks, including the reasoning behind the main decisions made during the project.

## 5. Dataset Overview

The dataset used in this project is the Ames Housing dataset, which contains detailed information about residential properties sold in Ames, Iowa.

The original dataset has **2,930 rows and 82 columns**. The target variable is **SalePrice**, which represents the final sale price of each house. The remaining variables describe different property characteristics, such as lot area, living area, overall quality, year built, garage information, basement features, neighborhood and other structural attributes.

In the initial data inspection, no duplicated rows were found and the target variable did not contain missing values. However, several explanatory variables had missing values. In this dataset, some of these missing values do not necessarily mean data errors; in many cases, they indicate the absence of a specific property feature, such as no garage, no basement, no fireplace or no masonry veneer.

The dataset contains both numerical and categorical variables, which makes preprocessing an important part of the project. Numerical features require imputation and scaling, while categorical features require missing value treatment and encoding before being used by machine learning models.

<div align="center">

<table>
  <tr>
    <th>Item</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Dataset</td>
    <td>Ames Housing</td>
  </tr>
  <tr>
    <td>Rows</td>
    <td>2,930</td>
  </tr>
  <tr>
    <td>Columns</td>
    <td>82</td>
  </tr>
  <tr>
    <td>Target variable</td>
    <td><code>SalePrice</code></td>
  </tr>
  <tr>
    <td>Problem type</td>
    <td>Supervised regression</td>
  </tr>
  <tr>
    <td>Numerical columns</td>
    <td>39, including the target variable</td>
  </tr>
  <tr>
    <td>Categorical columns</td>
    <td>43</td>
  </tr>
  <tr>
    <td>Duplicated rows</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Missing values in target</td>
    <td>0</td>
  </tr>
</table>

</div>

