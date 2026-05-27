# house-prices-regression

## 1. Description

This project is my implementation of a machine learning workflow for predicting house sale prices using the Ames Housing dataset.

The idea behind the project was not only to train a regression model, but also to understand the main factors that influence residential property prices. For that reason, I started with a detailed exploratory data analysis before moving into feature engineering, preprocessing, model comparison and final evaluation.

The target variable is **SalePrice**, and the model uses property-related features such as overall quality, living area, basement size, garage capacity, neighborhood, house age, remodeling information and other structural characteristics.

I organized the work into two main notebooks. The first notebook focuses on EDA, where I analyzed the target distribution, missing values, outliers and the relationship between the main features and sale price. The second notebook focuses on modelling, where I built a preprocessing pipeline, created new features, compared different regression models and tuned the best-performing one.

After comparing multiple models with cross-validation, Gradient Boosting was selected as the final model. It provided the best balance between predictive performance and business interpretation, since its most important features were aligned with what would be expected in the real estate market: overall quality, total property size, number of bathrooms, house age, garage capacity and quality-related variables.

The final model, metrics and test predictions were saved as artifacts so the results can be reviewed and the project can be extended later.

## 2. Technologies and Tools

This project was developed using Python in Jupyter Notebook, with Visual Studio Code as the development environment and Git/GitHub for version control.

The main libraries used were:

- **Pandas and NumPy** for data manipulation and numerical operations.
- **Matplotlib and Seaborn** for exploratory data visualization.
- **Scikit-Learn** for preprocessing, pipelines, cross-validation, model training, hyperparameter tuning and evaluation.
- **Joblib** for saving the final trained model.

The regression models tested included Ridge Regression, ElasticNet, Random Forest, Extra Trees and Gradient Boosting Regressor.
