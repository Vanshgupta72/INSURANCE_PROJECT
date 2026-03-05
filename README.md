# Insurance Cost Prediction using Machine Learning

This project predicts **medical insurance charges** using machine learning techniques.  
The objective is to build a **clean and reproducible ML workflow** using preprocessing pipelines, feature engineering, and model evaluation.

---

## Project Overview

Insurance companies estimate medical costs based on factors such as:

- Age
- BMI
- Number of children
- Smoking status
- Region
- Gender

In this project, we build an **end-to-end machine learning pipeline** that preprocesses the data and trains a model to predict insurance charges.

---

## Dataset

The dataset contains the following features:

| Feature | Description |
|--------|-------------|
| age | Age of the individual |
| sex | Gender (male / female) |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance cost (Target Variable) |

---

## Project Workflow

### 1️⃣ Data Understanding
Initial inspection of the dataset:

- Checking dataset shape
- Inspecting column names
- Checking data types
- Identifying missing values
- Statistical summary of features

---

### 2️⃣ Exploratory Data Analysis (EDA)

Performed multiple types of analysis to understand the dataset.

#### Univariate Analysis
- Histograms
- Distribution plots
- Boxplots
- Pie charts for categorical variables

#### Multivariate Analysis
- Scatter plots
- Bar plots
- Heatmaps
- Pair plots
- Crosstab analysis

These visualizations helped identify relationships between features and the target variable.

---

## Feature Engineering

Separated features and target variable:

```python
X = df.drop(columns=['charges'])
y = df['charges']
```

Split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Data Preprocessing

Different preprocessing techniques were applied to different columns using **ColumnTransformer**.

### Numerical Features
- age
- bmi
- children

Transformations applied:

- SimpleImputer (median strategy)
- StandardScaler

### Categorical Features

Two different encoding methods were used.

#### Ordinal Encoding
Applied to:

- sex
- smoker

#### One-Hot Encoding
Applied to:

- region

---

## Machine Learning Pipeline

To prevent **data leakage** and create a clean workflow, a full **Scikit-learn Pipeline** was implemented.

Pipeline Structure:

```
ColumnTransformer
   │
   ├── Numerical Pipeline
   │       ├── SimpleImputer
   │       └── StandardScaler
   │
   ├── Ordinal Encoding
   │       └── OrdinalEncoder
   │
   └── OneHot Encoding
           └── OneHotEncoder
           
Model
   └── XGBoost Regressor
```

Example pipeline code:

```python
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', XGBRegressor())
])
```

---

## Model Training

The model is trained using the processed dataset.

```python
pipeline.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred = pipeline.predict(X_test)
```

---

## Model Evaluation

The following evaluation metrics were used:

- R² Score
- Mean Absolute Error (MAE)
- Cross Validation

### Results

| Metric | Score |
|------|------|
| Test R² Score | 0.85 |
| Mean Absolute Error | 2765 |
| Cross Validation R² | 0.79 |

These results indicate that the model explains approximately **85% of the variance** in medical insurance charges.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Key Learnings

Through this project I learned:

- Building end-to-end machine learning workflows
- Performing exploratory data analysis
- Applying multiple encoding techniques
- Using ColumnTransformer for feature preprocessing
- Creating reproducible pipelines
- Evaluating models using cross-validation

---

## Future Improvements

Possible improvements to this project:

- Hyperparameter tuning using GridSearchCV
- Testing additional models (Random Forest, Gradient Boosting)
- Feature engineering
- Model deployment using Flask or FastAPI

---

## Author

**Vansh Gupta**

Machine Learning Enthusiast  
Aspiring Data Scientist
