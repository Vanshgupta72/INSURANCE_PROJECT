import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport

df = pd.read_csv("PROJECTS\INSURANCE_PROJECT(1)\insurance.csv",on_bad_lines="skip" )  # Loading the dataset

# STEP-1 UNDERSTANDING THE DATA

# # print(df.shape)
# print(df.head)
# print(df.columns)
# print(df.info())
# print(df.isnull().sum())
# print(df.describe())
# print(df.corr(numeric_only=True))

# STEP-2 EXPLORATERY DATA ANALYSIS

# (a) Univariate analysis

# - categorical columns 

# sns.countplot(df['sex'])
# sns.countplot(df['smoker'])
# sns.countplot(df['region'])

# df['sex'].value_counts().plot(kind='pie',autopct='%.2f')
# df['smoker'].value_counts().plot(kind='pie',autopct='%.2f')
# df['region'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()

# - numerical columns

# plt.hist(df['age'],bins=10,edgecolor='black',rwidth=0.8)
# plt.hist(df['bmi'],bins=10)
# plt.hist(df['children'],bins=5)
# plt.hist(df['charges'],bins=50)
# plt.tight_layout
# plt.show()

# sns.distplot(df['age'])
# sns.displot(df['bmi'])
# sns.distplot(df['children'])
# sns.distplot(df['charges'])
# plt.show()

# sns.boxplot(df['age'])
# sns.boxplot(df['bmi'])
# sns.boxplot(df['children'])
# sns.boxplot(df['charges'])
# plt.show()

# print(df['age'].skew()) # isitarh min,max,mean,variance,standard_deviation sab nikal sakte hai 
# print(df['bmi'].skew())
# print(df['children'].skew())
# print(df['charges'].skew())


# (b) Multivariate analysis

# - Numerical-Numerical column

# sns.scatterplot(x=df['age'],y=df['bmi'],color='red',hue=df['children'],style=df['charges'])

# - Numerical-Categorical data

# sns.barplot(y=df['age'],x=df['sex'],hue=df['smoker'])

# sns.boxenplot(x=df['region'],y=df['age'])

# -Categorical-Categorical

# data=pd.crosstab(df['sex'],df['region'])
# sns.heatmap(data)

# data=pd.crosstab(df['sex'],df['region'])
# sns.clustermap(data)

# sns.pairplot(df)

# sns.lineplot(x=df['age'],y=df['children'])
# plt.show()

# PANDAS PROFILING: 

# prof = ProfileReport(df)
# prof.to_file('output.html')

# STEP-3 FEATURE ENGINEERING

X = df.drop(columns=['charges'])
y = df['charges']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train_numeric = X_train.select_dtypes(include=['int64','float64'])
X_test_numeric = X_test.select_dtypes(include=['int64','float64'])

# (I) Feature transformation:

# (a) Feature Scaling:

# (a1) Standardization:

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_numeric)
# X_test_scaled = scaler.transform(X_test_numeric)
# dftr = pd.DataFrame(X_train_scaled,columns=X_train_numeric.columns,index=X_train_numeric.index)
# dfte = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)

# (a2) Normalization

# (a21) MinMaxScaler:

# from sklearn.preprocessing import MinMaxScaler
# scaler1 = MinMaxScaler()
# X_train_scaled1 = scaler1.fit_transform(X_train_numeric)
# X_test_scaled1 = scaler1.transform(X_test_numeric)
# df1tr = pd.DataFrame(X_train_scaled1,columns=X_train_numeric.columns, index=X_train_numeric.index)
# df1te = pd.DataFrame(X_test_scaled1,columns=X_test_numeric.columns,index=X_test_numeric.index)

# (a22) Mean Normalization:

# Feature is not present in sklearn rarely used 

# (a23) Max_absloute_scaling:

# from sklearn.preprocessing import MaxAbsScaler
# scaler2 = MaxAbsScaler()
# X_train_scaled2 = scaler2.fit_transform(X_train_numeric)
# X_test_scaled2 = scaler2.transform(X_test_numeric)
# df2tr = pd.DataFrame(X_train_scaled2,columns=X_train_numeric.columns,index=X_train_numeric.index)
# df2te = pd.DataFrame(X_test_scaled2,columns=X_test_numeric.columns,index=X_test_numeric.index)

# (a24) Robust Scaler

# from sklearn.preprocessing import RobustScaler
# scaler3 = RobustScaler()
# X_train_scaled3 = scaler3.fit_transform(X_train_numeric)
# X_test_scaled3 = scaler3.transform(X_test_numeric)
# df3tr = pd.DataFrame(X_train_scaled3,columns=X_train_numeric.columns,index=X_train_numeric.index)  # Converting the provided array into dataframe
# df3te = pd.DataFrame(X_test_scaled3,columns=X_test_numeric.columns,index=X_test_numeric.index)

# (b) Encoding categorical data 

# (b1) Ordinal encoding with nominal categorical column

# from sklearn.preprocessing import OrdinalEncoder
# OE = OrdinalEncoder(categories=[['female','male'],['southeast','southwest','northeast','northwest'],['yes','no']])
# X_train[['sex','region','smoker']] = OE.fit_transform(X_train[['sex','region','smoker']])
# X_test[['sex','region','smoker']] = OE.transform(X_test[['sex','region','smoker']])
# print(X_train)

# (b2) One hot encoding with nominal categorical columns

# (b21) using pandas 

# X_train = pd.get_dummies(X_train,columns=['region'],drop_first=True)
# X_test = pd.get_dummies(X_test,columns=['region'],drop_first=True)
# print(X_test.head())

# (b22) using sklearn(Recommneded) 

# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder(drop='first',sparse_output=False)
# encoded = OHE.fit_transform(X_train[['sex','smoker','region']])
# encoded_df = pd.DataFrame(encoded,columns=OHE.get_feature_names_out(),index=X_train.index)
# X_train = pd.concat([X_train.drop(['sex','smoker','region'],axis=1),encoded_df],axis=1)


# (b3) Using columns transfromer

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# ct = ColumnTransformer(
#     transformers = [
#         ('ohe',OneHotEncoder(drop='first'),['region']),
#         ('ordinal',OrdinalEncoder(categories=[['female','male']]),['sex']),
#         ('ordinal1',OrdinalEncoder(categories=[['no','yes']]),['smoker'])
#     ],
#     remainder='passthrough'
# )

# X_train = pd.DataFrame(ct.fit_transform(X_train),columns=ct.get_feature_names_out(),index=X_train.index)
# X_test = pd.DataFrame(ct.transform(X_test),columns=ct.get_feature_names_out(),index=X_test.index)


# * CREATING PIPELINE OF ALL THE ABOVE STEPS:

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

nums_cols = ['age','bmi','children']
ohe_cols = ['region']
ordinal_cols = ['sex','smoker']

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

ordinal_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ordinal',OrdinalEncoder(categories=[['female','male'],['no','yes']]))
])

ohe_pipiline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ohe',OneHotEncoder(drop='first'))
])

# Combining using column transformer

preprocessor = ColumnTransformer([
    ('num',num_pipeline,nums_cols),
    ('ordinal',ordinal_pipeline,ordinal_cols),
    ('ohe',ohe_pipiline,ohe_cols)
])

pipeline= Pipeline([
    ('preprocessing',preprocessor),
    ('model',XGBRegressor())
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error

scores = cross_val_score(pipeline,X_train,y_train,cv=5,scoring='r2')
print("Cross Validation Scores:", scores)
print("Average R2 Score:", scores.mean())













