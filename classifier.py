
#Importing necessary library
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


#loading dataset
dataset = pd.read_csv(r"D:\Course\Brainworks\Python_Practice\Projects\Final_project\datasets\edu_mentor_dataset_final(3000).csv")

dataset.shape

dataset.columns

dataset['lms_test_scores'].value_counts()

dataset['assignment_completion'].value_counts()

dataset.head()

dataset.info()

#need to drop these columns as these columns are unnecessary in model training student_name, student_id, email_id ,password
dataset = dataset.drop(columns=["student_name", "student_id", "email_id" , "password", ], axis=1)

dataset.isnull().sum()


# # EDA process

numeric_columns = ['std', 'math_grade', 'english_grade',
                   'science_grade', 'history_grade','overall_grade','average_session_duration_minutes',
                   'completed_lessons','practice_tests_taken','lms_test_scores','risk_score','is_at_risk']

dataset[numeric_columns].hist(bins=15, figsize=(12,10))
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

#boxplot to detect outliers
numeric_columns = dataset.select_dtypes(include='number').columns[:6]  # limit to 6 columns
plt.figure(figsize=(12, 8))

for i, col in enumerate(numeric_columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=dataset[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

#visualization of categorical features
import matplotlib.pyplot as plt
import seaborn as sns

categorical_columns = ['learning_style', 'content_type_preference', 'teacher_comments_summary']

# Set up the figure
fig, axes = plt.subplots(nrows=1, ncols=len(categorical_columns), figsize=(18, 6))
fig.suptitle("Countplot of Categorical Features", fontsize=16)

# Loop through the categorical columns
for ax, col in zip(axes, categorical_columns):
    sns.countplot(data=dataset, x=col, ax=ax)
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
plt.show()

#check whether the data is balanced or imbalanced
sns.countplot(x='is_at_risk', data=dataset)
plt.title("Target Variable Distribution: is_at_risk")
plt.show()

sns.boxplot(x='learning_style', y='overall_grade', data=dataset)
plt.title("Final Grades by Learning Style")
plt.xticks(rotation=45)
plt.show()

#label Encoding(labelEncoder)
le = preprocessing.LabelEncoder()

dataset['learning_style'].value_counts()

dataset['learning_style'] = le.fit_transform(dataset['learning_style'])

dataset['learning_style'].value_counts()

dataset['content_type_preference'].value_counts()

dataset['content_type_preference'] = le.fit_transform(dataset['content_type_preference'])

dataset['content_type_preference'].value_counts()

dataset['teacher_comments_summary'].value_counts()

dataset.replace({'teacher_comments_summary':{'Falling behind in multiple areas':0, 'Lacks focus':1, 'Needs improvement in basics':2, 'Good participation':3,'Average progress':4, 'Can do better with more effort':5, 'Consistent performance':6, 'Excellent progress':7,'Shows initiative':8}}, inplace=True)

dataset['teacher_comments_summary'].value_counts()

dataset['suggestions_required'].value_counts()

dataset.replace({'suggestions_required':{'Yes':1, 'No':0}}, inplace=True)

dataset['suggestions_required'].value_counts()

dataset.info()

#Classification
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

xgb = XGBClassifier(scale_pos_weight=2192/808)  # Adjusts loss function
rf = RandomForestClassifier(class_weight='balanced')
lr = LogisticRegression(class_weight='balanced')
cb = CatBoostClassifier(auto_class_weights='Balanced')


X = dataset.drop(columns=['is_at_risk', 'risk_score', 'suggestions_required'], axis=1)
y = dataset['is_at_risk']

print(X.shape)
print(y.shape)

smt = SMOTETomek(random_state=42)
X, y = smt.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=48)

#using pipeline for classification
pipelines = {
    'RandomForest': Pipeline([('rf', RandomForestClassifier())]),
    'LogisticRegression': Pipeline([('scaler', StandardScaler()),('lr', LogisticRegression())]),
    'CatBoost': Pipeline([('cb', CatBoostClassifier(verbose=0))]),
    'XGBoost': Pipeline([('xgb', XGBClassifier())])
}

for name, pipeline in pipelines.items():
    pipeline.fit(x_train, y_train)
    print(f"{name} Accuracy: {pipeline.score(x_train, y_train)}")
    print(f"{name} test Accuracy: {pipeline.score(x_test, y_test)}")


import joblib
joblib.dump(pipeline, 'student_risk_pipeline.pkl')

rf.fit(x_train,y_train)

lr.fit(x_train,y_train)

cb.fit(x_train,y_train)

xgb.fit(x_train,y_train)

#Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.svm import SVR

xgb_regressor = XGBRegressor(scale_pos_weight=2192/808)
rf_regressor = RandomForestRegressor()
lr_regressor = LinearRegression()
svr_regressor = svm.SVR(kernel='linear')

#train_test_split for regression
A = dataset.drop(columns=['is_at_risk', 'risk_score', 'suggestions_required'], axis=1)
b = dataset['risk_score']

a_train, a_test, b_train, b_test = train_test_split(
    A, b, test_size=0.2, random_state=48)

#using pipeline for regression
pipelines_regressor = {
    'RandomForest': Pipeline([('rf_regressor', RandomForestRegressor(random_state=42))]),
    'LinearRegression': Pipeline([('scaler', StandardScaler()),('lr_regressor', LinearRegression())]),
    'SVR': Pipeline([('svr_regressor', svm.SVR())]),
    'XGBoost': Pipeline([('xgb_regressor', XGBRegressor())])
}

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error

for name, pipeline in pipelines_regressor.items():
    pipeline.fit(a_train, b_train)

# Predictions
    b_train_pred = pipeline.predict(a_train)
    b_test_pred = pipeline.predict(a_test)

# Train metrics
    r2_train = r2_score(b_train, b_train_pred)
    mse_train = mean_squared_error(b_train, b_train_pred)
    mae_train = mean_absolute_error(b_train, b_train_pred)
    rmse_train = np.sqrt(mse_train)

# Test metrics
    r2_test = r2_score(b_test, b_test_pred)
    mse_test = mean_squared_error(b_test, b_test_pred)
    mae_test = mean_absolute_error(b_test, b_test_pred)
    rmse_test = np.sqrt(mse_test)

# Print
    print(f"\n{name} Performance:")
    print(f"Train -> R2: {r2_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
    print(f"Test  -> R2: {r2_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")


rf_regressor.fit(a_train,b_train)

lr_regressor.fit(a_train,b_train)

svr_regressor.fit(a_train,b_train)

xgb_regressor.fit(a_train,b_train)


#input_data = (9,78,73,85,98,83.5,0.53,0.66,14,10,1,11,0.45,3,86,3,1,15,9,43,1)
#12,91,67,54,45,64.25,0.63,0.48,6,16,3,7,0.4,5,15,3,2,6,9,60,1
#input_data = (11,50,89,76,66,70.25,0.55,0.32,9,11,16,19,0.69,3,6,3,1,6,9,69,0)
#input_data = (10,60,97,63,88,77,0.47,0.44,14,16,2,17,0.61,5,56,0,3,4,0,69,2)
#input_data = (9,79,86,95,45,76.25,0.78,0.6,20,5,3,20,0.6,2,25,1,0,5,2,35,2)
#input_data = (12,91,67,54,45,64.25,0.63,0.48,6,16,3,7,0.4,5,15,3,2,6,9,60,1)
input_data = (11,83,90,56,49,69.5,0.58,0.81,12,16,16,20,0.8,6,39,2,0,10,2,41,0)


#change input data to numpy array type
arr = np.asarray(input_data)

#reshape the numpy array for only one instance
reshaped = arr.reshape(1,-1)

prediction = xgb.predict(reshaped)
predicted_class = prediction[0]

if predicted_class == 1:
    print("Need improvements...")
else:
    print("No any suggestions required... keep it up! ")


#input_data = (9,78,73,85,98,83.5,0.53,0.66,14,10,1,11,0.45,3,86,3,1,15,9,43,1)
input_data = (11,83,90,56,49,69.5,0.58,0.81,12,16,16,20,0.8,6,39,2,0,10,2,41,0)
#12,91,67,54,45,64.25,0.63,0.48,6,16,3,7,0.4,5,15,3,2,6,9,60,1
#input_data = (11,50,89,76,66,70.25,0.55,0.32,9,11,16,19,0.69,3,6,3,1,6,9,69,0)
#input_data = (10,60,97,63,88,77,0.47,0.44,14,16,2,17,0.61,5,56,0,3,4,0,69,2)
#input_data = (9,79,86,95,45,76.25,0.78,0.6,20,5,3,20,0.6,2,25,1,0,5,2,35,2)
#input_data = (12,91,67,54,45,64.25,0.63,0.48,6,16,3,7,0.4,5,15,3,2,6,9,60,1)


#change input data to numpy array type
arr = np.asarray(input_data)

#reshape the numpy array for only one instance
reshaped = arr.reshape(1,-1)

prediction = xgb_regressor.predict(reshaped)
#predicted_class = prediction[0]

print(f"Risk Score is {prediction}")

#get_ipython().system('jupyter nbconvert --to script EduMentor.ipynb')

import pickle


# # Save model using pickle
# with open("Edumentor_risk_calculator.pkl", "wb") as file:
#     pickle.dump(XGBRegressor, file)
# print("Model saved as Edumentor_risk_calculator.pkl")

# with open("Edumentor_risk_classifier.pkl", "wb") as file:
#     pickle.dump(XGBClassifier, file)
# print("Model saved as Edumentor_risk_classifier.pkl")

import joblib

joblib.dump(xgb, 'classification_model.pkl')

joblib.dump(xgb_regressor, 'regression_model.pkl')