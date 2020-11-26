# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:34:24 2020

@author: Hp
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

train_data  = pd.read_csv("application_train.csv")
test_date   = pd.read_csv("application_test.csv")

train_data['TARGET'].value_counts()
train_data['TARGET'].astype(int).plot.hist();

features_with_na = [features for features in train_data.columns if train_data[features].isnull().sum()>1]
#use models such as XGBoost that can handle missing values
# with no need for imputation. Another option would be to drop columns
#  with a high percentage of missing values, although it is impossible to know ahead of
#  time if these columns will be helpful to our model. 
#  Therefore, we will keep all of the columns for now.

for features in features_with_na:
    print(features, np.round(train_data[features].isnull().mean(), 4),  ' % missing values')
    
train_data.dtypes.value_counts()
train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train_data = pd.get_dummies(train_data,drop_first=True)
test_date = pd.get_dummies(test_date,drop_first=True)

train_target = train_data['TARGET']
train_data,test_date = train_data.align(test_date,join = 'inner', axis = 1)
train_data['TARGET'] = train_target
a= train_data.describe()

train_data['DAYS_BIRTH'] =  train_data['DAYS_BIRTH']/-365 
train_data['DAYS_EMPLOYED'].describe()
#it doesnt look right since the max value is 1000 years!

anom = train_data[train_data['DAYS_EMPLOYED'] == 365243]
non_anom = train_data[train_data['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# Create an anomalous flag column
train_data['DAYS_EMPLOYED_ANOM'] = train_data["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
train_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

test_date['DAYS_EMPLOYED_ANOM'] = test_date["DAYS_EMPLOYED"] == 365243
test_date["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

correlations = train_data.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
#As the client gets older, there is a negative linear relationship with the target meaning that as clients get older, they tend to repay their loans on time more often.

plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(train_data['DAYS_BIRTH'], edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');

plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(train_data.loc[train_data['TARGET'] == 0, 'DAYS_BIRTH'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(train_data.loc[train_data['TARGET'] == 1, 'DAYS_BIRTH'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');

#KDE plot shows that as we get older we tend to pay our loans back

data_age  = train_data[['TARGET','DAYS_BIRTH']]
data_age['YEARS_BINNED'] = pd.cut(data_age['DAYS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_group = data_age.groupby('YEARS_BINNED').mean()


plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_group.index.astype(str), 100 * age_group['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


ext_data = train_data[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');

#All three EXT_SOURCE featureshave negative correlations with the target, indicating that as the value of the EXT_SOURCE increases, the client is more likely to repay the loan. 

#For Feature Engineering use variants of gradient boosting
#Two methods are currently used for this Feature engineering
#1.Polynomial Features
#2.Domain Knowledge Features

#1.
poly_features = train_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = test_date[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# imputer for handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]

# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))

poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = train_data['SK_ID_CURR']
app_train_poly = train_data.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = test_date['SK_ID_CURR']
app_test_poly = test_date.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)

app_train_domain = train_data.copy()
app_test_domain = test_date.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']           
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# #LOGISTIC REGRESSION

from sklearn.preprocessing import MinMaxScaler

# # Drop the target from the training data
# if 'TARGET' in train_data:
#     train = train_data.drop(columns = ['TARGET'])
# else:
#     train = train_data.copy()
    
# # Feature names
# features = list(train.columns)

# # Copy of the testing data
# test = test_date.copy()

# # Median imputation of missing values
# imputer = SimpleImputer(strategy = 'median')

# # Scale each feature to 0-1
# scaler = MinMaxScaler(feature_range = (0, 1))

# # Fit on the training data
# imputer.fit(train)

# # Transform both training and testing data
# train = imputer.transform(train)
# test = imputer.transform(test)

# # Repeat with the scaler
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)

# print('Training data shape: ', train.shape)
# print('Testing data shape: ', test.shape)
train_labels = train_data['TARGET']

# from sklearn.linear_model import LogisticRegression

# # Make the model with the specified regularization parameter
# log_reg = LogisticRegression(C = 0.0001)
# # Train on the training data
# log_reg.fit(train, train_labels)

# log_reg_pred = log_reg.predict_proba(test)[:, 1]
# log_reg_pred 

# # Submission dataframe
# submit = test_date[['SK_ID_CURR']]
# submit['TARGET'] = log_reg_pred

# submit.head()
# from sklearn.naive_bayes import GaussianNB

# from sklearn.neighbors import KNeighborsClassifier

poly_features_names = list(app_train_poly.columns)

# Impute the polynomial features
imputer = SimpleImputer(strategy = 'median')

poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

# Scale the polynomial features
scaler = MinMaxScaler(feature_range = (0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
model1= LogisticRegression(solver='lbfgs')
model1.fit(poly_features,train_labels)
pred1= model1.predict(poly_features)
accuracy1= accuracy_score(train_labels,pred1)
print(round(accuracy1*100,2))
confusionmat2= confusion_matrix(train_labels,pred1,labels=[0,1])
print(confusionmat2)


"NAIVE CLASSIFIER"
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(poly_features,train_labels)
pred4= gnb.predict(poly_features)
accuracy2= accuracy_score(train_labels,pred4)
print(round(accuracy2*100,2))
confusionmat4= confusion_matrix(y_test,pred4,labels=[0,1])
print(confusionmat4)


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(poly_features,train_labels)
pred5= knn_model.predict(poly_features)
accuracy3= accuracy_score(train_labels,pred5)
print(round(accuracy3*100,2))
confusionmat4= confusion_matrix(y_test,pred4,labels=[0,1])
print(confusionmat4)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
random_forest.fit(poly_features,train_labels)
pred6= random_forest.predict(poly_features)
accuracy4= accuracy_score(train_labels,pred6)
print(round(accuracy4*100,2))
confusionmat4= confusion_matrix(y_test,pred4,labels=[0,1])
print(confusionmat4)