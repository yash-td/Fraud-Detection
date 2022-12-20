"""
Feature Space Fraud Detection Data Science Challenge

Name:- Yash Tusharbhai Desai

"""

# importing necessary libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix

# loading the dataset
desc = pd.read_excel('data-dictionary.xlsx', index_col=0)
data_features = pd.read_csv('transactions_obf.csv')
data_labels = pd.read_csv('labels_obf.csv')
fraud_transactions = data_labels['eventId'].tolist()
print('-->Loading our data...')
print('')
# creating the target variable "isFraud"
data_features.loc[data_features['eventId'].isin(fraud_transactions) , 'isFraud'] = int(1)
data_features.loc[data_features['eventId'].isin(fraud_transactions) == False , 'isFraud'] = int(0)
counts = data_features['isFraud'].value_counts()
print(f'Out of the total {len(data_features)} transactions, {dict(counts).get(0)} are genuine and {dict(counts).get(1)} are fraud.')


"""

Preparing the data for making it suitable to fit the model:-

1) Dealing with very high cardinality by using frequency instead of the original values
2) Transforming the categorical variables into a limited number of categories to preserve space and take care of the curse of dimensionality. (one hot encoding increases the cardinality)
3) Removing columns which have almost all unique values (represent ids and no significant information)
4) Removing rows which have negative values for "transactionAmount"
5) Removing columns with a high number of missing values
6) Removing redundant or highly correlated features
7) Creating a new feature from "transactionTime"

""" 


# data cleaning, data pre-processing and feature engineering
# performing each of the above feature by feature to have a better understanding of the featrues


# 1) transactionTime

"""

Transaction time has a very high cardinality with 99.6% unique values it should not be used to train
the model as it does not provide much information to the model instead I have extracted the hour from
the time and created a new feature which signifies the part of the day when the transaction was made.
This can be a great feature as the model can spot patterns to identify suspicious behaviour in unusual time of the day. 

"""

data_features['transactionTime'] = pd.to_datetime(data_features['transactionTime'])
data_features['transactionMonth'] = data_features['transactionTime'].apply(get_month)
data_features['transactionTime'] = data_features['transactionTime'].apply(get_pod)
data_features = create_dummies(data_features,data_features['transactionTime'],'time')


print('-->Creating a new feature from transactionTime which signifies the part of Day (morning, early morning, late night etc)...')

# 2) eventId

"""
Similar to transaction time 'eventId' has a very high cardinality as well with 100% unique values
and it should not be used to train the model. Hence we drop this column from our feature set.

"""

data_features = data_features.drop(columns=['eventId'])
print('')
print('-->Dropping eventId...')

# 3) accountNumber

"""
Account Number has just 766 unique values which is 0.6% of the total values. Account number as it is can't be a great predictor as it is just an id and does not provide any information. But instead the frequency of the account numbers can be calculated and used as a feature. The number of transactions by the same account is definitely a great predictor for fraud detection systems. High number of transactions from the same account can signal towards a probable fraud. Finally I standardise the highly skewed values using min max normalisation. By this the values in this column will follow a normal distribution with a mean of 0 and standard deviation of 1.
""" 
print('')
print('-->Computing frequency of account numbers instead of the ids...')
mm = StandardScaler()
data_features = get_frequency(data_features,data_features.accountNumber,'acc_freq')

# 4) merchantId

"""

Merchant Id is a unique id of the merchant and has 33327 unique values. It should not be used as it is while training the model because of its very high cardinality. Frequency of the merchant id could be used but there is a very high chance that it gets correlated with 'mcc' as similar merchant ids will always have similar mcc.
Hence we drop this column from our feature set.

"""
data_features = data_features.drop(columns=['merchantId'])
print('')
print('-->Dropping merchantId...')

# 5) mcc

"""

MCC represents the merchant category code of the merchant. It specifies the type of goods or services the merchant provides. It has a high cardinality as well and hence we use the frequency of MCC instead of the MCC codes. The frequency will represent the number of times a specific type of service or goods category appeared. I perform min max normnalisatio finally to normalise the values.

"""
data_features = get_frequency(data_features,data_features['mcc'],'mcc_freq')
print('')
print('-->Considering the frequency of merchant category code instead of the id...') 

# 6) merchantCountry

"""
It is the country of the merchant who charged for the transaction. It is a categorical variable with 82 different codes for
diffrent countries. If we were to use a one hot encoding technique directly to convert the categorical variable to numeric, we would end up adding 82 columns to our feature set which increases the cardinality. This would increase the memory consumption as well as increase the risk of "curse of dimesnioanlity" which says that:-

" as the number of features grows, the amount of data we need to accurately be able to distinguish between these features (in order to give us a prediction) and generalize our model grows EXPONENTIALLY "

Hence I will set a threshold value for the frequency of transactions in countries below which I will put all the countries in a separate column called 'low_freq_countries'
"""

merchant_country_freq = pd.DataFrame(data_features.merchantCountry.value_counts())
merchant_country_freq["mc"] = merchant_country_freq.index
country_list = merchant_country_freq.loc[merchant_country_freq.merchantCountry>100,"mc"] # extracting countires whose transaction frequency is more than 100
data_features.loc[data_features["merchantCountry"].isin(country_list)==False,"merchantCountry"]="low_freq_countires" # marking rest countries as "low_freq_countries"
data_features = create_dummies(data_features,data_features['merchantCountry'],'mc')
print('')
print('-->Separating low frequency countries and creating dummy variables...')

# 7) merchantZip

'''

It is the zip code of the postal address of the merchant. This column cointains 23005 missing values and the best method is to drop the column.
'''
print('')
print('The number of na values for the feature "merchantZip" is', data_features.apply(lambda x : x.isnull().sum()).to_dict().get('merchantZip'))
print('')
print('-->Dropping merchantZip...')
data_features = data_features.drop(columns=['merchantZip'])


# 8) posEntryMode

'''
It represents the point of sale entry mode and there are just 10 distinct values hence I have used a one hot encoder. This can be done using the get_dummies function from pandas.
'''

data_features = create_dummies(data_features,data_features['posEntryMode'],'pos_mode')
print('')
print('-->Creating dummy variables for posEntryMode...')

# 9) transactionAmount

"""
This column contains a certain number of negative values which are not veyr high in magnitude. The minimum tranasction amount is "-0.15". We can either reomove these 183 values or make them zero. I chose to remove the observations as -ve transaction values don't make sense and making them zero without any strong reason would not make any sense as well. As far as the skewness of this feature is concerned, I perform min max normalisation to normalise the values of this column.
"""

neg_indexes = data_features[ data_features['transactionAmount'] < 0 ].index
data_features.drop(neg_indexes , inplace=True)
print('')
print('-->Removing negative transactions from transactionAmount...')
# data_features['transactionAmount'] = mm.fit_transform(data_features[['transactionAmount']])


# 10) availableCash

"""
This column seems to be perfectly fine and can be used in our prediction models as it is. We just need to normalise the 
values as they are heavily skewed. I do this using log transformation like I did for transactionAmount. I have added 0.01
to deal with the 0 values as dividing by zero will give us inf.

"""
# data_features['availableCash'] = mm.fit_transform(data_features[['availableCash']])


data_features.to_csv('processed_df.csv')
print('')
print('-->Completed Pre-processing and feature engineering...')

"""
With only 0.74% transactions that are fraudulent, a classifier that predicts every transaction to be not fraudulent will achieve accuracy score of 99.26%. However, such a classifier cannot be used in . Therefore, in the cases when classes are imbalanced, metrics other than accuracy should be considered. These metrics include precision, recall and a combination of these two metrics (F2).

"""

# Splitting the data into train, test and validation
print('')
print('-->Creating Training, Testing and Validation sets...')
X_train,X_test,X_val,y_train,y_test,y_val = prepare_data(data_features)

# Balancing the train data

'''
The target variable "isFraud" is highly imbalanced. In such case the classifier will favour the majority classs start generating false predictions. Hence we upsample the minority class using synthetic monetory upsampling technique. I performed
the train_test split before balancing the data because only the training data is supposed to be balanced.
The test data is the real world data which is never going to be balanced. The model optimisation is performed on the 
validation data hence it needs to see real-world like data as well. 

The penalty for mislabeling a fraud transaction as legitimate is having a user’s money stolen, which the credit card company
typically reimburses. On the other hand, the penalty for mislabeling a legitimate transaction as fraud is having the user frozen out of their finances and unable to make payments. Balancing the data keeping in mind that we need to catch most of the fraudulent transactions (Aiming for maximum recall).

Reviewing the bank's request that after working these alerts, as much fraud value as possible needs to be prevented, I prioritise recall over precision and care less about the false positives. Hence I oversample with 1:1 ratio. More the synthetic samples the model sees, more the recall but also the chance of getting false positives increases a lot which would decrease precision.

'''

maj_class, min_class = class_dist(X_train,y_train,'isFraud')

X_train_sm, y_train_sm = upsample_SMOTE(X_train,y_train,'is_Fraud',0.5)

print('-'*70)
f1, f1_b = base_models(X_train,X_train_sm,X_test,y_train,y_train_sm,y_test)
print('-'*70)
print(best_metrics(f1,'precision_score','unbalanced'))
print(best_metrics(f1_b,'precision_score','balanced'))
print('-'*70)
print(best_metrics(f1,'recall_score','unbalanced'))
print(best_metrics(f1_b,'recall_score','balanced'))
print('-'*70)
print(best_metrics(f1,'f1_score','unbalanced'))
print(best_metrics(f1_b,'f1_score','balanced'))
print('-'*70)

# using cross validation

