"""
Feature Space Fraud Detection Data Science Challenge

Name:- Yash Desai

"""

# importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataprep.eda import *
import seaborn as sns
from datetime import datetime

# loading the dataset

desc = pd.read_excel('data-dictionary.xlsx', index_col=0)
data_features = pd.read_csv('transactions_obf.csv')
data_labels = pd.read_csv('labels_obf.csv')
fraud_transactions = data_labels['eventId'].tolist()
print('-->Loading our data...')

# creating the target variable "isFraud"

data_features.loc[data_features['eventId'].isin(fraud_transactions) , 'isFraud'] = 1
data_features.loc[data_features['eventId'].isin(fraud_transactions) == False , 'isFraud'] = 0
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

# helper functions

def get_pod(x):
    if (x.hour > 4) and (x.hour <= 8):
        return 'Early Morning'
    elif (x.hour > 8) and (x.hour <= 12 ):
        return 'Morning'
    elif (x.hour > 12) and (x.hour <= 16):
        return'Noon'
    elif (x.hour > 16) and (x.hour <= 20) :
        return 'Eve'
    elif (x.hour > 20) and (x.hour <= 24):
        return'Night'
    elif (x.hour <= 4):
        return'Late Night'


# data cleaning, data pre-processing and feature engineering
# performing each of the above feature by feature to have a better understanding of the featrues

# transactionTime

"""

Transaction time has a very high cardinality with 99.6% unique values it should not be used to train
the model as it does not provide much information to the model instead I have extracted the hour from
the time and created a new feature which signifies the part of the day when the transaction was made.
This can be a great feature as the model can spot patterns to identify suspicious behaviour in unusual time of the day. 

"""

data_features['transactionTime'] = pd.to_datetime(data_features['transactionTime'])
data_features['part_of_day'] = data_features['transactionTime'].apply(get_pod)
pod_dummies = pd.get_dummies(data_features['part_of_day'],prefix = 'tx_at')
data_features = pd.concat([data_features,pod_dummies],axis=1)
data_features = data_features.drop(columns=['transactionTime','part_of_day'])

print('-->Creating a new feature called part_of_day from transactionTime to signify the part of Day...')

# eventId

"""
Similar to transaction time 'eventId' has a very high cardinality as well with 100% unique values
and it should not be used to train the model. Hence we drop this column from our feature set.

"""

data_features = data_features.drop(columns=['eventId'])
print('')
print('-->Dropping eventId...')

# accountNumber

"""
Account Number has just 766 unique values which is 0.6% of the total values. Account number as it is can't be a great predictor as it is just an id and does not provide any information. But instead the frequency of the account numbers can be calculated and used as a feature. The number of transactions by the same account is definitely a great predictor for fraud detection systems. High number of transactions from the same account can signal towards a probable fraud.
""" 

acc_freq = data_features.accountNumber.value_counts()
acc_freq = pd.DataFrame(acc_freq)
acc_freq["name"] = acc_freq.index
acc_freq.rename(columns = {'accountNumber':'acc_freq', 'name':'accountNumber'}, inplace = True)
data_features = data_features.merge(acc_freq, on = 'accountNumber')
data_features = data_features.drop(columns=['accountNumber'])
print('')
print('-->Computing frequency of account numbers instead of the ids...')
# merchantId

"""

Merchant Id is a unique id of the merchant and has 33327 unique values. It should not be used as it is while training the model because of its very high cardinality. Frequency of the merchant id could be used but there is a very high chance that it gets correlated with 'mcc' as similar merchant ids will always have similar mcc.
Hence we drop this column from our feature set.

"""
data_features = data_features.drop(columns=['merchantId'])
print('')
print('-->Dropping merchantId...')

# mcc

"""

MCC represents the merchant category code of the merchant. It specifies the type of goods or services the merchant provides. It has a high cardinality as well and hence we use the frequency of MCC instead of the MCC codes. The frequency will represent the number of times a specific type of service or goods category appeared. 

"""
mcc_freq = data_features.mcc.value_counts()
mcc_freq = pd.DataFrame(mcc_freq)
mcc_freq["mcc_code"] = mcc_freq.index
mcc_freq.rename(columns = {'mcc':'mcc_freq', 'mcc_code':'mcc'}, inplace = True)
data_features = data_features.merge(mcc_freq, on = 'mcc')
data_features = data_features.drop(columns=['mcc'])
print('')
print('-->Considering the frequency of merchant category code instead of the id...')

# merchantCountry

"""
It is the country of the merchant who charged for the transaction. It is a categorical variable with 82 different codes for
diffrent countries. If we were to use a one hot encoding technique to convert the categorical variable to numeric, we would end up adding 82 columns to our feature set which increases the cardinality. This would increase the memory consumption as well as increase the risk of "curse of dimesnioanlity" which says that:-

" as the number of features grows, the amount of data we need to accurately be able to distinguish between these features (in order to give us a prediction) and generalize our model grows EXPONENTIALLY "

Hence I will set a threshold value for the frequency of transactions in countries below which I will put all the countries in a separate column called 'low_freq_countries'
"""

merchant_country_freq = pd.DataFrame(data_features.merchantCountry.value_counts())
merchant_country_freq["mc"] = merchant_country_freq.index
country_list = merchant_country_freq.loc[merchant_country_freq.merchantCountry>100,"mc"] # extracting countires whose transaction frequency is more than 100
data_features.loc[data_features["merchantCountry"].isin(country_list)==False,"merchantCountry"]="low_freq_countires" # marking rest countries as "low_freq_countries"

mc_dummies = pd.get_dummies(data_features['merchantCountry'],prefix = 'mc')
data_features = pd.concat([data_features,mc_dummies],axis=1)
data_features = data_features.drop(columns=['merchantCountry'])
print('')
print('-->Separating low frequency countries and creating dummy variables...')

# merchantZip

'''

It is the zip code of the postal address of the merchant. This column cointains 23005 missing values and the best method is to drop the column.
'''
print('')
print('The number of na values for the feature "merchantZip" is', data_features.apply(lambda x : x.isnull().sum()).to_dict().get('merchantZip'))
print('')
print('-->Dropping merchantZip...')
data_features = data_features.drop(columns=['merchantZip'])


# posEntryMode

'''
It represents the point of sale entry mode and there are just 10 distinct values hence I have used a one hot encoder. This can be done using the get_dummies function from pandas.
'''

pos_dummies = pd.get_dummies(data_features['posEntryMode'],prefix='pos_mode')
data_features = pd.concat([data_features,pos_dummies],axis=1)
data_features = data_features.drop(columns=['posEntryMode'])
print('')
print('-->Creating dummy variables for posEntryMode...')

# transactionAmount

"""
This column contains a certain number of negative values which are not veyr high in magnitude. The minimum tranasction amount is "-0.15". We can either reomove these 183 values or make them zero. It wont make a huge difference if I perform any of these two methods but leaving the negative values as it is can degrade the performance of our classifier. I chose to remove the observations as -ve transaction values don't make sense and making them zero without any strong reason would not make any sense as well. As far as the skewness of this feature is concerned, I do not plan to use a parametric model 
hence transforming the values according to a specific distribution won't make sense. 

"""

neg_indexes = data_features[ data_features['transactionAmount'] < 0 ].index
data_features.drop(neg_indexes , inplace=True)
print('')
print('-->Removing negative transactions from transactionAmount...')
# availableCash

"""
# This column seems to be perfectly fine and can be used in our prediction models as it is. 

"""

data_features.to_csv('processed_df.csv')
print('')
print('-->Completed Pre-processing and feature engineering...')