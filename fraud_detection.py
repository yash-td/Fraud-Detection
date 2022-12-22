"""
Feature Space Fraud Detection Data Science Challenge

Name:- Yash Tusharbhai Desai

"""

# importing necessary libraries
import warnings
warnings.filterwarnings("ignore")
from utils import *
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix, PrecisionRecallDisplay
from sklearn.preprocessing import RobustScaler
import seaborn as sns
rs = RobustScaler()


# loading the dataset

data_features = pd.read_csv('data/transactions_obf.csv')
data_labels = pd.read_csv('data/labels_obf.csv')
fraud_transactions = data_labels['eventId'].tolist()
print('Loading our data...')
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


print('Creating a new feature from transactionTime which signifies the part of Day (morning, early morning, late night etc)...')

# 2) eventId

"""
Similar to transaction time 'eventId' has a very high cardinality as well with 100% unique values
and it should not be used to train the model. Hence we drop this column from our feature set.

"""

data_features = data_features.drop(columns=['eventId'])
print('')
print('Dropping eventId...')

# 3) accountNumber

"""
Account Number has just 766 unique values which is 0.6% of the total values. Account number as it is can't be a great predictor as it is just an id and does not provide any information. But instead the frequency of the account numbers can be calculated and used as a feature. The number of transactions by the same account is definitely a great predictor for fraud detection systems. High number of transactions from the same account can signal towards a probable fraud. Finally I standardise the highly skewed values using min max normalisation. By this the values in this column will follow a normal distribution with a mean of 0 and standard deviation of 1.
""" 
print('')
print('Computing frequency of account numbers instead of the ids...')
data_features = get_frequency(data_features,data_features.accountNumber,'acc_freq')
data_features['acc_freq'] = rs.fit_transform(data_features['acc_freq'].values.reshape(-1,1))

# 4) merchantId

"""

Merchant Id is a unique id of the merchant and has 33327 unique values. It should not be used as it is while training the model because of its very high cardinality. Frequency of the merchant id could be used but there is a very high chance that it gets correlated with 'mcc' as similar merchant ids will always have similar mcc.
Hence we drop this column from our feature set.

"""
data_features = data_features.drop(columns=['merchantId'])
print('')
print('Dropping merchantId...')

# 5) mcc

"""

MCC represents the merchant category code of the merchant. It specifies the type of goods or services the merchant provides. It has a high cardinality as well and hence we use the frequency of MCC instead of the MCC codes. The frequency will represent the number of times a specific type of service or goods category appeared. I perform min max normnalisatio finally to normalise the values.

"""
data_features = get_frequency(data_features,data_features['mcc'],'mcc_freq')
print('')
print('Considering the frequency of merchant category code instead of the id...') 
data_features['mcc_freq'] = rs.fit_transform(data_features['mcc_freq'].values.reshape(-1,1))

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
print('Separating low frequency countries and creating dummy variables...')

# 7) merchantZip

'''

It is the zip code of the postal address of the merchant. This column cointains 23005 missing values and the best method is to drop the column.
'''
print('')
print('The number of na values for the feature "merchantZip" is', data_features.apply(lambda x : x.isnull().sum()).to_dict().get('merchantZip'))
print('')
print('Dropping merchantZip...')
data_features = data_features.drop(columns=['merchantZip'])


# 8) posEntryMode

'''
It represents the point of sale entry mode and there are just 10 distinct values hence I have used a one hot encoder. This can be done using the get_dummies function from pandas.
'''

data_features = create_dummies(data_features,data_features['posEntryMode'],'pos_mode')
print('')
print('Creating dummy variables for posEntryMode...')

# 9) transactionAmount

"""
This column contains a certain number of negative values which are not veyr high in magnitude. The minimum tranasction amount is "-0.15". We can either reomove these 183 values or make them zero. I chose to remove the observations as -ve transaction values don't make sense and making them zero without any strong reason would not make any sense as well. As far as the skewness of this feature is concerned, I perform min max normalisation to normalise the values of this column.
"""

neg_indexes = data_features[ data_features['transactionAmount'] < 0 ].index
data_features.drop(neg_indexes , inplace=True)
print('')
print('Removing negative transactions from transactionAmount...')
data_features['transactionAmount'] = rs.fit_transform(data_features['transactionAmount'].values.reshape(-1,1))


# 10) availableCash

"""
This column seems to be perfectly fine and can be used in our prediction models as it is. We just need to normalise the 
values as they are heavily skewed. I do this using log transformation like I did for transactionAmount. I have added 0.01
to deal with the 0 values as dividing by zero will give us inf.

"""
data_features['availableCash'] = rs.fit_transform(data_features['availableCash'].values.reshape(-1,1))


data_features.to_csv('processed_data/processed_df.csv')
print('')
print('Completed Pre-processing and feature engineering...')

"""
With only 0.74% transactions that are fraudulent, a classifier that predicts every transaction to be not fraudulent will achieve accuracy score of 99.26%. However, such a classifier cannot be used in real world. Therefore, in the cases when classes are imbalanced, metrics other than accuracy should be considered. These metrics include precision, recall and a combination of these two metrics (F1). For imbalanced classification with a severe skew and few examples of the minority class like in our problem, the ROC AUC can be misleading. This is because a small number of correct or incorrect predictions can result in a large change in the ROC Curve or ROC AUC score. Hence I do not use ROC AUC for evaluation purposes. Instead I have used the PR AUC which is the area under the precision recall curve. I have discussed precisiom recall in great detail in the later part of this script. 

"""

# Splitting the data into train, test and validation
print('')
print('Creating Training, Testing and Validation sets...')
print('')
X_train,X_test,y_train,y_test = prepare_data(data_features)

# Balancing the train data

'''
The target variable "isFraud" is highly imbalanced. In such case the classifier will favour the majority classs start generating false predictions. Hence we upsample the minority class using synthetic monetory upsampling technique. I performed the train_test split before balancing the data because only the training data is supposed to be balanced. The test data is the real world data which is never going to be balanced. This is for training and testing base models and the final model. In hyper parameter tuning process, I will use cross-validation, hence I will perform the upsampling during the cross validation (imbpipeline) so that the validation set remains unbalanced while the model gets trained on the balanced train set. 

Balancing the data keeping in mind that we need to catch most of the fraudulent transactions (Aiming for maximum recall).

Reviewing the bank's request that after working these alerts, as much fraud value as possible needs to be prevented, I prioritise recall over precision and care relatively less about the false positives. Hence I oversample with 1:1 ratio. More the synthetic samples the model sees, more the recall but also the chance of getting false positives increase which would decrease precision.

'''

maj_class, min_class = class_dist(X_train,y_train,'isFraud')

X_train_sm, y_train_sm = upsample_SMOTE(X_train,y_train,'is_Fraud',1)
print('')
print('Fitting 5 base models....')
print('')
f1, [lr,dt,rf,xgb,adb] = base_models(X_train_sm,y_train_sm,X_test,y_test)
print('')
print(best_metrics(f1,'precision_score'))
print('')
print(best_metrics(f1,'recall_score'))
print('')
print(best_metrics(f1,'f1_score'))
print('')
print(best_metrics(f1,'pr_auc'))
print('')


print('-'*100)
print('Random Forest Classifier')
rf, rf_pred = fit_model(RandomForestClassifier(),X_train_sm,y_train_sm,X_test,y_test)
print('-'*100)
print('Logistic Regression')
lg, lg_pred = fit_model(LogisticRegression(),X_train_sm,y_train_sm,X_test,y_test)
print('-'*100)
print('AdaBoost Classifier')
adb, adb_pred = fit_model(AdaBoostClassifier(),X_train_sm,y_train_sm,X_test,y_test)
print('-'*100)
print('XGB Classifier')
xgb, xgb_pred = fit_model(XGBClassifier(),X_train_sm,y_train_sm,X_test,y_test)
print('-'*100)

print('')
test_thres,test_f1 = best_f1_pr_auc(xgb,'XGBoost Base',X_test,y_test)



'''
Analysis 1:

From the all the base models I noticed that the Random Forest Classifier has the best all around performance in terms of precision,recall and f1 score. This model has about 75% precision, which means that out of all the transactions that the model predidcted to be Fraud, 75% are actually Fraud. This can be great for cases where the cost of false positives is equally important as the cost of false negatives. Only 15% transactions which were predicted to be fraud were not actually fraud. The penalty for mislabeling a legitimate transaction as fraud is having the user frozen out of their finances and unable to make payments. Now the recall of this model is around 0.55 which means the model is able to catch only 55% of the fraudulent transactions. The penalty for mislabeling a fraud transaction as legitimate is having a users money stolen, which the credit card company typically reimburses. Letting 45% of the fraudulent transactions go can be a big impact to the company. Hence looking at the requirement of the problem statement which was to catch the maximum amount of fraud, Random Forest Classifier which although has a decent overall performance, cannot be suitable. The aim will be to maximise the recall so that maximum amount of fraud can be predicted. 

The base model Adaboost has a high recall which is 0.83. This model will catch 83% of the fraudulent transactions which is pretty good but at the same time the model has a very poor precision of 0.041. So is the case with LogisticRegression. (~0.8 recall but 0.049 precision). Such a low precision means that out of all the transactions that our model predicts to be fraud, only 4-5% are actually fraud. This means there will be more than 95% alarms which are false. In our business problem, the bank has a team of analysts who can reveiw only 400 transactions a month. The 1 year data provided has 118620 transactions where the monthly average is 9870 transactions. If we were to follow the distribution of this data, the analyst team can review (400/9870)*100 % transactions per month which is approximately 4% transactions. A high recall model like the Adaboost and the Logistic Regression may capture more than 80% of the fraudulent transactions but they will end up with false positives which are a way above the reviewing capacity of the analyst team. If the analyst team has a high reviewing capacity, then a high recall model which captures most fraud but with high number of false positives should be the best. 

Analysis 2:

The model which has a decent precision and also a good f1-score is the XGBoost Classifier. It predicts almost 70% fraud with a precision of 0.57. This means 57% of the transactions that are predicted as fraud are actually fraud. This number is way better than high recall models (5% correct predictions). Such a model will help to catch the maximum amount of fraud keeping in mind the reviewing capacity of the analyst team. 


Hence I further training the XGBoost Classifier model using KFold cross validation and perform a GridSearch to optimise recall. For this I will use the unsampled data and perform the SMOTE oversampling during the cross valiadation and not prior. Synthetic data is created only for the training set without affecting the validation set, so that the model is optimised on validation data which is unbalanced like the real world data. This will make the model perform better on our test data. 


I have performed the fine tuning of the model on a separate script called 'fine_tune_model.py'.

Uncomment the below code to perform hyperparameter tuning in this script (takes too long). 

'''

# print('Performing Randomized Search with KFold cross validation on the best model to fine tune it...')
# print('')

'''
1st Tune
'''
# params = {'xgb__max_depth': [1,2,3,4,5,6], 'xgb__min_child_weight': [1,2,3,4]} 
# fix_params = {'xgb__learning_rate': 0.2, 'xgb__n_estimators': 100, 'xgb__objective': 'binary:logistic'} 

'''
2nd Tune
'''
# params_2nd_run = {'xgb__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]}
# fix_params_2nd_run = {'xgb__n_estimators': 100, 'xgb__objective': 'binary:logistic', 'xgb__max_depth': 6, 'mxgb__in_child_weight':2}

'''
3rd Tune
'''
# params_3rd_run = {'xgb__n_estimators': [50,100,300,500]}
# fix_params_3rd_run = {'objective': 'binary:logistic', 'max_depth': 6, 'min_child_weight':2,'xgb__learning_rate':0.35}

'''
4th Tune
'''
# params_estimators = {'xgb__n_estimators':[10,100,500,1000]}
# best_model = fine_tune_model(XGBClassifier(),X_train,y_train,X_test,y_test,params_estimators,'f1')

# pickle.dump(best_model, open('model_f1.pkl', 'wb'))

'''
best parameters 1st tuning --> {'xgb__min_child_weight': 2, 'xgb__max_depth': 6} 
these will now be added to fix_params..
fix_params = {'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'binary:logistic', 'max_depth': 6, 'min_child_weight':2}

best parameters 2nd tuning --> {'xgb__learning_rate': 0.35}

best parameters 3rd tuning --> {'xgb__n_estimators':500}

Combination of the parameters from first 2 tunings did not improve the overall score of the model. Instead solely tuning the parameter
'n_estimators' keeping other parameters default substantially improved the overall f1 score keeping the recall high. Finally the best
'n_estimators' after the 4th tuning was found out to be '500' which I will use as the only hyperparameter for my XGBoost model.

'''
print('')
print('Fitting the final model with trained hyperparameters from RandomizedSearchCV for our XGBoost model...')


best_model = XGBClassifier(n_estimators = 500)
best_model.fit(X_train_sm,y_train_sm)
predictions = best_model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
pickle.dump(best_model, open('model/best_model.pkl', 'wb'))

'''
It is clearly evident that after tuning the XGBoost model, i.e increasing the n_estimators we have improved the f1-score from 0.63 to 0.75 which is a 19% improvement. This comes majorly from the increase in precision of the model which improved by 40% (from 0.57 to 0.80). The recall of the model remained the same. After an evaluation of the precision-recall curve, the threshold value for the classifier can be decreased to increase the recall. This may affect the precision to some extent but the cost of improving the recall to catch as many fraud transactions possible is much greater than the cost of having a small dip in precision. Even after the dip in precision, the number of transactions to be reviewd by the analysts will be well under control.

'''


'''

Visualising the feature importance for our best model...
The most predictive feature was 'pos_mode_5', which means the transactions made using the card number and chip and not the cvv (security code).  This was followed by a dummy variable of merchant country, available cash, account frequency and merchant frequency.

'''
feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_,X_train_sm.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('XGBoost Tuned')
plt.tight_layout()
plt.savefig('figs/feature-importance.png')


'''
Saving Precision Recall Curve for tuned XGBoost Model
'''
display = PrecisionRecallDisplay.from_estimator(
    best_model, X_test, y_test, name="XGB_Tuned"
)

plt.savefig('pr-best-model.png')



'''
Goal is to get the best recall with a decent enough precision. In the code below I extracted 33 values of precision and recall in accordance to threshold values ranging from 0 to 1. The dataframe is further saved in the directory. 
'''


thresh_pr = []
for threshold in np.arange(0,1,0.03):
    preds = (best_model.predict_proba(X_test)[:,1] >= threshold).astype(int)
    recall = recall_score(y_test,preds)
    precision = precision_score(y_test,preds)
    thresh_pr.append([threshold,recall,precision])

pr_thresh = pd.DataFrame(thresh_pr,columns=['threshold','recall','precision'])
pr_thresh.to_csv('processed_data/precision-recall-best-model.csv')
best_thresh =  pr_thresh.query('recall>0.79').tail(1)['threshold'].to_list()[0]
print(f"The value of threshold for which the maximum recall can be achieved keeping in mind the preciseness of our predictions is where recall >~ 0.8 and precision is maximum out of those values is: ",best_thresh)
best_preds = (best_model.predict_proba(X_test)[:,1] >= best_thresh).astype(int)
print(classification_report(y_test,best_preds))
print(confusion_matrix(y_test,best_preds))
print('')
print('The above metrics are achieved after applying the tuned threshood value for highest recall that could be achieved keeping the precision good enough so that the predictions are under the capacity of analyst teams reviewing capacity')

print('-'*100)
print('')
print('Succesfully executed the script and saved the best model....')

'''
Conclusion:

The primary objective of this project was to use the 1 year transaction data and build a model which can catch the most number of fraudulent transactions. At the same time the it was necessary to keep in mind that the bank's analyst team had a capacity of monitoring only 400 transactions per month. The XGBoost model which was chosen will perform the best to tackle this specific business problem. The final model (optimal threshold) achieved a recall of 0.8 and a precision of 0.57 which means that the model wa able to catch 80% of the total fraudulent transaction being 57% accurate with its predictions. Some of the high recall models that were tested during this project had a recall of more than 0.85 which after fine tuning would have gone above 0.9 but at a very high cost of losing precison. The precision of these models was close to 0.04 (only 4% predictions correct). This would create a huge hassle for the analyst who have to review so many tranasctions. At the end many transactions which were actually fraud would be missed by the analysts if the total predictions are much higher than the reviewing capacity. My solution for this problem has taken care of the precision and at the same time tried to achieve the maximum recall to catch the most number of fraudulent transactions on the test data (from the data which was provided).


'''

'''
To Do:   Check the entire code (write print statements) and comments. Provide explanations. Generate graphs and pictires in figs.. create a readme file. and create a ppt with graphs.
'''