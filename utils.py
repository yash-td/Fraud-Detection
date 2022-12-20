
import pandas as pd
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.model_selection import train_test_split


# helper functions

'''
get_pod() takes one argument 'x', which will be the transactionTime column from the dataset. It will then create
6 equal time intervals namely Early Morning, Morning, Noon, Evening, Night and Late Night according to the time 
of the day. Here 'pod' is part-of-day. 
'''
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

'''
class_dist() takes 3 arguments -> X,y and target_var. Here X are the features, y is the label and target_var is the name
of the target variable. This function is basically used to show the class distribution before sampling.
'''
def class_dist(X, y, target_var): 
    # concatenate our training data back together
    resampling = X.copy()
    resampling[target_var] = y.values
    # separate minority and majority classes
    majority_class = resampling[resampling[target_var]==0]
    minority_class = resampling[resampling[target_var]==1]
    # Get a class count to understand the class imbalance.
    print('majority_class before sampling: '+ str(len(majority_class)))
    print('minority_class before sampling: '+ str(len(minority_class)))
    return majority_class, minority_class

'''
upsample_SMOTE() takes 4 arguments -> X,y and target_var and ratio. Here X are the features, y is the label, target_var is
the name of the target variable and ratio is the ratio to which the minority class will be upsampled. This function is used
to perform the SMOTE upsampling to our minority class. It returns X_sm and y_sm which are the upsampled features and labels.
'''

def upsample_SMOTE(X, y, target_var, ratio):
    """Upsamples minority class using SMOTE.
    Ratio argument is the percentage of the upsampled minority class in relation
    to the majority class. Default is 1.0
    """
    sm = SMOTE(random_state=42, sampling_strategy=ratio)
    X_sm, y_sm = sm.fit_resample(X, y)
    resampling = X_sm.copy()
    resampling[target_var] = y_sm.values
    majority_class = resampling[resampling[target_var]==0]
    minority_class = resampling[resampling[target_var]==1]
    print('majority_class after sampling: '+ str(len(majority_class)))
    print('minority_class after sampling: '+ str(len(minority_class)))

    return X_sm, y_sm


'''
get_frequency() takes 3 arguments: data, column and new_var. It is used to extract frequency from the features in our data
which have a high cardinality but not so many unique values, for example accountNumber and mcc. It returns data after creating 
the new feature, adding it to our feature_set and removing the original feature.
'''
def get_frequency(data,column,new_var):
    freq = column.value_counts()
    freq = pd.DataFrame(freq)
    freq["name"] = freq.index
    freq.rename(columns = {column.name:new_var, 'name':column.name}, inplace = True)
    data = data.merge(freq, on = column.name)
    data = data.drop(columns=[column.name])
    return data

'''
create_dummies() takes 3 arguments: data, column and prefix. It is created to avoid redundant code in our main script where
we perform one hot encoding (using pd.get_dumies) for teh categorical features. It returns data after creating the dummy variables,
adding them to our main feature_set and also dropping the original categorical column.
'''
def create_dummies(data,column,prefix):
    dummies = pd.get_dummies(data[column.name],prefix = prefix)
    data = pd.concat([data,dummies],axis=1)
    data = data.drop(columns=[column.name])
    return data

'''
prepare_data() takes only one argumrnt which is data. It is used to create training, testing and validation sets for our prediction
models. While using the train_test_split(), I have specified stratify=y for both the splits. This means that my test and validation
sets will have class distribution (majority:minority).
'''
def prepare_data(data):
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']
    X_train, X_rest, y_train, y_rest  = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.75, random_state=42, stratify=y_rest)

    return X_train,X_test,X_val,y_train,y_test,y_val
