
import pandas as pd
from numpy import argmax
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import AdaBoostClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

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
get_month() takes one argument 'x' which is the column transactionTime and simply return the month from the datatime data type.
'''
def get_month(x):
    return x.month

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
    print('')
    print('majority_class before sampling: '+ str(len(majority_class)))
    print('')
    print('minority_class before sampling: '+ str(len(minority_class)))
    print('')
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
    print('')
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
prepare_data() takes only one argumrnt which is data. It is used to create training,= and testing sets for our prediction
models. While using the train_test_split(), I have specified stratify=y. This means that my test set will have same class distribution
as my train set (majority:minority).
'''
def prepare_data(data):
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train,X_test,y_train,y_test

'''
base_models() takes 4 arguments. It takes the balanced training features and labels and also the teseting features and labels. It trains
5 base models and returns their precision,recall and f1-score.

'''

def base_models(X_train_sm,y_train_sm,X_test,y_test):

    models = {'LogisticRegression': LogisticRegression(),
    'DecisionTree':DecisionTreeClassifier(),
    'RandomForest':RandomForestClassifier(),
    'XGBoost':XGBClassifier(),
    'ADABoost': AdaBoostClassifier()
    }

    f1 = []
    fitted_models = []
    for u,v in models.items():
        b_model = v.fit(X_train_sm,y_train_sm)
        preds_b = b_model.predict(X_test)
        f1.append([u,precision_score(y_test,preds_b),recall_score(y_test,preds_b),f1_score(y_test,preds_b), average_precision_score(y_test,preds_b)])
        fitted_models.append(b_model)
    print('-'*70)
    print('Model Scores...')
    f1 = pd.DataFrame(f1,columns=['models','precision_score','recall_score','f1_score','pr_auc'])
    print(f1)

    return f1, fitted_models


'''
best_metrics() takes 2 arguments: data and metric. It returns the best model according to the metric passed. The data here is the dataframe of the results of 5
base models that are fitted. 
'''   
def best_metrics(data,metric):
    max_m = max(data[metric])
    res = f"The model with the best {metric} {max_m} is {data['models'][data[metric].idxmax()]}"
    return res

'''
best_f1_pr_auc() takes 4 arguments: x is the model, name is the model name, and the other two arguments is the test data and labels. This function returns the 
best f1 score achieved for a model among all the threshold values of the classifiers. It also plots and saves the precision recall curve with the best threshold point.
'''

def best_f1_pr_auc(x,name,X_test,y_test):
    yhat = x.predict_proba(X_test)
    yhat = yhat[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    print(f'Best Threshold = {thresholds[ix]} and best F-Score {round(fscore[ix],3)} achieved for {name}')
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label=f'Baseline score {round(no_skill,3)}')
    plt.plot(recall, precision, marker='.', label=name)
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'figs/{name}.png')
    return thresholds[ix], fscore[ix]


'''
fit_model() takes 5 arguments: one is model and the other four are the train and test data sets. It simply trains the model, creates classification report, 
creates confusion matrix and returns the trained model and predictions. Throughout the script I have trained many models, hence this function helps remove the redundant
code making the script efficient.
'''
def fit_model(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print('Classification Report:')
    print(classification_report(y_test,pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,pred))
    return model,pred


'''
fine_tune_model() takes 7 arguments: model, train-test sets, parameter grid to be tuned and the metric to optimise during the tuning. This functions performs performs
hyperparameter tuning using RandomnisedSearchCV and uses imbpipeline to train the model using StratifiedKFold cross validation. This functions prints the best parameters,
confusion matrix, classification report and returns just the trained model.
'''
def fine_tune_model(model,X_train,y_train,X_test,y_test,param_grid,metric):


    model =  imbpipeline(steps = [['smote', SMOTE()],['xgb', model]])

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = RandomizedSearchCV(
        model, 
        param_grid,
        scoring=metric,
        cv=stratified_kfold, verbose=10)

    tqdm(param_grid.fit(X_train, y_train))

    print('Best parameters found:\n', param_grid.best_params_)

    print(param_grid.best_score_)
    rfc_grid_predict = param_grid.predict(X_test)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,rfc_grid_predict))
    print('Classification Report:')
    print(classification_report(y_test,rfc_grid_predict))

    return param_grid