from utils import *
import pickle
import warnings
warnings.filterwarnings("ignore")


processed_df = pd.read_csv('processed_df.csv').drop(columns=['Unnamed: 0'])

X_train,X_test,y_train,y_test = prepare_data(processed_df)

print('Performing Grid Search with KFold cross validation on the best model to fine tune it...')
print('')


params = {'xgb__max_depth': [1,2,3,4,5,6], 'xgb__min_child_weight': [1,2,3,4]} 
fix_params = {'xgb__learning_rate': 0.2, 'xgb__n_estimators': 100, 'xgb__objective': 'binary:logistic'} 


params_2nd_run = {'xgb__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]}
fix_params_2nd_run = {'xgb__n_estimators': 100, 'xgb__objective': 'binary:logistic', 'xgb__max_depth': 6, 'mxgb__in_child_weight':2}

params_3rd_run = {'xgb__n_estimators': [50,100,300,500]}
fix_params_3rd_run = {'objective': 'binary:logistic', 'max_depth': 6, 'min_child_weight':2,'xgb__learning_rate':0.35}


params_estimators = {'xgb__n_estimators':[10,100,500,1000]}
best_model = fine_tune_model(XGBClassifier(),X_train,y_train,X_test,y_test,params_estimators,'f1')

pickle.dump(best_model, open('model_f1.pkl', 'wb'))

'''
best parameters 1st tuning --> {'xgb__min_child_weight': 2, 'xgb__max_depth': 6} 
these will now be added to fix_params..
fix_params = {'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'binary:logistic', 'max_depth': 6, 'min_child_weight':2}

best parameters 2nd tuning --> {'xgb__learning_rate': 0.35}

best parameters 3rd tuning --> {'xgb__n_estimators':500}

Combination of the parameters from first 2 tunings did not improve the overall score of the model. Instead solely tuning the parameter 'n_estimators' keeping other 
parameters default substantially improved the overall f1 score keeping the recall high.
'''