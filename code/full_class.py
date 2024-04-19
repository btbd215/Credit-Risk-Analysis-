import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,classification_report,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce 
from scipy import stats
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from itertools import product
from sklearn.tree import DecisionTreeClassifier
import optuna
from scipy import stats

class Risk:
    def __init__(self,data,selection = False):
        self.data = data
        # encode = OrdinalEncoder(categories=[['good','bad']])
        # self.data['class'] = encode.fit_transform(self.data[['class']])
        if selection:
            self.data = self.feature_selection(self.data)
        self.X_train,self.X_test,self.y_train,self.y_test = self.split(self.data)

    def feature_selection(self,data):
        _,iv_df = self.data_vars(data.drop('class',axis = 1),data['class'])
        cols_to_drop = iv_df[iv_df['IV'] <= 0.02]['VAR_NAME'].unique()
        dropped_df = data.drop(cols_to_drop,axis = 1)
        return dropped_df
    
    def cotinuous_var(self,Y, X, n=20):
        df = pd.DataFrame({"X": X, "Y": Y})
        bins = pd.cut(df.X, n)
        binned = df.groupby(bins)
        
        d3 = pd.DataFrame()
        d3["COUNT"] = binned.count().Y
        d3["EVENT"] = binned.sum().Y
        d3["NONEVENT"] = d3.COUNT - d3.EVENT
        
        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"].replace([np.inf, -np.inf], 0, inplace=True)
        d3["VAR_NAME"] = "VAR"
        
        return d3

    def categorical_var(self,Y, X):
        df = pd.DataFrame({"X": X, "Y": Y})
        binned = df.groupby('X')
        d3 = pd.DataFrame()
        d3["COUNT"] = binned.count().Y
        d3["EVENT"] = binned.sum().Y
        d3["NONEVENT"] = d3.COUNT - d3.EVENT
        
        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"].replace([np.inf, -np.inf], 0, inplace=True)

        d3["VAR_NAME"] = "VAR"
        
        return d3

    def data_vars(self,df, target):
        iv_df = pd.DataFrame()
        
        for col in df.columns:
            if df[col].dtype != "object":
                conv = self.cotinuous_var(target, df[col])
            else:
                conv = self.categorical_var(target, df[col])
            conv["VAR_NAME"] = col
                
            iv_df = pd.concat([iv_df, conv], ignore_index=True)
        
        iv = iv_df.groupby('VAR_NAME')['IV'].sum().reset_index()
        
        return iv_df, iv
    
    def split(self,data):
        X = data.drop('class',axis = 1)
        y = data['class']
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 42, test_size= 0.2)
        return X_train, X_test, y_train, y_test
    
    
    def encode_categorical(self,encode_method = None):
        cate = self.data.select_dtypes(include = 'O')
        to_ordinal = ['checking_status','savings_status','employment']
        to_label = []
        for i in cate.columns:
            if i not in to_ordinal:
                if i == 'class':
                    continue
                to_label.append(i)
        # le = LabelEncoder()
        # self.y_train = le.fit_transform(self.y_train)
        # self.y_test = le.transform(self.y_test)
        if encode_method is None:
            cs = OrdinalEncoder(categories=[['no checking','<0', '0<=X<200','>=200']])
            ss = OrdinalEncoder(categories=[['no known savings', '<100', '100<=X<500','500<=X<1000', '>=1000']])
            emp = OrdinalEncoder(categories=[['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']])
            ord_list = [cs,ss,emp]
            for i, j in zip(to_ordinal,ord_list):
                self.X_train[i] = j.fit_transform(self.X_train[[i]])
                self.X_test[i] = j.transform(self.X_test[[i]])
            le = LabelEncoder()
            for i in to_label:
                self.X_train[i] = le.fit_transform(self.X_train[i])
                self.X_test[i] = le.transform(self.X_test[i])
        else: 
            encode = ce.WOEEncoder(cate.columns)
            encode.fit(self.X_train,self.y_train)
            self.X_train = encode.transform(self.X_train)
            self.X_test = encode.transform(self.X_test)
        
        return self.X_train,self.X_test
    
    def scale(self,scale_method = None, encode_method = None):
        if scale_method is None:
            scale = MinMaxScaler()
        else: 
            scale = StandardScaler()
        self.X_train = scale.fit_transform(self.X_train)
        self.X_test = scale.transform(self.X_test)
        return self.X_train, self.X_test

    def remove_outlier(self, df):
        to_remove = df['credit_amount']
        keep_indices = np.abs(stats.zscore(to_remove)) < 2
        self.X_train = self.X_train[keep_indices]
        self.y_train = self.y_train[keep_indices]
        return self.X_train, self.y_train

    def rebalance(self,X,y):
        smote = SMOTE(random_state=42)
        X_re,y_re = smote.fit_resample(X,y)
        return X_re,y_re
    
    def model_mapping(self,model):
        mapping = {
            'Logistics' : LogisticRegression,
            'KNN' : KNeighborsClassifier,
            'LightGBM' : lgb.LGBMClassifier,
            'AdaBoost': AdaBoostClassifier,
            'CatBoost': CatBoostClassifier,
            'XGBoost': xgb.XGBClassifier,
            'DecisionTree' : DecisionTreeClassifier,
            'Logistics Optuna' : LogisticRegression,
            'KNN Optuna' : KNeighborsClassifier,
            'LightGBM Optuna' : lgb.LGBMClassifier,
            'AdaBoost Optuna': AdaBoostClassifier,
            'CatBoost Optuna': CatBoostClassifier,
            'XGBoost Optuna': xgb.XGBClassifier,
            'DecisionTree Optuna' : DecisionTreeClassifier
        }
        return model,mapping[model]
    
    def train_report(self,model,X_train,y_train,X_test,y_test,param = None):
        clf = model(**param)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        y_pred_roc = clf.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test,y_pred_roc)
        return accuracy,recall,f1,precision,roc
    

    def suggest_parameter(self, trial, param_name, param_type, param_range):
        if param_type == 'int':
            return trial.suggest_int(name=param_name, low=param_range[0], high=param_range[1])
        elif param_type == 'float':
            if len(param_range) == 3:
                return trial.suggest_float(name=param_name, low=param_range[0], high=param_range[1], log=param_range[2])
            else:
                return trial.suggest_float(name=param_name, low=param_range[0], high=param_range[1])
            #return trial.suggest_float(name=param_name, low=param_range[0], high=param_range[1], log=param_range[2])
        elif param_type == 'categorical':
            return trial.suggest_categorical(name=param_name, choices=param_range)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    def objective(self, trial, param_search_space, model):
        param = {key: self.suggest_parameter(
            trial, key, value[0], value[1]) for key, value in param_search_space.items()}
        clf = model(**param)
        return cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3, scoring='f1').mean()

    def optimize_and_train_inner(self, model, param_search_space, n_trials=100):
        study = optuna.create_study(direction='maximize')

        def objective_wrapper(trial):
            return self.objective(trial, param_search_space, model)

        study.optimize(objective_wrapper, n_trials=n_trials)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        return study.best_trial.params

    def get_param_search_space(self, model):
        param_search_spaces = {
            LogisticRegression: {
                'C': ('float', (1e-5, 1e5, 'log')),  # Regularization parameter
                'penalty': ('categorical', ['l1', 'l2', None]),  # Regularization type
                'solver' : ('categorical', ['saga'])
            },
            KNeighborsClassifier: {
                'n_neighbors': ('int', (1, 50)),  # Number of neighbors
                'weights': ('categorical', ['uniform', 'distance']),  # Weight function used in prediction
                'algorithm': ('categorical', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithm used to compute the nearest neighbors
            },
            lgb.LGBMClassifier: {
                'num_leaves': ('int', (10, 50)),  # Maximum tree leaves for base learners
                'max_depth': ('int', (5, 20)),  # Maximum tree depth for base learners
                'learning_rate': ('float', (0.01, 0.1, 'log')),  # Boosting learning rate
                'n_estimators': ('int', (50, 200)),  # Number of boosting iterations
                'reg_alpha': ('float', (0, 1)),  # L1 regularization term on weights
                'reg_lambda': ('float', (0, 1))  # L2 regularization term on weights
            
            },
            AdaBoostClassifier: {
                'n_estimators': ('int', (49, 200)),  # Number of boosting iterations
                'learning_rate': ('float', (0.01, 1.1, 'log')),  # Boosting learning rate
                'algorithm': ('categorical', ['SAMME', 'SAMME.R'])  # Boosting algorithm
            },
            CatBoostClassifier: {
                'iterations': ('int', (50, 200)),  # Number of boosting iterations
                'learning_rate': ('float', (0.01, 0.1, 'log')),  # Boosting learning rate
                'depth': ('int', (4, 10)),  # Depth of trees
            },
            xgb.XGBClassifier: {
                'learning_rate': ('float', (0.01, 0.1, 'log')),  # Boosting learning rate
                'n_estimators': ('int', (50, 200)),  # Number of boosting iterations
                'max_depth': ('int', (3, 10)),  # Maximum tree depth
                'min_child_weight': ('int', (1, 10)),  # Minimum sum of instance weight needed in a child
                'subsample': ('float', (0.5, 1)),  # Subsample ratio of the training instances
                'colsample_bytree': ('float', (0.5, 1)),  # Subsample ratio of columns when constructing each tree
            },
            DecisionTreeClassifier: {
                'max_depth': ('int', (3, 20)),  # Maximum tree depth
                'min_samples_split': ('int', (2, 10)),  # Minimum number of samples required to split an internal node
                'min_samples_leaf': ('int', (1, 10)),  # Minimum number of samples required to be at a leaf node
                'criterion': ('categorical', ['gini', 'entropy'])  # Function to measure the quality of a split
            }
        }
        return param_search_spaces.get(model, {})

    def optimize_and_train(self, model, n_trials=100):
        param_search_space = self.get_param_search_space(model)
        return self.optimize_and_train_inner(model, param_search_space, n_trials)

    def model_selection(self,model = 'Logistic',outlier = True, balance = True, encode_method = None, scale_method = None, param = None):
        model_name,model_func = self.model_mapping(model)
        if encode_method is not None:
            enc = 'woe'
        else:
            enc = 'ordinal'
        if scale_method is not None:
            sca = 'standard'
        else:
            sca = 'minmax'
        if outlier:
            self.X_train,self.y_train = self.remove_outlier(self.X_train)
            out = 'yes'
        else:
            out = 'no'
        self.X_train,self.X_test = self.encode_categorical(encode_method=encode_method)
        self.X_train,self.X_test = self.scale(scale_method = scale_method)
        if balance:
            bal = 'yes'
            self.X_train, self.y_train = self.rebalance(self.X_train,self.y_train)
        else:
            bal = 'no'
        if 'Optuna' in model_name:
            param = self.optimize_and_train(model_func, n_trials=100)
            print('Best parameters:', param)
        acc,rec,f1,pre,roc = self.train_report(model = model_func, X_train = self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test,param = param)
        return out,bal,enc,sca,acc,rec,f1,pre,roc
    
def main(model,data,selection = True,outlier = True, balance = True,encode_method = None, scale_method = None,param = None):
        result_df = pd.DataFrame(columns = ['model','remove_outlier','rebalance_data','encode_method','scale_method','accuracy', 'recall', 'f1_score', 'precision', 'roc_auc'])
        #risk = Risk(data=data)
        for o,b,ec,sc in product([outlier],[balance],[encode_method],[scale_method]):
            risk = Risk(data=data,selection=selection)
            out,bal,enc,sca,acc,rec,f1,pre,roc = risk.model_selection(model = model, outlier = o, balance = b,encode_method = ec, scale_method = sc, param = param)
            if len(param) > 2:
                model_str = f'{model}_{param}'
            else:
                model_str = model
            result_df_length = len(result_df)
            result_df.loc[result_df_length] = [model_str,out,bal,enc,sca,acc,rec,f1,pre,roc]
        return result_df

if __name__ == '__main__':
        data  = pd.read_csv('credit_customers.csv')
        encode = [None,'minmax']
        scale = [None, 'standard']
        out = [True, False]
        bal= [True,False]
        params = {
            'Logistics' : {},
            # 'KNN' : {},
            # 'CatBoost' : {},
            # 'XGBoost' : {},
            # 'AdaBoost' :{},
            # 'LightGBM':{},
            # 'DecisionTree' : {}
        }

        # result_df = pd.DataFrame(columns = ['model','remove_outlier','rebalance_data','encode_method','scale_method','accuracy', 'recall', 'f1_score', 'precision', 'roc_auc'])
        # for model_name,param in params.items():
        #     result = main(model=model_name,data=data,outlier=out,balance=bal,encode_method=encode,scale_method=scale,param=param)
        #     result_df = pd.concat([result_df,result],ignore_index=True)
        # print(result_df)

        list_model_optuna = ['XGBoost Optuna']
        for i in list_model_optuna:
            main(model=i,data=data,outlier=True,balance=True,scale_method='minmax',selection=False)




            
        
        








        

        


        

        
        
        
        
  
