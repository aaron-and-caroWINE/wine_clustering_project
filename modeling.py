import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import numpy as np
import wrangle_wine as w
import preprocess as p

X_train_scaled, X_validate_scaled, X_test_scaled, X_train, X_validate, X_test, y_train, y_validate, y_test = p.preprocess_clustering_wine()

def classification_eval_dict():

    class_model_dict = {
    'Baseline':
    {
        'train_score': 0.467, 
        'validate_score': 0.467,
        'difference': 0.000
    },
    'DecisionTreeClassifier(max_depth=12)':
    {
        'train_score': round(0.905395, 3),
        'validate_score': round(0.595657, 3), 
        'difference': round(0.309739, 3)
    
    },
    'RandomForestClassifier(max_depth=8, min_samples_leaf=3)':
    {
        
        'train_score': round(0.74834, 3),
        'validate_score': round(0.58738, 3), 
        'difference': round(0.16096, 3)
    },
    'Logistic Regression':
    {
        'train_score': round(0.5698447893569845, 3),
        'validate_score': round(0.5491209927611168, 3),
        'difference': round((0.5698447893569845 - 0.5491209927611168), 3)
        
    },
    'KNeighborsClassifier(n_neighbors=6)':
    {
        'train_score': 0.693,
        'validate_score': 0.426,
        'difference': 0.267
    }
}

    class_model_dict = pd.DataFrame(class_model_dict).T

    return class_model_dict

def regression_eval_dict():

    regress_model_dict = {
    'Baseline':
    {
        'RMSE_train': 0.786, 
        'RMSE_validate': 0.785,
        'R2_validate': 0.000
    },
    'OLS_Regressor':
    {
        'RMSE_train': 0.707, 
        'RMSE_validate': 0.723,
        'R2_validate': 0.107
    
    },
    'LassoLars(alpha=1)':
    {
        
        'RMSE_train': 0.786, 
        'RMSE_validate': 0.785,
        'R2_validate': 0.000
    },
    'TweetieRegressor_GLM(power=1, alpha=0)':
    {
        'RMSE_train': 0.707, 
        'RMSE_validate': 0.724,
        'R2_validate': 0.106
        
    },
    'Polynomial_Regression':
    {
        'RMSE_train': 0.679, 
        'RMSE_validate': 0.714,
        'R2_validate': 0.127
    }
}

    regress_model_dict = pd.DataFrame(regress_model_dict).T

    return regress_model_dict


    





