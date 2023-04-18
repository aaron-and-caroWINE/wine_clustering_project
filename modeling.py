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
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

X_train_scaled, X_validate_scaled, X_test_scaled, X_train, X_validate, X_test, y_train, y_validate, y_test = p.preprocess_clustering_wine()

def classification_eval_dict():

    '''
    Creates a dictionary containing the evaluation metrics for the classification models used with the 
    clusters as a feature. Publishes the transposed dictionary as a pandas dataframe.
    '''

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

    '''
    Creates a dictionary containing the evaluation metrics for the linear regression models used 
    with the clusters as a feature. Publishes the transposed dictionary as a pandas dataframe.
    '''

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

def clustering_viz():

    '''
    Makes a clustering model using 4 centroids with the specified mdeling features to create clusters
    that will be used for modelig features. The function then creates two subplots visualizing the 
    difference in the clusters assigned to the actual target quality rating.
    '''
  
    # lets make a clustering object from sklearn
    clustering_feats = ['alcohol', 'density', 'chlorides', 'volatile_acidity']
    # Make a thing! Thats my favorite!
    k_means_prototype = KMeans(n_clusters=4)
    # fit the thing!!!!
    k_means_prototype.fit(X_train_scaled[clustering_feats])
    # use the thing
    clusters = k_means_prototype.predict(
        X_train_scaled[clustering_feats])
    
    X_train_scaled['cluster_assigned'] = clusters

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    sns.set_palette('rocket')

    for quality, subset in X_train_scaled.groupby('quality'):
        axs[0].scatter(subset.alcohol,
                    subset.density,
                    label=quality)
    axs[0].legend()
    axs[0].set(title='Actual Quality')

    for cluster in X_train_scaled.cluster_assigned.unique():
        axs[1].scatter(X_train_scaled[X_train_scaled.cluster_assigned == cluster].alcohol,
                X_train_scaled[X_train_scaled.cluster_assigned == cluster].density,
                label=cluster)
    axs[1].legend()
    axs[1].set(title='Cluster Assignment')

    plt.show()

def test_best_model():

    '''
    Creates a random forest model with a max depth of 4 to run the test dataset on. The accuracy score
    is then generated using the test dataset.
    '''

    rf = RandomForestClassifier(max_depth=4)
    rf.fit(X_train, y_train['quality'])
    test_score = rf.score(X_test, y_test['quality'])
    print(f'Test Dataset Accuracy Score: {test_score}')




