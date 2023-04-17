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

from IPython.display import display

def get_red_wine():

    red_wine_df = pd.read_csv('winequality-red.csv')

    red_wine_df['red_or_white'] = 'red'

    return red_wine_df

def get_white_wine():

    white_wine_df = pd.read_csv('winequality-white.csv')

    white_wine_df['red_or_white'] = 'white'

    return white_wine_df

def merge_wine():

    df = pd.merge(get_red_wine(), get_white_wine(), how='outer')

    return df

def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('                    SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    display(pd.DataFrame(df.head(3)))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    display(pd.DataFrame(df.info()))
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    display(pd.DataFrame(df.describe().T))
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            display(pd.DataFrame(df[col].value_counts()))
        else:
            display(pd.DataFrame(df[col].value_counts(bins=10, sort=False)))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    display(pd.DataFrame(nulls_by_col(df)))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    display(pd.DataFrame(nulls_by_row(df)))
    print('=====================================================')


def remove_outliers(df, k=1.5):
    col_qs = {}
    df_cols = df_cols=df.columns
    df_cols = df_cols.to_list()
    df_cols.remove('red_or_white')

    for col in df_cols:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
        # print(col_qs)
    
    for col in df_cols:    
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (iqr*k)
        upper_fence = col_qs[col][0.75] + (iqr*k)
        #print(f'Lower fence of {col}: {lower_fence}')
        #print(f'Upper fence of {col}: {upper_fence}')
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

def split_data(df, target):
    '''
    split data takes in a dataframe or function which returns a dataframe
    and will split data based on the values present in a cleaned 
    version of the dataframe. Also you must provide the target
    at which you'd like the stratify (a feature in the DF)
    '''
    train_val, test = train_test_split(df, 
                                       train_size=.8,
                                       random_state=1349, 
                                       stratify=df[target])
    train, validate = train_test_split(train_val, 
                                       train_size=0.7,
                                       random_state=1349,
                                       stratify=train_val[target])
    return train, test, validate

def get_dummies(df):
    df = pd.concat(
        [df, pd.get_dummies(df[['red_or_white']], 
                            drop_first=True)], axis=1)
    df = df.drop(columns=['red_or_white'])
    df = df.rename(columns={'red_or_white_white': 'white_wine'})
    print(df)
    return df

def scale_data(train, 
               validate, 
               test,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''

    columns_to_scale = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values, 
                                                  index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def wrangle_wine():

    # acquire dataframes from locally saved version and merge
    df = pd.merge(get_red_wine(), get_white_wine(), how='outer')

    # removes outliers from all columns except for the string 'red_or-white' column
    df = remove_outliers(df, k=1.5)

    # get dummies of the 'red_or-white' column
    df = get_dummies(df)

    # split into train, validate, test
    train, validate, test = split_data(df, 'quality')

    # scale data
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, return_scaler=False)

    return train, validate, test, train_scaled, validate_scaled, test_scaled

# --------------------- Visualizations ------------------------

def bivariate_visulization(df, target):
    
    cat_cols, num_cols = [], []
    
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else: 
                num_cols.append(col)
                
    print(f'Numeric Columns: {num_cols}')
    print(f'Categorical Columns: {cat_cols}')
    explore_cols = cat_cols + num_cols

    for col in explore_cols:
        if col in cat_cols:
            if col != target:
                print(f'Bivariate assessment of feature {col}:')
                sns.barplot(data = df, x = df[col], y = df[target], palette='crest')
                plt.show()

        if col in num_cols:
            if col != target:
                print(f'Bivariate feature analysis of feature {col}: ')
                plt.scatter(x = df[col], y = df[target], color='turquoise')
                plt.axhline(df[target].mean(), ls=':', color='red')
                plt.axvline(df[col].mean(), ls=':', color='red')
                plt.show()

    print('_____________________________________________________')
    print('_____________________________________________________')
    print()

def univariate_visulization(df):
    
    cat_cols, num_cols = [], []
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 5:
                cat_cols.append(col)
            else: 
                num_cols.append(col)
                
    explore_cols = cat_cols + num_cols
    print(f'cat_cols: {cat_cols}')
    print(f'num_cols: {num_cols}')
    for col in explore_cols:
        
        if col in cat_cols:
            print(f'Univariate assessment of feature {col}:')
            sns.countplot(data=df, x=col, color='turquoise', edgecolor='black')
            plt.show()

        if col in num_cols:
            print(f'Univariate feature analysis of feature {col}: ')
            plt.hist(df[col], color='turquoise', edgecolor='black')
            plt.show()
            df[col].describe()
    print('_____________________________________________________')
    print('_____________________________________________________')
    print()

def viz_explore(train, target):

    univariate_visulization(train)

    bivariate_visulization(train, target)

    corr = train.corr(method='spearman')
    plt.figure(figsize=(20,15))
    plt.rc('font', size=14)
    sns.heatmap(corr, cmap='crest', annot=True, mask=np.triu(corr))
    plt.show()
    

