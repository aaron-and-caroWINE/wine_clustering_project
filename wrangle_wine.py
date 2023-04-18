# import s
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

# function to get red wine
def get_red_wine():
    '''
    Actions: opens red wine csv and adds an identifier column
    '''
    # reads csv and assigns to a variable
    red_wine_df = pd.read_csv('winequality-red.csv')
    
    # add a column to identify the wine type
    red_wine_df['red_or_white'] = 'red'

    # exit the function and return the dataframe
    return red_wine_df

# functoi. to get the red wine dataset
def get_white_wine():
    '''
    Actions: opens white wine csv and adds an identifier column
    '''
    # reads the csv file and assigns to a variable
    white_wine_df = pd.read_csv('winequality-white.csv')

    # add a column to identify the wine type
    white_wine_df['red_or_white'] = 'white'

    # exit function and returns dataframe
    return white_wine_df

def merge_wine():
    '''
    Actions: gets red wine and white wine dataframes and merges them together
    '''
    # gets red wine and white wine dataframes and merges them together
    df = pd.merge(get_red_wine(), get_white_wine(), how='outer')

    # exit function and return merged df
    return df


def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    # adds all the nulls for each row and assigns it to a variable
    num_missing = df.isnull().sum(axis=1)
    
    # divides the total nulls missing by the number of features; this is mulitplied by 100 to get the percent of rows missing
    percent_miss = num_missing / df.shape[1] * 100
    
    # create a dataframe with both variables from above and assigns to a variable
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    
    
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    
    # returns the dataframe with the largest numbers first
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    # adds all the nulls for each row and assigns it to a variable
    num_missing = df.isnull().sum()
    
    # assigne the number of rows to a variable
    rows = df.shape[0]
    
    # calculates the percentage of the column that's missing
    percent_missing = num_missing / rows * 100
    
    # creates a dataframe using the actual number ofmissing values and the percetage of the column that is missing
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    
    # returns the dataframe with the largest numbers first
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
    '''
    Actions: removies outliers using the IQR with a default k of 1.5
    '''
    # initialize dictionary
    col_qs = {}
    
    # assign column names to variable
    df_cols = df.columns
    
    # creates a list of column names
    df_cols = df_cols.to_list()
    
    # remove cat cols
    df_cols.remove('red_or_white')

    # for each column
    for col in df_cols:
        
        # create qualtiles and put them in a dict
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])

    # for each col
    for col in df_cols:    
        
        # calculate the iqr
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        
        # calculate the lower fence
        lower_fence = col_qs[col][0.25] - (iqr*k)
        
        # calculates the upper fence
        upper_fence = col_qs[col][0.75] + (iqr*k)
        
        # remove outliers from df for each col
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
        
    # exit df and return new df
    return df


def split_data(df, target):
    '''
    split data takes in a dataframe or function which returns a dataframe
    and will split data based on the values present in a cleaned 
    version of the dataframe. Also you must provide the target
    at which you'd like the stratify (a feature in the DF)
    '''
    # split for test
    train_val, test = train_test_split(df, 
                                       train_size=.8,
                                       random_state=1349, 
                                       stratify=df[target])
    
    # split for train, validate
    train, validate = train_test_split(train_val, 
                                       train_size=0.7,
                                       random_state=1349,
                                       stratify=train_val[target])
    # return train, test, validate
    return train, validate, test


def get_dummies(df):
    '''
    Encodes variables
    '''
    # adds encoded variable to df
    df = pd.concat(
        [df, pd.get_dummies(df[['red_or_white']], 
                            drop_first=True)], axis=1)
    
    # removes non-encoded columns
    df = df.drop(columns=['red_or_white'])
    
    # changes the name of the encoded variable column
    df = df.rename(columns={'red_or_white_white': 'white_wine'})
    
    # exits function and returns df
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

    columns_to_scale = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                        'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
                        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 
                        'alcohol']

    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #     make the thing
    scaler = MinMaxScaler()
    
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(train[columns_to_scale]),
        columns=train[columns_to_scale].columns.values, 
        index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(validate[columns_to_scale]),
        columns=validate[columns_to_scale].columns.values).set_index(
        [validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(test[columns_to_scale]),
        columns = test[columns_to_scale].columns.values).set_index(
        [test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def wrangle_wine():
    '''
    
    '''
    
    # acquire dataframes from locally saved version and merge
    df = pd.merge(get_red_wine(), get_white_wine(), how='outer')

    # ername columns for pyhton friendly names
    df = df.rename(columns={'fixed acidity': 'fixed_acidity',
                             'volatile acidity': 'volatile_acidity',
                               'citric acid': 'citric_acid',
                               'residual sugar': 'residual_sugar',
                               'free sulfur dioxide': 'free_sulfur_dioxide',
                                'total sulfur dioxide': 'total_sulfur_dioxide'})

    # removes outliers from all columns except for the string 'red_or-white' column
    df = remove_outliers(df, k=1.5)

    # get dummies of the 'red_or-white' column
    df = get_dummies(df)

    # split into train, validate, test
    train, validate, test = split_data(df, 'quality')

    # scale data
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, return_scaler=False)

    # return
    return train, validate, test, train_scaled, validate_scaled, test_scaled