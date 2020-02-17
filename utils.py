# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats

def drop_rows_cloumns(df,unknown_0,unknown_9,unknown_neg1,transactions_0,transactions_10, online_0):
    print('****** Step 1 - dropping rows *****')
    print(f'Number of rows before dropping: {df.shape[0]}')
    df=df[df.isnull().sum(axis=1) <= 17].reset_index(drop=True)
    print(f'Number of rows after dropping: {df.shape[0]}\n')
    
    print('****** Step 2 - dropping columns *****')
    print(f'Number of features before dropping: {df.shape[1]}')
    
    #before dropping, filling NaN values for each columns 
    #that NaN values indicated -1, 0, 9 and 10
    # fill unknown_df
    df[unknown_0]=df[unknown_0].fillna(0)
    df[unknown_9]=df[unknown_9].fillna(9)
    df[unknown_neg1]=df[unknown_neg1].fillna(-1)

    # fill transactions_df
    df[transactions_0]=df[transactions_0].fillna(0)
    df[transactions_10]=df[transactions_10].fillna(10)

    # fill online_transactions_df
    df[online_0]=df[online_0].fillna(0)

    #drop 8 features: 6 features that percentage missing is over 65% 
    #and 2 features that contains unnecessary different items:
    drop_cols= ['ALTER_KIND1', 
                'ALTER_KIND2', 
                'ALTER_KIND3', 
                'ALTER_KIND4', 
                'EXTSEL992',
                'LNR',
                'KK_KUNDENTYP', 
                'D19_LETZTER_KAUF_BRANCHE',
                'EINGEFUEGT_AM']
    df.drop(drop_cols, axis=1, inplace=True)
    print(f'Number of features after dropping: {df.shape[1]}\n')
    return df

def remove_highly_correlated_columns(df, correlation):
    print('****** Step 3 - Remove highly correlated features *****')
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than correlation coefficient
    to_drop = [column for column in upper.columns if any(upper[column] > correlation)]
    print(f'Number of features before removing highly correlated columns: {df.shape[1]}')
    df.drop(to_drop, axis=1, inplace=True)
    print(f'Number of features after removing highly correlated columns: {df.shape[1]}')
    return df
    
    
def encode(df):
    #encode values using get_dummies
    # CAMEO_DEU_2015, CAMEO_INTL_2015 replaced by -1
    print('****** Step 4 - Encode Categorical Features *****')
    
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].replace('XX', -1)
    df['CAMEO_DEU_2015'] =df['CAMEO_DEU_2015'].replace('XX', -1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].fillna(-1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].astype(int)
    
    # CAMEO_DEUG_2015 replaced by -1
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].replace('X', -1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].fillna(-1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].astype(int)
    
    # NaN values OST_WEST_KZ replaced by -1, 
    #followed by encoding 1 and 2 for O and W, respectively
    df['OST_WEST_KZ']=df['OST_WEST_KZ'].fillna(-1)
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({'O':1, 'W':2})
    df['ALTER_HH']=df['ALTER_HH'].fillna(0)
    
    #with  all null data now handled, we should focus on getting
    #objects/categorical variables to numbers via one hot encoding
    print(f'number of features before encoding categorical: { df.shape[1]}')
    df = pd.get_dummies(df, drop_first=True)
    print(f'number of features agter encoding categorical:{ df.shape[1]} \n')
    
    return df


def impute(df):
    print('****** Step 5 - Impute values *****')
    # impute nans using mode value
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df = pd.DataFrame(imp.fit_transform(df))
    print(f'check NaN values in dataset: {(df.isnull().sum()>0).sum()}')
    return df

    
def transform(df, method, columns):
    print('****** Step 6 - Transform Dataset *****')
    if(method=='StandardScaler'):
        scale = StandardScaler()
        df=pd.DataFrame(scale.fit_transform(df), columns=columns)
    elif (method == 'MinMaxScaler'):
        scale = MinMaxScaler()
        df = pd.DataFrame(scale.fit_transform(df), columns=columns)
        
    return df

def clean_data_for_supervised_learning(df,unknown_0,unknown_9,unknown_neg1,transactions_0,transactions_10, online_0, correlation, method):   
    print('****** Step 1 - dropping columns *****')
    print(f'Number of features before dropping: {df.shape[1]}')
    
    #before dropping, filling NaN values for each columns 
    #that NaN values indicated -1, 0, 9 and 10
    # fill unknown_df
    df[unknown_0]=df[unknown_0].fillna(0)
    df[unknown_9]=df[unknown_9].fillna(9)
    df[unknown_neg1]=df[unknown_neg1].fillna(-1)

    # fill transactions_df
    df[transactions_0]=df[transactions_0].fillna(0)
    df[transactions_10]=df[transactions_10].fillna(10)

    # fill online_transactions_df
    df[online_0]=df[online_0].fillna(0)

    #drop 8 features: 6 features that percentage missing is over 65% 
    #and 2 features that contains unnecessary different items:
    drop_cols= ['ALTER_KIND1', 
                'ALTER_KIND2', 
                'ALTER_KIND3', 
                'ALTER_KIND4', 
                'EXTSEL992',
                'KK_KUNDENTYP', 
                'D19_LETZTER_KAUF_BRANCHE',
                'EINGEFUEGT_AM']
    df.drop(drop_cols, axis=1, inplace=True)
    print(f'Number of features after dropping: {df.shape[1]}\n')
  
    #encode values using get_dummies
    # CAMEO_DEU_2015, CAMEO_INTL_2015 replaced by -1
    print('****** Step 2 - Encode Categorical Features *****')
    
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].replace('XX', -1)
    df['CAMEO_DEU_2015'] =df['CAMEO_DEU_2015'].replace('XX', -1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].fillna(-1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].astype(int)
    
    # CAMEO_DEUG_2015 replaced by -1
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].replace('X', -1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].fillna(-1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].astype(int)
    
    # NaN values OST_WEST_KZ replaced by -1, 
    #followed by encoding 1 and 2 for O and W, respectively
    df['OST_WEST_KZ']=df['OST_WEST_KZ'].fillna(-1)
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({'O':1, 'W':2})
    df['ALTER_HH']=df['ALTER_HH'].fillna(0)
    
    #with  all null data now handled, we should focus on getting
    #objects/categorical variables to numbers via one hot encoding
    print(f'number of features before encoding categorical: { df.shape[1]}')
    df = pd.get_dummies(df, drop_first=True)
    print(f'number of features agter encoding categorical:{ df.shape[1]} \n')
    #get columns header list
    columns = df.columns 
    
    print('****** Step 3 - Impute values *****')
    # impute nans using mode value
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df = pd.DataFrame(imp.fit_transform(df))
    print(f'check NaN values in dataset: {(df.isnull().sum()>0).sum()} \n')

    if(method=='StandardScaler'):
        print('****** Step 4 - Transform Dataset *****')
        scale = StandardScaler()
        df=pd.DataFrame(scale.fit_transform(df), columns=columns)
    elif (method == 'MinMaxScaler'):
        print('****** Step 6 - Transform Dataset *****')
        scale = MinMaxScaler()
        df = pd.DataFrame(scale.fit_transform(df), columns=columns)
    #finally set index of dataframe by column LNR, LNR contain unique values for each customers
    df = df.set_index('LNR')
    print(f'Shape after clean data: {df.shape}')
    print('FINISH CLEAN DATASET \n')
    return df


def clean_data_for_unsupervised_learning(df,unknown_0,unknown_9,unknown_neg1,transactions_0,transactions_10, online_0, correlation, method):
    print('****** Step 1 - dropping rows *****')
    print(f'Number of rows before dropping: {df.shape[0]}')
    df=df[df.isnull().sum(axis=1) <= 17].reset_index(drop=True)
    print(f'Number of rows after dropping: {df.shape[0]}\n')
    
    print('****** Step 2 - dropping columns *****')
    print(f'Number of features before dropping: {df.shape[1]}')
    
    #before dropping, filling NaN values for each columns 
    #that NaN values indicated -1, 0, 9 and 10
    # fill unknown_df
    df[unknown_0]=df[unknown_0].fillna(0)
    df[unknown_9]=df[unknown_9].fillna(9)
    df[unknown_neg1]=df[unknown_neg1].fillna(-1)

    # fill transactions_df
    df[transactions_0]=df[transactions_0].fillna(0)
    df[transactions_10]=df[transactions_10].fillna(10)

    # fill online_transactions_df
    df[online_0]=df[online_0].fillna(0)

    #drop 8 features: 6 features that percentage missing is over 65% 
    #and 2 features that contains unnecessary different items:
    drop_cols= ['ALTER_KIND1', 
                'ALTER_KIND2', 
                'ALTER_KIND3', 
                'ALTER_KIND4', 
                'EXTSEL992',
                'KK_KUNDENTYP', 
                'D19_LETZTER_KAUF_BRANCHE',
                'EINGEFUEGT_AM']
    df.drop(drop_cols, axis=1, inplace=True)
    print(f'Number of features after dropping: {df.shape[1]}\n')
    
    print('****** Step 3 - Remove highly correlated features *****')
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than correlation coefficient
    to_drop = [column for column in upper.columns if any(upper[column] > correlation)]
    print(f'Number of features before removing highly correlated columns: {df.shape[1]}')
    df.drop(to_drop, axis=1, inplace=True)
    print(f'Number of features after removing highly correlated columns: {df.shape[1]} \n')
    
    #encode values using get_dummies
    # CAMEO_DEU_2015, CAMEO_INTL_2015 replaced by -1
    print('****** Step 4 - Encode Categorical Features *****')
    
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].replace('XX', -1)
    df['CAMEO_DEU_2015'] =df['CAMEO_DEU_2015'].replace('XX', -1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].fillna(-1)
    df['CAMEO_INTL_2015']=df['CAMEO_INTL_2015'].astype(int)
    
    # CAMEO_DEUG_2015 replaced by -1
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].replace('X', -1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].fillna(-1)
    df['CAMEO_DEUG_2015']=df['CAMEO_DEUG_2015'].astype(int)
    
    # NaN values OST_WEST_KZ replaced by -1, 
    #followed by encoding 1 and 2 for O and W, respectively
    df['OST_WEST_KZ']=df['OST_WEST_KZ'].fillna(-1)
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({'O':1, 'W':2})
    df['ALTER_HH']=df['ALTER_HH'].fillna(0)
    
    #with  all null data now handled, we should focus on getting
    #objects/categorical variables to numbers via one hot encoding
    print(f'number of features before encoding categorical: { df.shape[1]}')
    df = pd.get_dummies(df, drop_first=True)
    print(f'number of features agter encoding categorical:{ df.shape[1]} \n')
    #get columns header list
    columns = df.columns
    
    
    print('****** Step 5 - Impute values *****')
    # impute nans using mode value
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df = pd.DataFrame(imp.fit_transform(df))
    print(f'check NaN values in dataset: {(df.isnull().sum()>0).sum()}')

    if(method=='StandardScaler'):
        print('****** Step 6 - Transform Dataset *****')
        scale = StandardScaler()
        df=pd.DataFrame(scale.fit_transform(df), columns=columns)
    elif (method == 'MinMaxScaler'):
        print('****** Step 6 - Transform Dataset *****')
        scale = MinMaxScaler()
        df = pd.DataFrame(scale.fit_transform(df), columns=columns)
    #finally set index of dataframe by column LNR, LNR contain unique values for each customers
    df = df.set_index('LNR')
    print(f'Shape after clean data: {df.shape}')
    print('FINISH CLEAN DATASET \n')
    return df
    
    
    
                                                  
                                                  
    
                                                  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


  
    
