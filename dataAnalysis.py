def data_overview(df):
    
    # print list of columns
    print('List of columns in teh given data are: \n', df.columns)
    
    # change the name of columns to lowercase & replace space with '_'
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    print('\n List of columns in teh given data are: \n', df.columns)
    
    # details of the data
    print('\n Shape of the data is: \n',df.shape)
    print('\n Data information is: \n', df.info())
    print('\n Five point summary of the data is: \n', df.describe().T)
    print('\n Number of NA values in the data: \n', df.isna().sum())
    print('\n Number of Null values in the data: \n', df.isnull().sum())
    print('\n Number of duplicated records are : \n', df.duplicated().sum())

    return df

