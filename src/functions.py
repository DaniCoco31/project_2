import pandas as pd
"""
To clean the na values of a df
"""
def clean_data(df_input : pd.DataFrame):
    total_rows = df_input.shape[0]
    threshold = int(total_rows*0.05)
    na_columns = df_input.isna().sum()

    #print('Checking null values per column:')
    for column in df_input:
        na_count = df_input[column].isna().sum()
        #print(column,na_count)
        if na_count > threshold:
            print(f'To many NaN values in column:',column)
        else:
            #print(f'Cleaning column:',column)
            df_input = df_input.dropna(subset=column)
    
    return df_input
    
def print_clean_data(df_input):
    print('Before')
    print(df_input.isna().sum())
    print('------------------------------------------')
    df_input = clean_data(df_input)
    print('------------------------------------------')
    print('After')
    print(df_input.isna().sum())
    return df_input
