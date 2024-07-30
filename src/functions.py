import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

# Function to calculate KPIs
def calculate_kpis(df): 
    kpis = {}

    # Completion Rate
    total_visits = df['visit_visitor_id'].nunique()
    completed_visits = df[df['last_step'] == 'confirm']['visit_visitor_id'].nunique()
    kpis['completion_rate'] = completed_visits / total_visits

    # Time Spent on Each Step
    kpis['avg_start_time'] = df['start_time'].mean()
    kpis['avg_step_1_time'] = round(df['step_1'].mean(),2)
    kpis['avg_step_2_time'] = round(df['step_2'].mean(),2)
    kpis['avg_step_3_time'] = round(df['step_3'].mean(),2)

    # Error Rates (new definition)
    errors_1st_step = df['1st_step'].sum()
    errors_2nd_step = df['2nd_step'].sum()
    errors_3rd_step = df['3rd_step'].sum()
    total_errors = errors_1st_step + errors_2nd_step + errors_3rd_step


    kpis['error_rate'] = (total_errors / total_visits)

    return kpis

# Function to create a frequency table and proportion table
def frequency_proportion(df:pd.DataFrame, column :list): 
    print(f'Frequency:{df[column].value_counts()}') # Frequency table for 'column'
    print(f'Proportion: {df[column].value_counts(normalize=True)}') # Calculating the proportion of each unique value in the 'column'


def cross_table(df:pd.DataFrame, column :list):
    # Create a cross-tabulation table of the specified column with the count of occurrences
    my_table = pd.crosstab(index=df[column], columns="count").reset_index()
    # Remove the column name from the table
    my_table.columns.name = None
    # Return the cross-tabulation table
    return my_table

def key_stats(df:pd.DataFrame, column :list): # Changed to str
    for col in column:
        print('Column:', col)
        print(f'Variance: {df[col].var()}') # Access column directly
        print(f'std_dev: {df[col].std()}')
        print(f'min: {df[col].min()}')
        print(f'max: {df[col].max()}')
        print(f'range: {df[col].max() - df[col].min()}')
        print(f'quantiles: {df[col].quantile([0.25, 0.5, 0.75])}')
        print(f'Skewness: {df[col].skew()}')
        print(f'Kurtosis: {df[col].kurt()}')
        print('Median:', df[col].median())
        print('Mode:', df[col].mode())
        print('------------------------------------------------')


def convert_data_types_final(df):
    df['client_id'] = df['client_id'].astype(object)
    df['visit_visitor_id'] = df['visit_visitor_id'].astype(object)
    df['start_time'] = round(df['start_time'].astype(float), 2)
    df['step_1'] = round(df['step_1'].astype(float), 2)
    df['step_2'] = round(df['step_2'].astype(float), 2)
    df['step_3'] = round(df['step_3'].astype(float), 2)
    df['time_completion'] = round(df['time_completion'].astype(float), 2)
    df['navigations_bt_start_last'] = df['navigations_bt_start_last'].fillna(0).astype(int)
    df['completion'] = df['completion'].astype(bool)
    df['start_step'] = df['start_step'].astype(int)
    df['1st_step'] = df['1st_step'].fillna(0).astype(int)
    df['2nd_step'] = df['2nd_step'].fillna(0).astype(int)
    df['3rd_step'] = df['3rd_step'].fillna(0).astype(int)
    df['last_step'] = df['last_step'].astype(object)
    df['total_time_visit'] = round(df['total_time_visit'].astype(float), 2)
    df['variation'] = df['variation'].astype(object)
    df['clnt_tenure_yr'] = df['clnt_tenure_yr'].astype(int)
    df['clnt_tenure_mnth'] = df['clnt_tenure_mnth'].astype(int)
    df['clnt_age'] = df['clnt_age'].astype(int)
    df['gendr'] = df['gendr'].astype(object)
    df['bal'] = round(df['bal'].astype(float), 2)
    df['num_accts'] = df['num_accts'].astype(int)
    df['calls_6_mnth'] = df['calls_6_mnth'].astype(int)
    df['logons_6_mnth'] = df['logons_6_mnth'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df['initial_date'] = pd.to_datetime(df['initial_date'])
    df['final_date'] = pd.to_datetime(df['final_date'])
    return df

def convert_data_types_with_hour(df):
    df['client_id'] = df['client_id'].astype(object)
    df['visit__id'] = df['visit_id'].astype(object)
    df['visitor__id'] = df['visitor_id'].astype(object)
    df['process_step'] = df['process_step'].astype(object)
    df['date_time'] = pd.to_datetime(df.date_time)
    df['Variation'] = df['Variation'].astype(object)
    df['clnt_tenure_yr'] = df['clnt_tenure_yr'].astype(int)
    df['clnt_tenure_mnth'] = df['clnt_tenure_mnth'].astype(int)
    df['clnt_age'] = df['clnt_age'].astype(int)
    df['gendr'] = df['gendr'].astype(object)
    df['num_accts'] = df['num_accts'].astype(int)
    df['calls_6_mnth'] = df['calls_6_mnth'].astype(int)
    df['logons_6_mnth'] = df['logons_6_mnth'].astype(int)
    return df

def convert_data_types(df):
    df['client_id'] = df['client_id'].astype(object)
    df['visit_visitor_id'] = df['visit_visitor_id'].astype(object)
    df['start_time'] = round(df['start_time'].astype(float),2)
    df['step_1'] = round(df['step_1'].astype(float),2)
    df['step_2'] = round(df['step_2'].astype(float),2)
    df['step_3'] = round(df['step_3'].astype(float),2)
    df['date'] = pd.to_datetime(df.date)
    df['1st_step'] = df['1st_step'].fillna(0).astype(int)
    df['2nd_step'] = df['2nd_step'].fillna(0).astype(int)
    df['3rd_step'] = df['3rd_step'].fillna(0).astype(int)
    df['navigations_bt_start_last'] = df['navigations_bt_start_last'].fillna(0).astype(int)
    df['last_step'] = df['last_step'].astype(object)
    df['completion'] = df['completion'].astype(bool)
    df['total_time_visit'] = round(df['total_time_visit'].astype(float),2)
    df['variation'] = df['variation'].astype(object)
    df['clnt_tenure_yr'] = df['clnt_tenure_yr'].astype(int)
    df['clnt_tenure_mnth'] = df['clnt_tenure_mnth'].astype(int)
    df['clnt_age'] = df['clnt_age'].astype(int)
    df['gendr'] = df['gendr'].astype(object)
    df['num_accts'] = df['num_accts'].astype(int)
    df['calls_6_mnth'] = df['calls_6_mnth'].astype(int)
    df['logons_6_mnth'] = df['logons_6_mnth'].astype(int)
    return df

# Helper function to convert and round numerical columns
def convert_round(df, columns, dtype, decimals=2):
    for column in columns:
        df[column] = df[column].astype(dtype)
        if dtype == float:
            df[column] = round(df[column], decimals)
    return df

# Helper function to convert columns to a specific dtype
def convert_dtype(df, columns, dtype):
    for column in columns:
        df[column] = df[column].astype(dtype)
    return df