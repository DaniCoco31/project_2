
import streamlit as st

# Functions
import pandas as pd
import sys
import numpy as np
sys.path.append('../src')
from functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import *
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



# Function to convert data types for df_test_final
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

# Function to convert data types for df_test and df_control with date and time
def convert_data_types_with_hour(df):
    df['client_id'] = df['client_id'].astype(object)
    df['visit_id'] = df['visit_id'].astype(object)
    df['visitor_id'] = df['visitor_id'].astype(object)
    df['process_step'] = df['process_step'].astype(object)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['Variation'] = df['Variation'].astype(object)
    df['clnt_tenure_yr'] = df['clnt_tenure_yr'].astype(int)
    df['clnt_tenure_mnth'] = df['clnt_tenure_mnth'].astype(int)
    df['clnt_age'] = df['clnt_age'].astype(int)
    df['gendr'] = df['gendr'].astype(object)
    df['num_accts'] = df['num_accts'].astype(int)
    df['calls_6_mnth'] = df['calls_6_mnth'].astype(int)
    df['logons_6_mnth'] = df['logons_6_mnth'].astype(int)
    return df

# Function to convert data types for generic data frame
def convert_data_types(df):
    df['client_id'] = df['client_id'].astype(object)
    df['visit_visitor_id'] = df['visit_visitor_id'].astype(object)
    df['start_time'] = round(df['start_time'].astype(float), 2)
    df['step_1'] = round(df['step_1'].astype(float), 2)
    df['step_2'] = round(df['step_2'].astype(float), 2)
    df['step_3'] = round(df['step_3'].astype(float), 2)
    df['date'] = pd.to_datetime(df['date'])
    df['1st_step'] = df['1st_step'].fillna(0).astype(int)
    df['2nd_step'] = df['2nd_step'].fillna(0).astype(int)
    df['3rd_step'] = df['3rd_step'].fillna(0).astype(int)
    df['navigations_bt_start_last'] = df['navigations_bt_start_last'].fillna(0).astype(int)
    df['last_step'] = df['last_step'].astype(object)
    df['completion'] = df['completion'].astype(bool)
    df['total_time_visit'] = round(df['total_time_visit'].astype(float), 2)
    df['variation'] = df['variation'].astype(object)
    df['clnt_tenure_yr'] = df['clnt_tenure_yr'].astype(int)
    df['clnt_tenure_mnth'] = df['clnt_tenure_mnth'].astype(int)
    df['clnt_age'] = df['clnt_age'].astype(int)
    df['gendr'] = df['gendr'].astype(object)
    df['num_accts'] = df['num_accts'].astype(int)
    df['calls_6_mnth'] = df['calls_6_mnth'].astype(int)
    df['logons_6_mnth'] = df['logons_6_mnth'].astype(int)
    return df


def convert_data_types_combined(df):
    column_type_mappings = {
        'client_id': 'object',
        'visit_visitor_id': 'object',
        'visit_id': 'object',
        'visitor_id': 'object',
        'process_step': 'object',
        'start_time': 'float',
        'step_1': 'float',
        'step_2': 'float',
        'step_3': 'float',
        'time_completion': 'float',
        'navigations_bt_start_last': 'int',
        'completion': 'int',
        'start_step': 'int',
        '1st_step': 'int',
        '2nd_step': 'int',
        '3rd_step': 'int',
        'last_step': 'object',
        'total_time_visit': 'float',
        'variation': 'object',
        'clnt_tenure_yr': 'int',
        'clnt_tenure_mnth': 'int',
        'clnt_age': 'int',
        'gendr': 'object',
        'bal': 'float',
        'num_accts': 'int',
        'calls_6_mnth': 'int',
        'logons_6_mnth': 'int',
        'date': 'datetime64[ns]',
        'initial_date': 'datetime64[ns]',
        'final_date': 'datetime64[ns]',
        'date_time': 'datetime64[ns]',
    }

    missing_columns = []

    for column, dtype in column_type_mappings.items():
        if column in df.columns:
            if dtype == 'float':
                df[column] = round(df[column].astype(dtype), 2)
            elif dtype == 'int':
                df[column] = df[column].fillna(0).astype(dtype)
            elif dtype == 'datetime64[ns]':
                df[column] = pd.to_datetime(df[column])
            else:
                df[column] = df[column].astype(dtype)
    return df



def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Calculate correlation coefficient and p-value
def calculate_correlation(df, var1, var2):
    correlation_coef, p_value = pearsonr(df[var1], df[var2])
    return correlation_coef, p_value

def plot_distribution(df, variable, title):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[variable], kde=False, discrete=True, bins=range(df[variable].max() + 1))
    plt.title(title)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.show()

    # Function to calculate error rate and completion rate
def calculate_rates(df):
    # Sort the dataframe by 'visit_id' and 'date_time'
    df = df.sort_values(by=['visit_id', 'date_time'])
    
    # Calculate the time difference between steps
    df['time_diff'] = df.groupby('visit_id')['date_time'].diff().dt.total_seconds()
    
    # Identify completions (where process step is 'confirm')
    df['completion'] = df['process_step'] == 'confirm'
    
    # Convert 'process_step' to a category type and then to codes for numerical comparison
    df['process_step_code'] = df['process_step'].astype('category').cat.codes
    
    # Identify errors (going back to the previous step in less than 30 seconds)
    df['error'] = (df['time_diff'] < 30) & (df['process_step_code'].diff() < 0)
    
    # Calculate the daily error rate
    error_rate = df.groupby(df['date_time'].dt.date)['error'].mean()
    
    # Calculate the daily completion rate
    completion_rate = df.groupby(df['date_time'].dt.date)['completion'].mean()
    
    return error_rate, completion_rate


def main():
    st.title("Streamlit Application")

    st.header("Univariate Analysis")
    import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler

import pandas as pd
import sys
sys.path.append('../src')
from functions import *
import seaborn as sns
# raw data
df_clients_profile = pd.read_csv('../Data/Raw/df_final_demo.txt')
df_web_data_1 = pd.read_csv('../Data/Raw/df_final_web_data_pt_1.txt')
df_web_data_2 = pd.read_csv('../Data/Raw/df_final_web_data_pt_2.txt')
df_experiment_clients = pd.read_csv('../Data/Raw/df_final_experiment_clients.txt')
# processed data
df_test = pd.read_csv('../Data/Cleaned_Data/df_test.csv')
df_test_final = pd.read_csv('../Data/Cleaned_Data/df_test_final.csv')
df_control = pd.read_csv('../Data/Cleaned_Data/df_control.csv')
df_control_final = pd.read_csv('../Data/Cleaned_Data/df_control_final.csv')
df_final = pd.read_csv('../Data/Cleaned_Data/df_final.csv')
df_combined = pd.read_csv('../Data/Cleaned_Data/df_combined.csv')
pd.set_option('display.max_columns', None)
df_web_data = pd.concat([df_web_data_1, df_web_data_2], ignore_index= True)

df_test = convert_data_types_combined(df_test)
df_test_final = convert_data_types_combined(df_test_final)
df_control = convert_data_types_combined(df_control)
df_control_final = convert_data_types_combined(df_control_final)
df_final = convert_data_types_combined(df_final)
df_combined = convert_data_types_combined(df_combined)

df_test_final = pd.read_csv('../Data/Cleaned_Data/df_test_final.csv')
df_control_final = pd.read_csv('../Data/Cleaned_Data/df_control_final.csv')
pd.set_option('display.max_columns', None)

df_control_final.dtypes

df_control_final

# Extracting column names with numerical data types from the dataframe
df_control_final.select_dtypes("object").columns


test_categorical_columns = ['last_step', 'gendr',]

# Extracting column names with numerical data types from the dataframe
df_control_final.select_dtypes("object").nunique().sort_values(ascending=False)

frequency_proportion(df_control_final, 'last_step')

frequency_proportion(df_control_final, 'gendr')

tab_control_last_step = cross_table(df_control_final, 'last_step')
tab_control_last_step

df_control_final['last_step'].value_counts()

tab_control_last_step = df_control_final['last_step'].value_counts().reset_index()
tab_control_last_step.columns = ['last_step', 'count']
tab_control_last_step

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Assuming 'tab_control_last_step' is your DataFrame
colors = cm.viridis(range(len(tab_control_last_step)))

tab_control_last_step.plot(x='last_step', y='count', kind='bar', color=colors)
plt.xlabel('Last Step')
plt.ylabel('Count')
plt.title('Last Step Distribution in Control Group')
plt.show()


tab_control_gender = cross_table(df_control_final, 'gendr')
tab_control_gender

# Calculating the proportions for each value in 'tab_test_last_step' and rounding the results to two decimal places
(tab_control_gender['count'] / tab_control_gender['count'].sum()).round(2)

tab_control_gender.plot.pie(y='count', labels=tab_control_gender['gendr'], autopct='%1.1f%%')
plt.title('Gender Distribution in Control Group')
plt.axis('equal')
plt.show()

# Extracting column names with numerical data types from the dataframe
control_numerical_columns = df_control_final.select_dtypes("number").columns
print(control_numerical_columns)
print(df_control_final.dtypes)

control_numerical_columns = pd.DataFrame

control_numerical_columns = ['start_time', 'step_1', 'step_2', 'step_3', 'time_completion',
       'navigations_bt_start_last', 'start_step', '1st_step', '2nd_step',
       '3rd_step', 'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age',
       'num_accts', 'bal', 'calls_6_mnth', 'logons_6_mnth',
       'total_time_visit']
print(control_numerical_columns)

# Extracting column names with numerical data types from the dataframe
df_control_final.select_dtypes("number").nunique().sort_values(ascending=False)

df_control_final.describe()

# Filtering the numerical columns for analysis
df_numerical_control = pd.DataFrame(df_control_final[control_numerical_columns])

# Plotting histograms for the numerical columns before removing outliers
df_numerical_control.hist(figsize=(15, 20), bins=60, xlabelsize=1, ylabelsize=10);

# Applying IQR method to each specified column
for column in control_numerical_columns:
    df_control_final = remove_outliers_iqr(df_control_final, column)
    df_numerical_control = pd.DataFrame(df_control_final[control_numerical_columns])

# Plotting histograms for the numerical columns after removing outliers
df_numerical_control.hist(figsize=(15, 20), bins=60, xlabelsize=10, ylabelsize=10);

# List of columns to apply log transformation
log_transform_columns = [
    'start_time', 'step_1', 'step_2', 'step_3', 'navigations_bt_start_last',
    'start_step', '1st_step', '2nd_step', '3rd_step', 'bal',
    'calls_6_mnth', 'logons_6_mnth', 'total_time_visit'
]
# Applying log transformation
for column in log_transform_columns:
    df_control_final[column] = np.log1p(df_control_final[column])

# Reapplying IQR method to each specified column after log transformation
control_numerical_columns = log_transform_columns + [
    'time_completion', 'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age', 'num_accts'
]
for column in control_numerical_columns:
    df_control_final = remove_outliers_iqr(df_control_final, column)
# Filtering the numerical columns for analysis
df_numerical_control = pd.DataFrame(df_control_final[control_numerical_columns]) 

# Plotting histograms for the numerical columns after removing outliers
df_numerical_control.hist(figsize=(15, 20), bins=60, xlabelsize=10, ylabelsize=10);


df_control_final.to_csv('../Data/Cleaned_Data/df_control_final.csv', index=False)  

df_test_final = convert_data_types_final(df_test_final)


# Extracting column names with numerical data types from the dataframe
df_test_final.select_dtypes("object").columns


test_categorical_columns = ['last_step', 'gendr']

# Extracting column names with numerical data types from the dataframe
df_test_final.select_dtypes("object").nunique().sort_values(ascending=False)

frequency_proportion(df_test_final, 'last_step')

frequency_proportion(df_test_final, 'gendr')

cross_table(df_test_final, 'start_time')

frequency_proportion(df_test_final, 'start_time')

df_test_final['last_step'].value_counts()

tab_test_last_step = df_test_final['last_step'].value_counts().reset_index()
tab_test_last_step.columns = ['last_step', 'count']
tab_test_last_step

# Calculating the proportions for each value in 'tab_test_last_step' and rounding the results to two decimal places
(tab_test_last_step['count'] / tab_test_last_step['count'].sum())

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Assuming 'tab_test_last_step' is your DataFrame
colors = cm.viridis(range(len(tab_test_last_step)))

tab_test_last_step.plot(x='last_step', y='count', kind='bar', color=colors)
plt.xlabel('Last Step')
plt.ylabel('Count')
plt.title('Last Step Distribution in Test Group')
plt.show()


tab_test_gender = cross_table(df_test_final, 'gendr')
tab_test_gender


# Calculating the proportions for each value in 'tab_test_last_step' and rounding the results to two decimal places
(tab_test_gender['count'] / tab_test_gender['count'].sum()).round(2)

tab_test_gender.plot.pie(y='count', labels=tab_test_gender['gendr'], autopct='%1.1f%%')
plt.title('Gender Distribution in Test Group')
plt.axis('equal')
plt.show()

# Extracting column names with numerical data types from the dataframe
df_test_final.select_dtypes("number").columns

df_test_final.dtypes


test_numerical_columns = ['start_time', 'step_1', 'step_2', 'step_3', 'time_completion',
       'navigations_bt_start_last', 'start_step', '1st_step', '2nd_step',
       '3rd_step', 'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age',
       'num_accts', 'bal', 'calls_6_mnth', 'logons_6_mnth',
       'total_time_visit']

test_numerical_columns

# Extracting column names with numerical data types from the dataframe
df_test_final.select_dtypes("number").nunique().sort_values(ascending=False)


df_test_final.describe()

df_numerical_test = pd.DataFrame(df_test_final[test_numerical_columns]) 

df_numerical_test.hist(figsize=(15, 20), bins=60, xlabelsize=1, ylabelsize=10);

for column in test_numerical_columns:
    df_test_final = remove_outliers_iqr(df_test_final, column)

    df_numerical_test = pd.DataFrame(df_test_final[test_numerical_columns]) 


df_numerical_test.hist(figsize=(15, 20), bins=60, xlabelsize=10, ylabelsize=10);

# List of columns to apply log transformation
log_transform_columns = ['step_1', 'step_2', 'step_3', 'bal', 'total_time_visit']

# Apply log transformation
for column in log_transform_columns:
    df_test_final[column] = np.log1p(df_test_final[column])


df_test_final = df_test_final.to_csv('../Data/Cleaned_Data/df_test_final.csv', index=False)



    st.header("Multivariate Analysis")
    import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


import sys
sys.path.append('../src')
from functions import *
# raw data
df_clients_profile = pd.read_csv('../Data/Raw/df_final_demo.txt')
df_web_data_1 = pd.read_csv('../Data/Raw/df_final_web_data_pt_1.txt')
df_web_data_2 = pd.read_csv('../Data/Raw/df_final_web_data_pt_2.txt')
df_experiment_clients = pd.read_csv('../Data/Raw/df_final_experiment_clients.txt')
# processed data
df_test = pd.read_csv('../Data/Cleaned_Data/df_test.csv')
df_test_final = pd.read_csv('../Data/Cleaned_Data/df_test_final.csv')
df_control = pd.read_csv('../Data/Cleaned_Data/df_control.csv')
df_control_final = pd.read_csv('../Data/Cleaned_Data/df_control_final.csv')
df_final = pd.read_csv('../Data/Cleaned_Data/df_final.csv')
df_combined = pd.read_csv('../Data/Cleaned_Data/df_combined.csv')
pd.set_option('display.max_columns', None)
df_web_data = pd.concat([df_web_data_1, df_web_data_2], ignore_index= True)

df_test = convert_data_types_combined(df_test)
df_test_final = convert_data_types_combined(df_test_final)
df_control = convert_data_types_combined(df_control)
df_control_final = convert_data_types_combined(df_control_final)
df_final = convert_data_types_combined(df_final)
df_combined = convert_data_types_combined(df_combined)

correlation_matrix = df_final.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(20, 18))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 10})  # Adjust font size
plt.title('Correlation Matrix', fontsize=20)  # Adjust title font size
plt.xticks(rotation=45, ha='right', fontsize=12)  # Adjust x-axis tick labels
plt.yticks(rotation=0, fontsize=12)  # Adjust y-axis tick labels
plt.show()


test_age = df_final[df_final['variation'] == 'Test']['clnt_age']
control_age = df_final[df_final['variation'] == 'Control']['clnt_age']
t_stat_age, p_value_age = ttest_ind(test_age, control_age)
print(f'Age t-statistic: {t_stat_age}, p-value: {p_value_age}')

test_tenure = df_final[df_final['variation'] == 'Test']['clnt_tenure_yr']
control_tenure = df_final[df_final['variation'] == 'Control']['clnt_tenure_yr']
t_stat_tenure, p_value_tenure = ttest_ind(test_tenure, control_tenure)
print(f'Tenure t-statistic: {t_stat_tenure}, p-value: {p_value_tenure}')

contingency_table_gender_process = pd.crosstab(df_final['gendr'], df_final['variation'])
chi2_stat_gender, p_value_gender, _, _ = chi2_contingency(contingency_table_gender_process)
print(f'Chi-square statistic for gender and process: {chi2_stat_gender}, p-value: {p_value_gender}')

# Compute the contingency table
contingency_table = pd.crosstab(df_final['variation'], df_final['completion'])

# Perform the Chi-square test
chi2_stat_completion, p_value_completion, _, _ = chi2_contingency(contingency_table)

# Calculate Cramér's V
n = contingency_table.sum().sum()  # Total number of observations
min_dimension = min(contingency_table.shape) - 1  # Minimum dimension - 1

cramers_v = np.sqrt((chi2_stat_completion / n) / min_dimension)

# Print the results
print(f'Chi-square statistic for completion rates: {chi2_stat_completion}, p-value: {p_value_completion}')
print(f"Cramér's V for the association between variation and completion: {cramers_v}")

# Calculate Completion Rates
completion_rates = df_final.groupby('variation')['completion'].mean()
completion_rate_test = completion_rates.get('Test', 0)
completion_rate_control = completion_rates.get('Control', 0)
percentage_increase = ((completion_rate_test - completion_rate_control) / completion_rate_control) * 100 if completion_rate_control != 0 else float('inf')

print(f'Completion rate for Test group: {completion_rate_test}')
print(f'Completion rate for Control group: {completion_rate_control}')
print(f'Percentage increase in completion rate: {percentage_increase}%')

import matplotlib.pyplot as plt
import seaborn as sns

# Data
completion_rate_test = 0.5857757461683248
completion_rate_control = 0.49886769039863504
percentage_increase = 17.421063228256642

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Create bar plot
bars = ax.bar(['Test Group', 'Control Group'], [completion_rate_test, completion_rate_control], color=colors)

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=12)

# Add title and labels
ax.set_title('Completion Rates for Test and Control Groups', fontsize=16)
ax.set_ylabel('Completion Rate', fontsize=14)
ax.set_xlabel('Group', fontsize=14)
ax.set_ylim(0, 1)

plt.show()


# Perform the Two-Proportion Z-Test:
from statsmodels.stats.proportion import proportions_ztest

# Define number of successes (completed visits) and number of trials (total visits) for both groups
num_success_test = df_final[df_final['variation'] == 'Test']['completion'].sum()
num_trials_test = len(df_final[df_final['variation'] == 'Test'])
num_success_control = df_final[df_final['variation'] == 'Control']['completion'].sum()
num_trials_control = len(df_final[df_final['variation'] == 'Control'])


# Perform the two-proportion z-test
successes = [num_success_test, num_success_control]
trials = [num_trials_test, num_trials_control]
z_stat, p_value = proportions_ztest(successes, trials)

z_stat, p_value = proportions_ztest([num_success_test, num_success_control], [num_trials_test, num_trials_control])
print(f'Two-proportion z-test statistic: {z_stat}, p-value: {p_value}')

# Define the threshold for percentage increase
threshold = 5.0
if percentage_increase >= threshold:
    print('The observed increase in completion rate meets or exceeds the 5% threshold.')
else:
    print('The observed increase in completion rate does not meet the 5% threshold.')

if p_value < 0.05:
    print('The increase in completion rate is statistically significant.')
else:
    print('The increase in completion rate is not statistically significant.')


import matplotlib.pyplot as plt
import seaborn as sns

# Data
completion_rate_test = 0.5857757461683248
completion_rate_control = 0.49886769039863504
percentage_increase = 17.421063228256642

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Create bar plot
bars = ax.bar(['Test Group', 'Control Group'], [completion_rate_test, completion_rate_control], color=colors)

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=12)

# Add title and labels
ax.set_title('Completion Rates for Test and Control Groups', fontsize=16)
ax.set_ylabel('Completion Rate', fontsize=14)
ax.set_xlabel('Group', fontsize=14)
ax.set_ylim(0, 1)

plt.show()


# Calculate the mean of total_time_visit for 'Test' variation
mean_total_time_test = df_final.loc[df_final['variation'] == 'Test', 'total_time_visit'].mean()

# Calculate the mean of total_time_visit for 'Control' variation
mean_total_time_control = df_final.loc[df_final['variation'] == 'Control', 'total_time_visit'].mean()

print(f"Mean total time visit for 'Test' variation: {mean_total_time_test}")
print(f"Mean total time visit for 'Control' variation: {mean_total_time_control}")


# Extract total time visit for each group
test_time = df_final[df_final['variation'] == 'Test']['total_time_visit']
control_time = df_final[df_final['variation'] == 'Control']['total_time_visit']


# Perform t-test
t_stat_time, p_value_time = ttest_ind(test_time, control_time)
print(f'Total time visit t-statistic: {t_stat_time}, p-value: {p_value_time}')

import matplotlib.pyplot as plt
import seaborn as sns

# Data
mean_total_time_test = 5.258645872546383
mean_total_time_control = 4.6747097874980605

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Create bar plot
bars = ax.bar(['Test Group', 'Control Group'], [mean_total_time_test, mean_total_time_control], color=colors)

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f} min', ha='center', va='bottom', fontsize=12)

# Add title and labels
ax.set_title('Average Total Time Spent for Test and Control Groups', fontsize=16)
ax.set_ylabel('Total Time Spent (minutes)', fontsize=14)
ax.set_xlabel('Group', fontsize=14)
ax.set_ylim(0, max(mean_total_time_test, mean_total_time_control) + 1)

plt.show()


# Extract number of accounts for each group
test_accounts = df_final[df_final['variation'] == 'Test']['num_accts']
control_accounts = df_final[df_final['variation'] == 'Control']['num_accts']

# Perform t-test
t_stat_accounts, p_value_accounts = ttest_ind(test_accounts, control_accounts)
print(f'Number of accounts t-statistic: {t_stat_accounts}, p-value: {p_value_accounts}')

# Calculate the mean of total_time_visit for 'Test' variation
mean_logons_test = df_final.loc[df_final['variation'] == 'Test', 'logons_6_mnth'].mean()

# Calculate the mean of total_time_visit for 'Control' variation
mean_logons_control = df_final.loc[df_final['variation'] == 'Control', 'logons_6_mnth'].mean()

print(f"Mean logons for 'Test' variation: {mean_total_time_test}")
print(f"Mean logons for 'Control' variation: {mean_total_time_control}")

test_logons = df_final[df_final['variation'] == 'Test']['logons_6_mnth']
control_logons = df_final[df_final['variation'] == 'Control']['logons_6_mnth']
t_stat_balances, p_value_balances = ttest_ind(test_logons, control_logons)
print(f'Balance t-statistic: {t_stat_balances}, p-value: {p_value_balances}')

test_calls = df_final[df_final['variation'] == 'Test']['calls_6_mnth']
control_calls = df_final[df_final['variation'] == 'Control']['calls_6_mnth']
t_stat_calls, p_value_calls = ttest_ind(test_calls, control_calls)
print(f'Calls in last 6 months t-statistic: {t_stat_calls}, p-value: {p_value_calls}')


# Extract recent call activity for each group
test_navigations = df_final[df_final['variation'] == 'Test']['navigations_bt_start_last']
control_navigations = df_final[df_final['variation'] == 'Control']['navigations_bt_start_last']

# Perform t-test
t_stat_navigations, p_value_navigations = ttest_ind(test_navigations, control_navigations)
print(f'Navigations between start and last t-statistic: {t_stat_navigations}, p-value: {p_value_navigations}')

test_navigations = df_final[df_final['variation'] == 'Test']['navigations_bt_start_last']
control_navigations = df_final[df_final['variation'] == 'Control']['navigations_bt_start_last']

t_stat_navigations, p_value_navigations = ttest_ind(test_navigations, control_navigations)
print(f'Navigations between start and last t-statistic: {t_stat_navigations}, p-value: {p_value_navigations}')

import matplotlib.pyplot as plt
import seaborn as sns

# Data
mean_logons_test = 5.258645872546383
mean_logons_control = 4.6747097874980605

mean_calls_test = 4.6747097874980605  # Example values, replace with actual
mean_calls_control = 3.318472950676876

mean_navigations_test = 4.8  # Example values, replace with actual
mean_navigations_control = 4.9

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Plot side by side
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot logons
bars = ax[0].bar(['Test Group', 'Control Group'], [mean_logons_test, mean_logons_control], color=colors)
for bar in bars:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
ax[0].set_title('Average Logons in Last 6 Months', fontsize=16)
ax[0].set_ylabel('Average Logons', fontsize=14)
ax[0].set_ylim(0, max(mean_logons_test, mean_logons_control) + 1)

# Plot calls
bars = ax[1].bar(['Test Group', 'Control Group'], [mean_calls_test, mean_calls_control], color=colors)
for bar in bars:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
ax[1].set_title('Average Calls in Last 6 Months', fontsize=16)
ax[1].set_ylabel('Average Calls', fontsize=14)
ax[1].set_ylim(0, max(mean_calls_test, mean_calls_control) + 1)

# Plot navigations
bars = ax[2].bar(['Test Group', 'Control Group'], [mean_navigations_test, mean_navigations_control], color=colors)
for bar in bars:
    yval = bar.get_height()
    ax[2].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
ax[2].set_title('Average Navigations Between Start and Last Steps', fontsize=16)
ax[2].set_ylabel('Average Navigations', fontsize=14)
ax[2].set_ylim(0, max(mean_navigations_test, mean_navigations_control) + 1)

# Add common labels
fig.text(0.5, 0.04, 'Group', ha='center', fontsize=14)
plt.tight_layout()
plt.show()


import pandas as pd
from scipy.stats import chi2_contingency

# Example DataFrame creation (replace with actual data loading code)
# df_combined = pd.read_csv('your_file.csv')

# Convert date_time to pandas datetime format
df_combined['date_time'] = pd.to_datetime(df_combined['date_time'])

# Sort by client_id and date_time
df_combined = df_combined.sort_values(by=['client_id', 'date_time'])

# Calculate the time difference to the previous step
df_combined['time_to_previous_step'] = df_combined.groupby('client_id')['date_time'].diff().dt.total_seconds()

# Identify errors: going back to any previous step in less than 30 seconds
df_combined['is_error'] = df_combined['time_to_previous_step'] < 30

# Count errors and non-errors
errors_test = df_combined[(df_combined['Variation'] == 'Test') & (df_combined['is_error'])].shape[0]
non_errors_test = df_combined[(df_combined['Variation'] == 'Test') & (~df_combined['is_error'])].shape[0]

errors_control = df_combined[(df_combined['Variation'] == 'Control') & (df_combined['is_error'])].shape[0]
non_errors_control = df_combined[(df_combined['Variation'] == 'Control') & (~df_combined['is_error'])].shape[0]

# Calculate total observations for each group
total_test = errors_test + non_errors_test
total_control = errors_control + non_errors_control

# Compute error rates
error_rate_test = errors_test / total_test if total_test > 0 else 0
error_rate_control = errors_control / total_control if total_control > 0 else 0

# Print error rates
print(f'Error rate for Test group: {error_rate_test:.4f}')
print(f'Error rate for Control group: {error_rate_control:.4f}')

# Create a contingency table
contingency_table = [
    [errors_test, non_errors_test],  # Test group
    [errors_control, non_errors_control]  # Control group
]

# Perform the Chi-Square test
chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

print(f'Chi-square statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}')

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in error rates between the Test and Control groups.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in error rates between the Test and Control groups.")

import matplotlib.pyplot as plt
import seaborn as sns

# Data
error_rate_test = 0.0362
error_rate_control = 0.0356

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create bar plot
bars = ax.bar(['Test Group', 'Control Group'], [error_rate_test, error_rate_control], color=colors)

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=12)

# Add title and labels
ax.set_title('Error Rates for Test and Control Groups', fontsize=16)
ax.set_ylabel('Error Rate', fontsize=14)
ax.set_xlabel('Group', fontsize=14)
ax.set_ylim(0, max(error_rate_test, error_rate_control) + 0.01)

plt.show()


df_test_final['navigations_bt_start_last'].value_counts()

# Create a new column categorizing the total number of navigations
df_final['total_navigations_category'] = pd.cut(df_final['navigations_bt_start_last'], 
                                                bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf], 
                                                labels=['0-1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51-100', '100+'])

# Plot the frequency of total navigations for completion
plt.figure(figsize=(12, 8))
sns.countplot(data=df_final, x='total_navigations_category', hue='completion', palette='viridis')
plt.title('Frequency of Total Navigations for Completion')
plt.xlabel('Total Navigations')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Completion', loc='upper right')
plt.show()

# We will create a frequency plot for total navigations based on completion status

# Create a new column categorizing the total number of navigations
df_final['total_navigations_category'] = pd.cut(df_final['navigations_bt_start_last'], 
                                                bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf], 
                                                labels=['0-1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51-100', '100+'])

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# Plot for the Test variation
sns.countplot(data=df_final[df_final['variation'] == 'Test'], 
              x='total_navigations_category', 
              palette='viridis', 
              ax=axes[0])
axes[0].set_title('Test Variation - Frequency of Total Navigations')
axes[0].set_xlabel('Total Navigations')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Plot for the Control variation
sns.countplot(data=df_final[df_final['variation'] == 'Control'], 
              x='total_navigations_category', 
              palette='viridis', 
              ax=axes[1])
axes[1].set_title('Control Variation - Frequency of Total Navigations')
axes[1].set_xlabel('Total Navigations')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()


# We will create a frequency plot for total navigations based on completion status

# Create a new column categorizing the total number of navigations
df_final['total_navigations_category'] = pd.cut(df_final['navigations_bt_start_last'], 
                                                bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf], 
                                                labels=['0-1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51-100', '100+'])

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# Plot for the Test variation
sns.countplot(data=df_final[df_final['variation'] == 'Test'], 
              x='total_navigations_category', 
              hue='completion', 
              palette='viridis', 
              ax=axes[0])
axes[0].set_title('Test Variation - Frequency of Total Navigations for Completion')
axes[0].set_xlabel('Total Navigations')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Plot for the Control variation
sns.countplot(data=df_final[df_final['variation'] == 'Control'], 
              x='total_navigations_category', 
              hue='completion', 
              palette='viridis', 
              ax=axes[1])
axes[1].set_title('Control Variation - Frequency of Total Navigations for Completion')
axes[1].set_xlabel('Total Navigations')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()



import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Assuming error_rate_test, completion_rate_test, error_rate_control, and completion_rate_control are pandas Series or similar structures
# Example data for illustration (replace with actual data)
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
error_rate_test = pd.Series(np.random.rand(10) / 10, index=dates)
completion_rate_test = pd.Series(np.random.rand(10) / 10, index=dates)
error_rate_control = pd.Series(np.random.rand(10) / 10, index=dates)
completion_rate_control = pd.Series(np.random.rand(10) / 10, index=dates)

# Create a scatter plot with lines for error rate vs. completion rate
fig = go.Figure()

# Line+scatter for test data error rate
fig.add_trace(go.Scatter(
    x=error_rate_test.index,
    y=error_rate_test.values,
    mode='lines+markers',
    name='Test Data Error Rate',
    line=dict(shape='linear', color='rgb(68, 1, 84)')  # Color similar to viridis palette
))

# Line+scatter for test data completion rate
fig.add_trace(go.Scatter(
    x=completion_rate_test.index,
    y=completion_rate_test.values,
    mode='lines+markers',
    name='Test Data Completion Rate',
    line=dict(shape='linear', color='rgb(49, 104, 142)')  # Color similar to viridis palette
))

# Line+scatter for control data error rate
fig.add_trace(go.Scatter(
    x=error_rate_control.index,
    y=error_rate_control.values,
    mode='lines+markers',
    name='Control Data Error Rate',
    line=dict(shape='linear', color='rgb(33, 145, 140)')  # Color similar to viridis palette
))

# Line+scatter for control data completion rate
fig.add_trace(go.Scatter(
    x=completion_rate_control.index,
    y=completion_rate_control.values,
    mode='lines+markers',
    name='Control Data Completion Rate',
    line=dict(shape='linear', color='rgb(253, 231, 37)')  # Color similar to viridis palette
))

# Update layout
fig.update_layout(
    title='Completion Rate vs. Error Rate (Test and Control Data)',
    xaxis_title='Date',
    yaxis_title='Rate',
    yaxis_tickformat=',.0%',
    legend_title_text='Rate Type'
)

# Show the plot
fig.show()



# Assuming df_final is already loaded and processed

# Filter the dataframe for navigations <= 30 and completion = 1
filtered_df = df_final[(df_final['navigations_bt_start_last'] <= 30) & (df_final['completion'] == 1)]

# Calculate the frequency of 'navigations_bt_start_last' for each variation
freq = filtered_df.groupby('variation')['navigations_bt_start_last'].value_counts().unstack().fillna(0)

# Normalize the frequencies
norm_freq = freq.div(freq.sum(axis=1), axis=0)

# Reset index for plotting
norm_freq = norm_freq.reset_index()

# Melt DataFrame for easier plotting
norm_freq_melted = norm_freq.melt(id_vars='variation', var_name='Navigations', value_name='Normalized Frequency')

# Define viridis-like colors for the variations
colors = {
    'Test': 'rgb(68, 1, 84)',
    'Control': 'rgb(49, 104, 142)'
}

# Create the figure
fig = go.Figure()

# Add traces for each variation
for variation in norm_freq_melted['variation'].unique():
    df_variation = norm_freq_melted[norm_freq_melted['variation'] == variation]
    fig.add_trace(go.Scatter(
        x=df_variation['Navigations'],
        y=df_variation['Normalized Frequency'],
        mode='lines+markers',
        name=variation,
        line=dict(color=colors[variation])
    ))

# Update layout
fig.update_layout(
    title='Total Navigations done between start and last page by users who completed the process',
    xaxis_title='Navigations',
    yaxis_title='Normalized Frequency',
    legend_title='Variation',
    template='plotly_white'
)

# Show plot
fig.show()


import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure date column is in datetime format
df_final['date'] = pd.to_datetime(df_final['date'])

# Add a 'week' column to the DataFrame
df_final['week'] = df_final['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Calculate weekly completion rate for Test and Control variations
weekly_completion = df_final.groupby(['week', 'variation'])['completion'].mean().unstack()

# Define viridis-like colors for the variations
colors = {
    'Test': 'rgb(68, 1, 84)',       # Purple
    'Control': 'rgb(253, 231, 37)'  # Yellow
}

# Create a line+scatter plot
fig = go.Figure()

# Line+scatter for test data
fig.add_trace(go.Scatter(
    x=weekly_completion.index,
    y=weekly_completion['Test'],
    mode='lines+markers',
    name='Test Data',
    line=dict(shape='linear', color=colors['Test'])
))

# Line+scatter for control data
fig.add_trace(go.Scatter(
    x=weekly_completion.index,
    y=weekly_completion['Control'],
    mode='lines+markers',
    name='Control Data',
    line=dict(shape='linear', color=colors['Control'])
))

# Update layout
fig.update_layout(
    title='Weekly Completion Rate for Test and Control Data',
    xaxis_title='Week',
    yaxis_title='Completion Rate',
    yaxis_tickformat=',.0%',
    legend_title_text='Data Source',
    template='plotly_white'
)

# Show the plot
fig.show()


# Convert 'date' columns to datetime
df_test_final['date'] = pd.to_datetime(df_test_final['date'])
df_control_final['date'] = pd.to_datetime(df_control_final['date'])

# Filter data where total_navigations is 5
filtered_test_df = df_test_final[df_test_final['navigations_bt_start_last'] == 5]
filtered_control_df = df_control_final[df_control_final['navigations_bt_start_last'] == 5]

# Count occurrences of total_navigations = 5 by date for test data
counts_test = filtered_test_df.groupby(filtered_test_df['date'].dt.date).size()

# Count occurrences of total_navigations = 5 by date for control data
counts_control = filtered_control_df.groupby(filtered_control_df['date'].dt.date).size()

# Define viridis-like colors for the variations
colors = {
    'Test': 'rgb(68, 1, 84)',       # Purple
    'Control': 'rgb(253, 231, 37)'  # Yellow
}

# Create a line+scatter plot
fig = go.Figure()

# Line+scatter for test data
fig.add_trace(go.Scatter(
    x=counts_test.index,
    y=counts_test.values,
    mode='lines+markers',
    name='Test Data',
    line=dict(shape='linear', color=colors['Test'])
))

# Line+scatter for control data
fig.add_trace(go.Scatter(
    x=counts_control.index,
    y=counts_control.values,
    mode='lines+markers',
    name='Control Data',
    line=dict(shape='linear', color=colors['Control'])
))

# Update layout
fig.update_layout(
    title='Frequency of Total Steps Taken = 5 by Date',
    xaxis_title='Date',
    yaxis_title='Frequency of Total Steps Taken = 5',
    legend_title_text='Data Source',
    template='plotly_white'
)

# Show the plot
fig.show()



import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Assuming df_test and df_control are already loaded and processed
df_test['date_time'] = pd.to_datetime(df_test['date_time'])
df_control['date_time'] = pd.to_datetime(df_control['date_time'])

# Calculate rates for test data
error_rate_test, completion_rate_test = calculate_rates(df_test)

# Calculate rates for control data
error_rate_control, completion_rate_control = calculate_rates(df_control)

# Create a figure with two vertical subplots
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=("Error Rate", "Completion Rate"),
                    shared_xaxes=True,
                    vertical_spacing=0.1)

# Define colors from the viridis palette
colors = {
    'Test Data Error Rate': 'rgb(68, 1, 84)',  # Deep purple
    'Control Data Error Rate': 'rgb(49, 104, 142)',  # Blue
    'Test Data Completion Rate': 'rgb(33, 145, 140)',  # Teal
    'Control Data Completion Rate': 'rgb(253, 231, 37)'  # Yellow
}

# Error Rate Graph
fig.add_trace(go.Scatter(
    x=error_rate_test.index,
    y=error_rate_test.values,
    mode='lines+markers',
    name='Test Data Error Rate',
    line=dict(shape='linear', color=colors['Test Data Error Rate'])
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=error_rate_control.index,
    y=error_rate_control.values,
    mode='lines+markers',
    name='Control Data Error Rate',
    line=dict(shape='linear', color=colors['Control Data Error Rate'])
), row=1, col=1)

# Completion Rate Graph
fig.add_trace(go.Scatter(
    x=completion_rate_test.index,
    y=completion_rate_test.values,
    mode='lines+markers',
    name='Test Data Completion Rate',
    line=dict(shape='linear', color=colors['Test Data Completion Rate'])
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=completion_rate_control.index,
    y=completion_rate_control.values,
    mode='lines+markers',
    name='Control Data Completion Rate',
    line=dict(shape='linear', color=colors['Control Data Completion Rate'])
), row=2, col=1)

# Update layout
fig.update_layout(
    title_text="Error Rate and Completion Rate (Test and Control Data)",
    height=800,
    width=1000,
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Error Rate", tickformat=".0%", row=1, col=1)
fig.update_yaxes(title_text="Completion Rate", tickformat=".0%", row=2, col=1)

# Show the figure
fig.show()


from scipy.stats import shapiro

# Check normality for total_navigations
stat, p = shapiro(df_test_final['navigations_bt_start_last'])
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))

if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

from scipy.stats import spearmanr, kendalltau

# Spearman correlation for test data
spearman_corr, spearman_p = spearmanr(df_test_final['navigations_bt_start_last'], df_test_final['completion'])
print(f'Spearman correlation: {spearman_corr}, p-value: {spearman_p}')

# Kendall's Tau correlation for test data
kendall_corr, kendall_p = kendalltau(df_test_final['navigations_bt_start_last'], df_test_final['completion'])
print(f'Kendall Tau correlation: {kendall_corr}, p-value: {kendall_p}')

from scipy.stats import shapiro
shapiro_test = shapiro(df_test_final['navigations_bt_start_last'])
print(shapiro_test)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming 'df_test' and 'df_control' DataFrames are already defined and have a 'gendr' column

# Get gender distribution tables
tab_test_gender = df_test['gendr'].value_counts().reset_index(name='count')
tab_control_gender = df_control['gendr'].value_counts().reset_index(name='count')

# Create subplots for side-by-side pie charts
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# --- Custom Viridis-inspired colormap ---

# Get Viridis colors (more than needed)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))  

# Extract specific color indices to get shades of pink, blue, and green
# You might want to play around with these indices to find the exact shades you like
pink_index = 150
blue_index = 200
green_index = 80

custom_colors = [viridis_colors[pink_index], viridis_colors[blue_index], viridis_colors[green_index]]

# --- Plot Pie Charts ---

# Pie chart for test group (left subplot)
axs[0].pie(tab_test_gender['count'], labels=tab_test_gender['gendr'], autopct='%1.1f%%', 
           startangle=140, colors=custom_colors[:len(tab_test_gender)])
axs[0].set_title('Gender Distribution in Test Group')

# Pie chart for control group (right subplot)
axs[1].pie(tab_control_gender['count'], labels=tab_control_gender['gendr'], autopct='%1.1f%%', 
           startangle=140, colors=custom_colors[:len(tab_control_gender)])
axs[1].set_title('Gender Distribution in Control Group')

# Equal aspect ratio for circular shape
axs[0].axis('equal')
axs[1].axis('equal')

# Show the plot
plt.show()




# Create tenure bins of 5 years
bins = np.arange(0, df_final['clnt_tenure_yr'].max() + 5, 5)
labels = [f'{i}-{i+4}' for i in bins[:-1]]

# Bin the tenures
df_final['tenure_bin'] = pd.cut(df_final['clnt_tenure_yr'], bins=bins, labels=labels, right=False)

# Create separate DataFrames for Test and Control groups with tenure bins
df_test_tenure = df_final[df_final['variation'] == 'Test']['tenure_bin'].value_counts().sort_index().reset_index(name='count').rename(columns={'index': 'tenure_bin'})
df_control_tenure = df_final[df_final['variation'] == 'Control']['tenure_bin'].value_counts().sort_index().reset_index(name='count').rename(columns={'index': 'tenure_bin'})

# Setup figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Share y axis for easy comparison

# Colors from the Viridis palette
colors = plt.cm.viridis(np.linspace(0, 1, len(df_test_tenure)))

# Bar chart for Test group
sns.barplot(data=df_test_tenure, x='tenure_bin', y='count', ax=axs[0], palette=colors)
axs[0].set_title('Tenure Distribution in Test Group')
axs[0].set_xlabel('Client Tenure (Years)')
axs[0].set_ylabel('Count')
axs[0].tick_params(axis='x', rotation=45)

# Bar chart for Control group
sns.barplot(data=df_control_tenure, x='tenure_bin', y='count', ax=axs[1], palette=colors)
axs[1].set_title('Tenure Distribution in Control Group')
axs[1].set_xlabel('Client Tenure (Years)')
axs[1].tick_params(axis='x', rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# Create age bins of 10 years
bins = np.arange(0, 110, 10)
labels = [f'{i}-{i+9}' for i in bins[:-1]]

# Bin the ages
df_final['age_bin'] = pd.cut(df_final['clnt_age'], bins=bins, labels=labels, right=False)

# Create separate DataFrames for Test and Control groups with age bins
df_test_age = df_final[df_final['variation'] == 'Test']['age_bin'].value_counts().sort_index().reset_index(name='count').rename(columns={'index': 'age_bin'})
df_control_age = df_final[df_final['variation'] == 'Control']['age_bin'].value_counts().sort_index().reset_index(name='count').rename(columns={'index': 'age_bin'})

# Setup figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Share y axis for easy comparison

# Colors from the Viridis palette
colors = plt.cm.viridis(np.linspace(0, 1, len(df_test_age)))

# Bar chart for Test group
sns.barplot(data=df_test_age, x='age_bin', y='count', ax=axs[0], palette=colors)
axs[0].set_title('Age Distribution in Test Group')
axs[0].set_xlabel('Client Age ')
axs[0].set_ylabel('Count')

# Bar chart for Control group
sns.barplot(data=df_control_age, x='age_bin', y='count', ax=axs[1], palette=colors)
axs[1].set_title('Age Distribution in Control Group')
axs[1].set_xlabel('Client Age')


# Show the plot
plt.tight_layout()
plt.show()



import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Ensure date column is in datetime format
df_final['date'] = pd.to_datetime(df_final['date'])

# Add a 'week' column to the DataFrame
df_final['week'] = df_final['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Calculate weekly completion rate for Test and Control variations
weekly_completion = df_final.groupby(['week', 'variation'])['completion'].mean().unstack()

# Perform Chi-square test for completion rates
contingency_table = pd.crosstab(df_final['variation'], df_final['completion'])
chi2_stat_completion, p_value_completion, _, _ = chi2_contingency(contingency_table)

# Calculate Cramér's V
n = contingency_table.sum().sum()  # Total number of observations
min_dimension = min(contingency_table.shape) - 1  # Minimum dimension - 1
cramers_v = np.sqrt((chi2_stat_completion / n) / min_dimension)

# Print the results
print(f'Chi-square statistic for completion rates: {chi2_stat_completion}, p-value: {p_value_completion}')
print(f"Cramér's V for the association between variation and completion: {cramers_v}")

# Calculate Completion Rates
completion_rates = df_final.groupby('variation')['completion'].mean()
completion_rate_test = completion_rates.get('Test', 0)
completion_rate_control = completion_rates.get('Control', 0)
percentage_increase = ((completion_rate_test - completion_rate_control) / completion_rate_control) * 100 if completion_rate_control != 0 else float('inf')

print(f'Completion rate for Test group: {completion_rate_test}')
print(f'Completion rate for Control group: {completion_rate_control}')
print(f'Percentage increase in completion rate: {percentage_increase}%')

# Define the cost-effectiveness threshold (example: 0.05 or 5%)
cost_effectiveness_threshold = 0.05

# Define viridis-like colors for the variations
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
test_color = 'rgb({}, {}, {})'.format(*[int(c * 255) for c in viridis_colors[68]])
control_color = 'rgb({}, {}, {})'.format(*[int(c * 255) for c in viridis_colors[253]])

# Create a line+scatter plot
fig = go.Figure()

# Line+scatter for test data
fig.add_trace(go.Scatter(
    x=weekly_completion.index,
    y=weekly_completion['Test'],
    mode='lines+markers',
    name='Test Data',
    line=dict(shape='linear', color=test_color)
))

# Line+scatter for control data
fig.add_trace(go.Scatter(
    x=weekly_completion.index,
    y=weekly_completion['Control'],
    mode='lines+markers',
    name='Control Data',
    line=dict(shape='linear', color=control_color)
))

# Add cost-effectiveness threshold line
fig.add_trace(go.Scatter(
    x=weekly_completion.index,
    y=[cost_effectiveness_threshold] * len(weekly_completion),
    mode='lines',
    name='Cost-Effectiveness Threshold',
    line=dict(shape='linear', color='rgba(255, 0, 0, 0.6)', dash='dash')  # Red dashed line
))

# Update layout
fig.update_layout(
    title='Weekly Completion Rate for Test and Control Data with Cost-Effectiveness Threshold',
    xaxis_title='Week',
    yaxis_title='Completion Rate',
    yaxis_tickformat=',.0%',
    legend_title_text='Data Source',
    template='plotly_white'
)

# Show the plot
fig.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_demographics(df, column, bin_size=None):
    plt.figure(figsize=(12, 6))
    if bin_size and pd.api.types.is_numeric_dtype(df[column]):
        bins = range(0, int(df[column].max() + bin_size), bin_size)
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, right=False)
        sns.countplot(data=df, x=f'{column}_binned', hue='variation', palette='viridis')
        plt.xlabel(f'{column.capitalize()} (Binned)')
    else:
        sns.countplot(data=df, x=column, hue='variation', palette='viridis')
        plt.xlabel(column.capitalize())
    plt.title(f'{column.capitalize()} Distribution by Variation')
    plt.ylabel('Count')
    plt.legend(title='Variation')
    plt.xticks(rotation=45)
    plt.show()

# Plot demographic distributions
plot_demographics(df_final, 'clnt_age')
plot_demographics(df_final, 'gendr')
plot_demographics(df_final, 'clnt_tenure_yr', bin_size=4)


df_final['date'] = pd.to_datetime(df_final['date'])
weekly_completion = df_final.groupby(['week', 'variation'])['completion'].mean().unstack()
weekly_error = df_final.groupby(['week', 'variation'])['is_error'].mean().unstack()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
weekly_completion.plot(ax=ax[0], title='Weekly Completion Rate')
weekly_error.plot(ax=ax[1], title='Weekly Error Rate')
plt.tight_layout()
plt.show()


import seaborn as sns

# Example heatmap for navigations
navigation_data = df_final.pivot_table(index='week', columns='variation', values='navigations_bt_start_last', aggfunc='mean')
sns.heatmap(navigation_data, cmap='viridis', annot=True)
plt.title('Heatmap of Average Navigations by Week and Variation')
plt.show()


df_final.shape(10)



if __name__ == "__main__":
    main()
