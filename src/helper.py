import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from functions import *
# raw data
df_clients_profile = pd.read_csv('Data/Raw/df_final_demo.txt')
df_web_data_1 = pd.read_csv('Data/Raw/df_final_web_data_pt_1.txt')
df_web_data_2 = pd.read_csv('Data/Raw/df_final_web_data_pt_2.txt')
df_experiment_clients = pd.read_csv('Data/Raw/df_final_experiment_clients.txt')
# processed data
df_test = pd.read_csv('Data/Cleaned_Data/df_test.csv')
df_test_final = pd.read_csv('Data/Cleaned_Data/df_test_final.csv')
df_control = pd.read_csv('Data/Cleaned_Data/df_control.csv')
df_control_final = pd.read_csv('Data/Cleaned_Data/df_control_final.csv')
df_final = pd.read_csv('Data/Cleaned_Data/df_final.csv')
df_combined = pd.read_csv('Data/Cleaned_Data/df_combined.csv')
pd.set_option('display.max_columns', None)
df_web_data = pd.concat([df_web_data_1, df_web_data_2], ignore_index= True)


df_test = convert_data_types_combined(df_test)
df_test_final = convert_data_types_combined(df_test_final)
df_control = convert_data_types_combined(df_control)
df_control_final = convert_data_types_combined(df_control_final)
df_final = convert_data_types_combined(df_final)
df_combined = convert_data_types_combined(df_combined)


