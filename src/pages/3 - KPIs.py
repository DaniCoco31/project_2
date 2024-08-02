from helper import *
import streamlit as st
st.title("KPI's")
  

# Generate plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for the completion rate plot
completion_rate_test = 0.5857757461683248
completion_rate_control = 0.49886769039863504
percentage_increase = 17.421063228256642

# Generate plot for completion rates
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

st.pyplot(plt)

# Markdown text for the completion rate plot
st.title('Completion Rates for Test and Control Groups')
st.pyplot(plt)  # Display the plot in Streamlit
st.markdown('''
### Completion rate for Test group: 0.5858
Completion rate for Control group: 0.4989
Percentage increase in completion rate: 17.42%
            
The observed increase in completion rate meets 
or exceeds the 5% threshold.
The increase in completion rate is statistically
 significant.
''')
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

st.pyplot(plt)

# Analysis of the error rates plot
st.title('Error Rates Analysis for Test and Control Groups')
st.markdown('''
### Analysis:
- **Test Group Error Rate**: 3.62%
- **Control Group Error Rate**: 3.56%
- **Comparison**: The error rate for the test group is slightly higher than the control group.
- **Implications**: This small difference in error rates indicates that the test group experienced a marginally higher error rate.
- **Conclusion**: Despite the increase, the difference is minimal, suggesting both groups performed similarly in terms of error rates.
''')


