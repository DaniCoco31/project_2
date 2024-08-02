from helper import *
from functions import * 
import streamlit as sl

sl.title('Completion Rates for Test and Control Groups')


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

plt.show()

# Markdown text for the completion rate plot
sl.pyplot(plt)  # Display the plot in Streamlit
sl.markdown('''
### Completion rate for Test group: 0.5858
Completion rate for Control group: 0.4989
Percentage increase in completion rate: 17.42%
''')