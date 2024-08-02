from helper import *
import streamlit as st
st.title('Average Navigations Between Start and Last Steps')
  
# Data
data = {
    'Group': ['Test Group', 'Control Group'],
    'Average Navigations': [5.09, 4.91]
}

df = pd.DataFrame(data)

# Use viridis color palette
colors = sns.color_palette("viridis", 2)

# Plot Average Navigations
fig_navigations, ax_navigations = plt.subplots(figsize=(8, 6))
bars = ax_navigations.bar(df['Group'], df['Average Navigations'], color=colors)
for bar in bars:
    yval = bar.get_height()
    ax_navigations.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
ax_navigations.set_title('Average Navigations Between Start and Last Steps', fontsize=16)
ax_navigations.set_ylabel('Average Navigations', fontsize=14)
ax_navigations.set_ylim(0, max(df['Average Navigations']) + 1)

fig_navigations.tight_layout()
plt.show()

# Markdown text for the completion rate plot
st.pyplot(plt)  # Display the plot in Streamlit
st.markdown('''### 3. Average Navigations:
Finding: There is a statistically significant difference in the number of navigations between the start and last steps between the Test and Control groups (t-statistic: 8.70, p-value: 3.33e-18).
Recommendation: Examine the navigation patterns to identify if users are struggling with specific steps. Consider implementing user testing sessions to observe how users interact with the site and identify any confusing elements.
''')