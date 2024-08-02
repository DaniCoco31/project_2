from helper import *
import streamlit as st

# Generate plot
correlation_matrix = df_final.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(20, 18))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 10})  # Adjust font size
plt.title('Correlation Matrix', fontsize=20)  # Adjust title font size
plt.xticks(rotation=45, ha='right', fontsize=12)  # Adjust x-axis tick labels
plt.yticks(rotation=0, fontsize=12)  # Adjust y-axis tick labels

st.title('Correlation between numerical variables')
st.pyplot(plt)  # Display the plot in Streamlit
st.markdown('''
    ### Strong Positive Correlations
    Calls in Last 6 Months vs. Logons in Last 6 Months (0.99)
    This very high correlation indicates that clients who make more calls also tend to log in more frequently.
    Navigations Between Start and Last vs. 1st Step (0.81)
    Clients who navigate between the start and last step frequently also spend significant time on the 1st step.
    Navigations Between Start and Last vs. 2nd Step (0.76)
    Similarly, clients who navigate a lot between the start and last steps also spend more time on the 2nd step.
    Navigations Between Start and Last vs. 3rd Step (0.71)
    There is also a strong correlation indicating that frequent navigators also spend time on the 3rd step.
    Total Time Visit vs. Start Time (0.65)
    Clients who spend more time on their total visit tend to start earlier.
    ### Moderate Positive Correlations
    Step 1 vs. Step 2 (0.37)
    A moderate correlation indicating that clients who complete Step 1 are likely to proceed to Step 2.
    Step 1 vs. Step 3 (0.31)
    Indicates that clients who complete Step 1 are likely to proceed to Step 3 as well.
    Client Age vs. Client Tenure Year (0.32)
    Older clients tend to have been with the company for more years.
    Number of Accounts vs. Balance (0.25)
    Clients with more accounts tend to have a higher balance.
    ### Strong Negative Correlations
    Step 2 vs. Client Tenure Year (-0.38)
    Indicates that newer clients are more likely to complete Step 2.
    3rd Step vs. Client Tenure Year (-0.43)
    Indicates that newer clients are more likely to complete the 3rd step.
    ### Insights:
    Client Engagement:
    Clients who make more calls are also the ones who log in more frequently, suggesting that these clients are more engaged.
    The time spent on each step correlates positively with navigations between start and last step, indicating that clients who spend more time are thorough in their actions.
    New vs. Long-term Clients:
    There is a negative correlation between client tenure and the completion of later steps, suggesting that newer clients are more active in completing steps.
    Older Clients:
    Older clients tend to spend more time on the site and have a longer tenure with the company.
    Balance and Accounts:
    Clients with more accounts tend to have a higher balance, indicating that diversifying accounts may be a strategy for wealthier clients.
''')   
