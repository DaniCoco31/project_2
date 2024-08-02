# Vanguard A/B Testing Analysis

## Project Overview
Vanguard, a leading investment management company, is committed to delivering exceptional user experiences. To ensure continuous improvement, Vanguard employs A/B testing of its new User Interface (UI) aimed at enhancing user satisfaction and engagement.

## Datasets
- **Client Profiles:** Demographic and account data of clients.
- **Digital Footprints:** Interaction logs of clients with the website.
- **Experiment Roster:** Details of control and test groups in the experiment.

## Project Structure
- **Data Cleaning and Merging:** Initial processing and integration of datasets.
- **Exploratory Data Analysis (EDA):** Initial insights into client behavior and engagement.
- **Performance Metrics:** Evaluation of key performance indicators between control and test groups.
Completion Rates
![image](https://github.com/user-attachments/assets/4e257246-300a-43a2-b8e8-13638216f908)
Error rate
![image](https://github.com/user-attachments/assets/090cfaf5-c480-446a-9b18-634f21bb15b7)


- **Hypothesis Testing:** Statistical tests to determine the significance of observed differences.
![image](https://github.com/user-attachments/assets/62cc4112-bbc7-4281-aa3a-b3a9d8a1357c)
Average Navigations:
Finding: There is a statistically significant difference in the number of navigations between the start and last steps between the Test and Control groups (t-statistic: 8.70, p-value: 3.33e-18). Recommendation: Examine the navigation patterns to identify if users are struggling with specific steps. Consider implementing user testing sessions to observe how users interact with the site and identify any confusing elements.
![image](https://github.com/user-attachments/assets/4b5c73ed-be11-40ee-85ba-6be31dbb3cbc)
Frequency Of total Navigation
- **PowerBI Visualizations:** Interactive visualizations to illustrate the findings.

## Challenges
- Use of hte new tools and the time.

## Conclusion
- The new UI was effective, as evidenced by the higher completion rates and improved navigation flow in the test group. The similar error rates between both groups, and the overall user experience was enhanced.

## How to Run
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebooks for the analysis.

## Links
- [Project Presentation Slides](https://www.canva.com/design/DAGMf_xZqfQ/xw994e5afTbrq8Z6qF3c3A/edit?utm_content=DAGMf_xZqfQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Kanban Board](https://foregoing-rise-0d2.notion.site/49be1139443f407690fba3621b2d7452?v=cccc3c44450f45b0931570df4c5e4c62)
- [Data Sources](https://github.com/data-bootcamp-v4/lessons/tree/main/5_6_eda_inf_stats_tableau/project/files_for_project)

## Contributors
- [Reetu Sharma]
- [Daniela Trujillo]
