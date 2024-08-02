import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helper import *


def main():
    st.title('Vanguard AB Test')
    
    st.write('''
        Asd     

        An A/B test was set into motion from 3/15/2017 to 6/20/2017 by the team.

        Control Group: Clients interacted with Vanguard’s traditional online process.

        Test Group: Clients experienced the new, spruced-up digital interface.

        Both groups navigated through an identical process sequence: an initial page, three subsequent steps, and finally, a confirmation page signaling process completion.

        Goal: To see if the new design leads to a better user experience and higher process completion rates.
        ''')

st.markdown('<a href="/Correlation_between_numerical_variables" target="_self">Next page</a>', unsafe_allow_html=True)
if __name__ == '__main__':
    main()

