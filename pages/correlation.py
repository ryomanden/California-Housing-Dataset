import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from main import get_data, get_header


def correlation():
    # Correlation matrix
    data = get_data()
    corr = data.iloc[:,2:].corr()

    print(corr)

    # Plotting the heatmap
    fig1, ax1 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax1)
    fig1.savefig('result/corr_heatmap.png')

    # Plotting the pairplot
    fig2 = sns.pairplot(data.iloc[:,2:])
    fig2.savefig('result/corr_pairplot.png')

    # Display the heatmap on the streamlit
    st.title('Correlation Heatmap')
    st.pyplot(fig1)
    st.title('Pairplot')
    st.pyplot(fig2)


if __name__ == '__main__':
    correlation()
