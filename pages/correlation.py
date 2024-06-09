import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from main import get_data, get_header


def correlation():
    # データの選択
    sel_data = st.radio('データの選択', ('Raw Data', 'Divided Data'))
    if sel_data == 'Raw Data':
        data = get_data()
    else:
        data = get_divided_data()

    # 相関係数の計算
    corr = data.iloc[:, 2:].corr()

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
