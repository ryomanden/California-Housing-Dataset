import numpy as np
from scipy import stats as status
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import get_data, get_divided_data


def correlation():
    col1, col2 = st.columns(2)
    # データの選択
    sel_data = col1.radio('データの選択', ('Raw Data', 'Divided Data'))
    if sel_data == 'Raw Data':
        data = get_data()
    else:
        data = get_divided_data()

    # 外れ値の処理
    z_score_threshold = col2.number_input('Z-scoreの閾値', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    if_delete = col2.toggle('外れ値の処理')

    z_score = np.abs(status.zscore(data))
    if if_delete:
        cleaned_data = data[(z_score < z_score_threshold).all(axis=1)]
    else:
        cleaned_data = data


    # 相関係数の計算
    corr = cleaned_data.iloc[:, 2:].corr()

    # ヒートマップの描画
    fig1, ax1 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax1)
    fig1.savefig('result/corr_heatmap.png')

    # ペアプロットの描画
    fig2 = sns.pairplot(data.iloc[:,2:])
    fig2.savefig('result/corr_pairplot.png')

    # 表示
    st.title('Correlation Heatmap')
    st.pyplot(fig1)
    st.title('Pairplot')
    st.pyplot(fig2)


if __name__ == '__main__':
    correlation()
