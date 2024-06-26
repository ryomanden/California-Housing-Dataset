import numpy as np
import pandas as pd

from main import get_divided_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats as status
import streamlit as st
import matplotlib.pyplot as plt

# --- INIT --- #

# データの取得
data = get_divided_data()

# セッションの初期化
if 'single_mse_prev' not in st.session_state:
    st.session_state.single_mse_prev = None

if 'single_r2_prev' not in st.session_state:
    st.session_state.single_r2_prev = None

if 'multi_mse_prev' not in st.session_state:
    st.session_state.multi_mse_prev = None

if 'multi_r2_prev' not in st.session_state:
    st.session_state.multi_r2_prev = None


# --- 単回帰分析 --- #

def single_regr_analysis(object_val: str, explanatory_val: str, z_score_threshold=3.0, if_delete=False):
    # --- 外れ値の処理 --- #

    # Z-scoreの計算
    z_score = np.abs(status.zscore(data))  # <-- z = (x - mean) / std

    # 外れ値の削除可否
    if if_delete:
        cleaned_data = data[(z_score < z_score_threshold).all(axis=1)]
    else:
        cleaned_data = data

    st.write(f'Erased: {data.count()[0] - cleaned_data.count()[0]} pts')

    # データの確認
    with st.expander('データの確認'):
        st.write(cleaned_data)

    # データの取得
    x = cleaned_data[[object_val]]
    y = cleaned_data[[explanatory_val]]

    # データの分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 単回帰分析の実行
    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)

    # --- モデルの評価 --- #

    # 予測
    y_pred = model_lr.predict(x_test)

    # 平均二乗誤差 (MSE) の計算
    mse = mean_squared_error(y_test, y_pred)

    # 決定係数 (R^2) の計算
    r2 = r2_score(y_test, y_pred)

    # グラフにプロット
    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_pred)
    plt.xlabel(object_val)
    plt.ylabel(explanatory_val)

    # 前回との差分
    if st.session_state.single_mse_prev is None:
        mse_diff = 0
    else:
        mse_diff = mse - float(st.session_state.single_mse_prev)

    if st.session_state.single_r2_prev is None:
        r2_diff = 0
    else:
        r2_diff = r2 - float(st.session_state.single_r2_prev)

    # セッションの更新
    st.session_state.single_mse_prev = mse
    st.session_state.single_r2_prev = r2

    # --- 結果の表示 --- #

    # グラフの表示
    st.pyplot(plt)

    # 結果の表示
    col1, col2 = st.columns(2)
    col1.metric('平均二乗誤差', mse, mse_diff, delta_color="inverse")
    col2.metric('決定係数', r2, r2_diff)
    st.divider()

    # --- 任意値の予測 --- #

    # テストデータの表示
    with st.expander('テストデータ'):
        st.write(x_test.join(y_test))

    col3, col4 = st.columns(2)
    x_pre = col4.number_input('独立変数の値', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    y_pre = model_lr.predict([[x_pre]])

    # 予測値の表示
    col3.metric(f'{explanatory_val} の予測値', f'{y_pre[0][0]}$')


# --- 重回帰分析 --- #

def multi_regr_analysis(z_score_threshold=3.0, if_delete=False, if_enable=[True, True, True, True, True]):
    # --- 外れ値の処理 --- #

    # Z-scoreの計算
    z_score = np.abs(status.zscore(data))  # <-- z = (x - mean) / std

    # 外れ値の削除可否
    if if_delete:
        cleaned_data = data[(z_score < z_score_threshold).all(axis=1)]
    else:
        cleaned_data = data

    st.write(f'Erased: {data.count()[0] - cleaned_data.count()[0]} pts')

    # データの確認
    with st.expander('データの確認'):
        st.write(cleaned_data)

    # データの取得
    x = cleaned_data[cleaned_data.columns[2:7][if_enable]]
    y = cleaned_data[['medianHouseValue']]

    # データの分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 重回帰分析の実行
    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)

    # --- モデルの評価 --- #

    # 予測
    y_pred = model_lr.predict(x_test)

    # 平均二乗誤差 (MSE) の計算
    mse = mean_squared_error(y_test, y_pred)

    # 決定係数 (R^2) の計算
    r2 = r2_score(y_test, y_pred)

    # 前回との差分
    if st.session_state.multi_mse_prev is None:
        mse_diff = 0
    else:
        mse_diff = mse - float(st.session_state.multi_mse_prev)

    if st.session_state.multi_r2_prev is None:
        r2_diff = 0
    else:
        r2_diff = r2 - float(st.session_state.multi_r2_prev)

    # セッションの更新
    st.session_state.multi_mse_prev = mse
    st.session_state.multi_r2_prev = r2

    # coefficientの計算
    coef = pd.DataFrame(model_lr.coef_, columns=x.columns)

    # --- 結果の表示 --- #

    # 結果の表示
    col1, col2 = st.columns(2)
    col1.metric('平均二乗誤差', mse, mse_diff, delta_color="inverse")
    col2.metric('決定係数', r2, r2_diff)
    st.write(coef)
    st.divider()

    # --- 任意値の予測 --- #

    # テストデータの表示
    with st.expander('テストデータ'):
        st.write(x_test.join(y_test))

    pred = []
    col3, col4 = st.columns(2)
    if if_enable[0]:
        pred.append(col4.number_input('築年数', min_value=0, max_value=100, value=0, step=1))
    if if_enable[1]:
        pred.append(col4.number_input('部屋数', min_value=0, max_value=100, value=0, step=1))
    if if_enable[2]:
        pred.append(col4.number_input('ベッドルーム数', min_value=0, max_value=100, value=0, step=1))
    if if_enable[3]:
        pred.append(col4.number_input('世帯人数', min_value=0, max_value=100, value=0, step=1))
    if if_enable[4]:
        pred.append(col4.number_input('収入（万ドル？）', min_value=0, max_value=100, value=0, step=1))
    y_pre = model_lr.predict([pred])

    # 予測値の表示
    col3.metric('medianHouseValue の予測値', f'{y_pre[0][0]}$')


# --- MAIN --- #

def regr_analysis():
    # --- 単回帰分析form --- #
    with st.form('single_regr_analysis'):
        st.title('単回帰分析')

        # データの選択
        col1, col2 = st.columns(2)
        object_val = col1.selectbox('説明変数の選択', data.columns[2:])
        explanatory_val = col2.selectbox('目的変数の選択', data.columns[2:])

        # 外れ値の処理
        z_score_threshold = st.number_input('Z-scoreの閾値', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        if_delete = st.toggle('外れ値の処理', )

        # 実行ボタン
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            single_regr_analysis(object_val, explanatory_val, z_score_threshold, if_delete)

    # --- 重回帰分析form --- #
    with st.form('multi_regr_analysis'):
        st.title('重回帰分析')

        # 外れ値の処理
        z_score_threshold = st.number_input('Z-scoreの閾値', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        col1, col2 = st.columns(2)
        if_delete = col1.toggle('外れ値の処理')
        if_enable = list(range(5))
        with col2.popover('利用する説明変数'):
            if_enable[0] = st.checkbox('housingMedianAge', value=True)
            if_enable[1] = st.checkbox('totalRooms', value=True)
            if_enable[2] = st.checkbox('totalBedrooms', value=True)
            if_enable[3] = st.checkbox('population', value=True)
            if_enable[4] = st.checkbox('medianIncome', value=True)

        # 実行ボタン
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            multi_regr_analysis(z_score_threshold, if_delete, if_enable)

    st.markdown('''
    ### 作りたいモデル
    各説明変数を入力することで、家賃の予測を行いたい。
    ### やったこと
    - 1世帯の家賃が知りたいので、totalRooms, totalBedrooms, populationを世帯数で割ったデータを使用した。
    - テストデータと訓練データに分割。７：３で分割した。
    - 菊池くんの指摘の通り、1世帯ではありえない値があったので、外れ値として削除した。
        - z scoreをもとに外れ値を切った。
    ''')
    st.latex(r'''z = \frac{データポイント - データセットの平均}{データセットの標準偏差}''')
    st.latex(r'''z = \frac{x - \mu}{\sigma}''')
    st.markdown('''
    ### 検証結果
    - 説明変数は多ければ多いほど良い（今回は）
    - z scoreをもとに外れ値を切ると、少し精度が上がる
    ''')
    with st.container(border=True):
        st.subheader('1世帯あたりのデータ')
        col1, col2 = st.columns(2)
        col1.image('result/divided_corr_heatmap.png', caption='相関係数のヒートマップ')
        col2.image('result/divided_corr_pairplot.png', caption='相関係数のペアプロット')
    with st.container(border=True):
        st.subheader('z scoreで外れ値を削除したデータ')
        col1, col2 = st.columns(2)
        col1.image('result/divided_zscore_corr_heatmap.png', caption='相関係数のヒートマップ')
        col2.image('result/divided_zscore_corr_pairplot.png', caption='相関係数のペアプロット')
    st.markdown('''
    ### 解決できなかった課題
    - 大きい値がクリッピングしているが、原因が分からなかった。
    - z scoreの閾値を小さくすると消えたが、決定係数下がった。
    - scirkitlearnを使わずに実装して比較したかった。
    ''')


if __name__ == '__main__':
    regr_analysis()
