from main import get_divided_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

data = get_divided_data()
model_lr = None


# --- 単回帰分析 --- #
def single_regr_analysis(object_val: str, explanatory_val: str):
    # 単回帰分析
    x = data[[object_val]]
    y = data[[explanatory_val]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # モデルの学習
    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)

    # --- モデルの評価 --- #

    # 予測
    y_pred = model_lr.predict(x_test)

    # 平均二乗誤差 (MSE) の計算
    mse = mean_squared_error(y_test, y_pred)

    # 決定係数 (R^2) の計算
    r2 = r2_score(y_test, y_pred)

    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_pred)
    plt.xlabel(object_val)
    plt.ylabel(explanatory_val)

    # 結果の表示
    st.pyplot(plt)

    st.write(f'平均二乗誤差: {mse}')
    st.write(f'決定係数: {r2}')

    # 任意値の予測
    x_pre = st.number_input('独立変数の値', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    y_pre = model_lr.predict([[x_pre]])

    st.write(f'{explanatory_val} の予測値は {y_pre[0][0]}')

# --- 重回帰分析 --- #
def multi_regr_analysis():
    # 重回帰分析
    x = data[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'medianIncome']]
    y = data[['medianHouseValue']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # モデルの学習
    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)

    # --- モデルの評価 --- #

    # 予測
    y_pred = model_lr.predict(x_test)

    # 平均二乗誤差 (MSE) の計算
    mse = mean_squared_error(y_test, y_pred)

    # 決定係数 (R^2) の計算
    r2 = r2_score(y_test, y_pred)

    # 結果の表示
    st.write(f'平均二乗誤差: {mse}')
    st.write(f'決定係数: {r2}')

    # 任意値の予測
    housingMedianAge = st.number_input('housingMedianAge', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    totalRooms = st.number_input('totalRooms', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    totalBedrooms = st.number_input('totalBedrooms', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    population = st.number_input('population', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    medianIncome = st.number_input('medianIncome', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    y_pre = model_lr.predict([[housingMedianAge, totalRooms, totalBedrooms, population, medianIncome]])

    st.write(f'medianHouseValue の予測値は {y_pre[0][0]}')


def regr_analysis():
    with st.form('single_regr_analysis'):
        st.title('単回帰分析')
        object_val = st.selectbox('説明変数の選択', data.columns[2:])
        explanatory_val = st.selectbox('目的変数の選択', data.columns[2:])
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            single_regr_analysis(object_val, explanatory_val)

    with st.form('multi_regr_analysis'):
        st.title('重回帰分析')
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            multi_regr_analysis()


if __name__ == '__main__':
    regr_analysis()
