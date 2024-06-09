from main import get_divided_data
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

data = get_divided_data()
model_lr = None

def single_regr_analysis(object_val: str, explanatory_val: str):
    # single regression analysis
    x = data[[object_val]]
    y = data[[explanatory_val]]

    model_lr = LinearRegression()
    model_lr.fit(x, y)

    plt.plot(x, y, 'o')
    plt.plot(x, model_lr.predict(x))
    plt.xlabel(object_val)
    plt.ylabel(explanatory_val)

    st.title('Single Regression Analysis')
    st.pyplot(plt)

    x_pre = st.number_input('Enter the value of the independent variable', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    y_pre = model_lr.predict([[x_pre]])
    st.write(f'The predicted value of {explanatory_val} is {y_pre[0][0]}')

def multi_regr_analysis():
    pass

def regr_analysis():
    with st.form('single_regr_analysis'):
        object_val = st.selectbox('Choose the independent variable', header[2:])
        explanatory_val = st.selectbox('Choose the dependent variable', header[2:])
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            single_regr_analysis(object_val, explanatory_val)

if __name__ == '__main__':
    regr_analysis()