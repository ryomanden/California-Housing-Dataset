import streamlit as st
import pandas as pd
from pandas import DataFrame

# --- LOAD --- #
header: list[str] = [
    'longitude',
    'latitude',
    'housingMedianAge',
    'totalRooms',
    'totalBedrooms',
    'population',
    'households',
    'medianIncome',
    'medianHouseValue'
]
data: DataFrame = pd.read_csv('CaliforniaHousing/cal_housing.data', header=None, names=header)
divided_data = data.copy()
divided_data['totalRooms'] = data['totalRooms'] / data['households']
divided_data['totalBedrooms'] = data['totalBedrooms'] / data['households']
divided_data['population'] = data['population'] / data['households']
divided_data.drop(columns=['households'], inplace=True)


def main():
    st.title('Raw Data')
    st.write(data)
    st.title('Divided Data')
    st.write(divided_data)
    st.title('Data Description')
    st.write(divided_data.describe())


# --- GETTER --- #
def get_header():
    return header


def get_data():
    return data


def get_divided_data():
    return divided_data


if __name__ == '__main__':
    main()
