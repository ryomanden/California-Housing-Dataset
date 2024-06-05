import streamlit as st
import pandas as pd
from pandas import DataFrame

# Load the data
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

# Getters
def get_header():
    return header


def get_data():
    return data


def main():
    st.title('Data')
    st.write(data)
    pass


if __name__ == '__main__':
    main()
