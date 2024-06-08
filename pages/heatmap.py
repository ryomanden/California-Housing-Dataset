import streamlit as st
from main import get_data, get_header
import pydeck as pdk

data = get_data()
header = get_header()


def plotHeatmap():
    st.title('Heatmap')
    select_heatmap = st.selectbox('Choose the heatmap variable', header[2:])
    pitch = st.slider('Select the weight of the heatmap', 0, 100, 0)
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(
            latitude=37.7749295,
            longitude=-122.4194155,
            zoom=5,
            pitch=pitch,
        ),
        layers=[
            pdk.Layer(
                'HeatmapLayer',
                data=data,
                get_position='[longitude, latitude]',
                get_weight=select_heatmap,
                radius=1000,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))


if __name__ == '__main__':
    plotHeatmap()
