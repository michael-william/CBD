from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet

st.write("""
# CBD Oil Interest Prediction App
This app predicts the interest in the search term 'cbd oil' via Google Trends.
Numbers represent search interest relative to the highest point on the chart for the given region and time. 
A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular. 
A score of 0 means there was not enough data for this term.
""")

@st.cache
def load_data(filename=None):
    data_source_us = 'https://github.com/michael-william/CBD/raw/master/resources/cbd_us.xlsx'
    df_us=pd.read_excel(data_source_us)
    data_source_sa = 'https://github.com/michael-william/CBD/raw/master/resources/cbd_sa.xlsx'
    df_sa=pd.read_excel(data_source_sa)
    data_source_nl = 'https://github.com/michael-william/CBD/raw/master/resources/cbd_nl.xlsx'
    df_nl=pd.read_excel(data_source_nl)
    return df_us, df_sa, df_nl

df_us, df_sa, df_nl = load_data()

def user_input_features():
        countries = ['South Africa', 'Netherlands', 'United States']
        country = st.selectbox('Country', countries)
        #latitude = st.sidebar.slider('Latitude', 50.770041, 53.333967, 51.2)
        #longitude = st.sidebar.slider('Longitude', 3.554188, 7.036756, 5.2)
        #data = {'Description': item,
                #'Country': country}
        #features = pd.DataFrame(data, columns = ['Description', 'Country'], index=[0])
        return country

def main():
    

    st.header('Country')
    country = user_input_features()
    
    def predict_df():
        if country == 'South Africa':
            predict_df = df_sa
        elif country == 'Netherlands':
            predict_df = df_nl
        else: 
            predict_df = df_us
        return predict_df 

    def predict():
        temp = p_df.copy()
        temp['cap'] = temp['y'].max()
        temp['floor'] = temp['y'].min()
        m = Prophet()
        fit = m.fit(temp)
        periods = 90
        future = m.make_future_dataframe(periods=periods)
        future["floor"] = temp['y'].min()
        future["cap"] = temp["y"].max()
        forecast = m.predict(future)
        fig = plot_plotly(m, forecast)
        fig.update_layout(title='{} CBD Oil Intrest Prediction'.format(country), xaxis={'title':'Date'}, yaxis={'title':'Intrest Prediction'})
        return fig, forecast, periods

    if st.button('Predict'):
        p_df = predict_df()
        fig, forecast, periods = predict()
        st.subheader('Predictive Sales Performance')
        forecast_display = forecast[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']]
        forecast_display.columns = ['Week', 'Prediction', 'Trend', 'Lower range', 'Upper range']
        forecast_display = forecast_display.set_index('Week')
        forecast_display = np.ceil(forecast_display[['Prediction', 'Trend', 'Lower range', 'Upper range']])
        st.plotly_chart(fig, use_container_width=True)
        st.write(forecast_display.tail(periods))
        

    

    
    #if st.sidebar.button('Predict'):
         
        


if __name__ == "__main__":
    main()