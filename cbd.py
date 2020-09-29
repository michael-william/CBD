from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import xlrd
from pytrends.request import TrendReq
from country_list import available_languages
from country_list import countries_for_language

import warnings
warnings.filterwarnings("ignore")

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.express as px


from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet

from sklearn.preprocessing import MinMaxScaler

#initiate plotly
pyoff.init_notebook_mode()

response = requests.get('https://github.com/michael-william/CBD/raw/master/resources/DS_logo_color.png')
img = Image.open(BytesIO(response.content))
st.image(img,caption='dataandstories.com',width=200)
st.write("""
# Keyword Interest & Prediction App
I built this app originally predicts the interest in the search term 'cbd oil' via Google Trends using the fbprophet package.
However, I've made it possible for you to put in your own keywords to satisfy other curiosities. Numbers represent search interest relative to the highest point on the chart for the given region and time. 
A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular. 
A score of 0 means there was not enough data for this term.
""")

countries = dict(countries_for_language('en'))
countries_df = pd.DataFrame.from_dict(countries.items())
countries_df.columns = ['Code', 'Name']

def user_input_features():
        countries = ['Worldwide']
        countries_list = list(countries_df['Name'])
        for c in countries_list:
            countries.append(c)
        country = st.selectbox('Country', countries)
        k_words = st.text_input('Keyword', 'CBD oil')
        return country, k_words

def main():
    

    st.header('Parameters')
    keywords=[]
    country, k_words = user_input_features()
    keywords.append(k_words)

    def c_code():
        if country == 'Worldwide':
            c_code = ''
        else:
            c_code = countries_df[countries_df['Name']==country]['Code'].values[0]
        return c_code
    
    c_code = c_code()

    #pytrend = TrendReq(timeout=(10,25), hl='en-US', tz=360)
    #pytrend.build_payload(kw_list=keywords, timeframe='today 5-y', geo=c_code)
    #five_years = pytrend.interest_over_time().reset_index()
    #five_years = five_years[:-1]
    #five_years.drop('isPartial', axis=1, inplace=True)
    #five_years.columns = ['ds', 'y']
    #temp = five_years.copy()
    #st.write(temp)

    def predict():
        pytrend = TrendReq(timeout=(10,25), hl='en-US', tz=360)
        pytrend.build_payload(kw_list=keywords, timeframe='today 5-y', geo=c_code)
        five_years = pytrend.interest_over_time().reset_index()
        if five_years[keywords].sum().values[0] < 5:
            return 'not enough data', 'not enough data', 'not enough data', 'not enough data', 'not enough data'
        else:
            #five_years = five_years[five_years.isPartial == 'False']
            five_years = five_years[:-1]
            five_years.drop('isPartial', axis=1, inplace=True)
            five_years.columns = ['ds', 'y']
            temp = five_years.copy()
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
            fig.update_layout(template='plotly_white',title='{} "{}" Interest Prediction'.format(country,keywords[0]), xaxis={'title':'Date'}, yaxis={'title':'Intrest & Prediction'})
            fig.update_traces(line_color="#F66162",
                  selector=dict(line_color='#0072B2'))
            fig.update_traces(marker_color="royalblue",
                  selector=dict(marker_color='black'))
            comp = plot_components_plotly(m, forecast)
            comp.update_layout(template='plotly_white',title='{} "{}" Seasonality Prediction'.format(country,keywords[0]),yaxis={'title':'Prediction'})
            comp.update_traces(line_color="#F66162",
                  selector=dict(line_color='#0072B2'))
            comp.update_traces(marker_color="royalblue",
                  selector=dict(marker_color='black'))
            return fig, comp, forecast, periods, pytrend

    #if st.button('Analyze and Predict'):
        #p_df = predict_df()
    fig, comp, forecast, periods, pytrend = predict()
    if fig == 'not enough data':
        st.write('Not enough data')
    else:
        st.subheader('5 Year Interest + 90 Day Projection')
        forecast_display = forecast[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']]
        forecast_display.columns = ['Week', 'Prediction', 'Trend', 'Lower range', 'Upper range']
        forecast_display = forecast_display.set_index('Week')
        forecast_display = np.ceil(forecast_display[['Prediction', 'Trend', 'Lower range', 'Upper range']])
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(comp, use_container_width=True)

        breakdown_df = pytrend.interest_by_region(inc_low_vol=True, inc_geo_code=False).reset_index().sort_values(keywords, ascending =False)
        breakdown_df.columns = ['Location', 'Relative interest']
        bd_fig = px.bar(breakdown_df.head(10), x='Location', y='Relative interest', color='Relative interest')
        bd_fig.update_layout(title ='Top 10 Locations by Interest for "{}"'.format(keywords[0]))
        st.plotly_chart(bd_fig, use_container_width=True, use_container_height=True)
        
        related_queries = pytrend.related_queries()
        key = list(related_queries.keys())[0]
        top_query = related_queries[key]['top'].head(10)
        rising_query = related_queries[key]['rising'].head(10)
        top_query.columns = ['Top Related Query', 'Relative interest']
        rising_query.columns = ['Rising Related Query', 'Value']
        all_cols = ['Top Related Query', 'Relative interest', 'Rising Related Query', 'Value']
        top_fig = go.Figure(data=[go.Table(
                                header=dict(values=all_cols,
                                            fill_color='paleturquoise',
                                            align='left'),
                                cells=dict(values=[top_query['Top Related Query'], top_query['Relative interest'], rising_query['Rising Related Query'], rising_query['Value']],
                                           fill_color='lavender',
                                           align='left'))
                            ])
        top_fig.update_layout(title='Top 10 Related and Rising Queries for "{}"'.format(keywords[0]), height=500)

        
        st.markdown('Users searching for your term also searched for these quries.Top Queries are the most popular queries.Scoring is on a relative scale where a value of 100 is the most commonly searched query and a value of 50 is a query searched half as often as the most popular term, and so on. Rising Queries are topics with the biggest increase in search frequency since the last time period. Results marked "Breakout" had a tremendous increase, probably because these queries are new and had few (if any) prior searches.')
        st.plotly_chart(top_fig, use_container_width=True)
        
        st.markdown('Users searching for your term also searched for these topics.Top Topics are the most popular topics.Scoring is on a relative scale where a value of 100 is the most commonly searched topic and a value of 50 is a topic searched half as often as the most popular term, and so on. Rising Related are topics with the biggest increase in search frequency since the last time period. Results marked "Breakout" had a tremendous increase, probably because these topics are new and had few (if any) prior searches.')
        related_topics = pytrend.related_topics()
        key = list(related_topics.keys())[0]
        top_topic = related_topics[key]['top'].drop(['formattedValue', 'hasData', 'topic_mid'], axis=1).head(10)
        top_topic.columns = ['Value','Link', 'Top 10 Topic Titles', 'Topic Type']
        rising_topic = related_topics[key]['rising'].head(10)
        rising_topic = rising_topic[['value', 'formattedValue','link','topic_title','topic_type']]
        rising_topic.columns = ['Value','Value Type','Link', 'Top 10 Rising Topic Titles', 'Topic Type']

        top_topic['Link'] = [('trends.google.com'+i) for i in top_topic['Link']]
        rising_topic['Link'] = [('trends.google.com'+i) for i in rising_topic['Link']]

        #def make_clickable(val):
         #   return '<a target="_blank" href="{}">{}</a>'.format(val, val)

        #top_topic = top_topic.head().style.format({'Link': make_clickable})
        #rising_topic = rising_topic.head().style.format({'Link': make_clickable})
        #all_cols = ['Top Related Query', 'Relative interest', 'Rising Related Query', 'Value']
        topic_fig = go.Figure(data=[go.Table(
                                header=dict(values=['Value','Link', 'Top 10 Topic Titles', 'Topic Type'],
                                            fill_color='paleturquoise',
                                            align='left'),
                                cells=dict(values=[top_topic['Value'], top_topic['Link'], top_topic['Top 10 Topic Titles'], top_topic['Topic Type']],
                                           fill_color='lavender',
                                           align='left'))
                            ])
        topic_fig.update_layout(title='Top 10 Related Topics for "{}"'.format(keywords[0]), height=550)
        st.plotly_chart(topic_fig, use_container_width=True)

        rising_fig = go.Figure(data=[go.Table(
                                header=dict(values=['Value','Value Type','Link', 'Top 10 Rising Topic Titles', 'Topic Type'],
                                            fill_color='paleturquoise',
                                            align='left'),
                                cells=dict(values=[rising_topic['Value'],rising_topic['Value Type'], rising_topic['Link'], rising_topic['Top 10 Rising Topic Titles'], rising_topic['Topic Type']],
                                           fill_color='lavender',
                                           align='left'))
                            ])
        rising_fig.update_layout(title='Top 10 Rising Related Topics for "{}"'.format(keywords[0]), height=550)
        st.plotly_chart(rising_fig, use_container_width=True)
       
         
        


if __name__ == "__main__":
    main()