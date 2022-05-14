import streamlit as st
import yfinance as yf
from datetime import date
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
dt1='2018-01-01'
dt2=date.today().strftime('%Y-%m-%d')

st.title('Stock Prediction Application :chart_with_upwards_trend::moneybag:')
stocks=['INFY','RELI',"WIT",'TCS.NS','TATASTEEL.NS','NTPC.NS','HDFCLIFE.NS','BAJAJ-AUTO.NS','RELIANCE.NS','COALINDIA.NS','BHARTIARTL.NS','KOTAKBANK.NS','ONGC.NS',
		'ICICIBANK.NS','CIPLA.NS','TATACONSUM.NS','MARUTI.NS','ITC.NS','TECHM.NS','TITAN.NS','WIPRO.NS','HEROMOTOCO.NS','APOLLOHOSP.NS','NESTLEIND.NS','BAJAJFINSV.NS','HINDALCO.NS',
		'SHREECEM.NS','BAJFINANCE.NS']

select_stock=st.selectbox('Select Dataset For Stock: ',stocks)

#title = st.text_input('Select Datasets for Stocks', 'stocks_list')
#st.write('The current dataset is', title)

n_years=st.slider('Years of Prediction',1,5)
days=n_years*365

data_load_state=st.text('loading Data....')

data=yf.download(select_stock,dt1,dt2)
data.reset_index(inplace=True)

data_load_state.text('Loading data.....Done')
st.write(data.tail())
fig=go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock_Open'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock_Close'))
fig.layout.update(title_text="Data with RangeSlider",xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

train_data=data[['Date','Close']]
train_data=train_data.rename(columns={'Date':'ds','Close':'y'})

obj=Prophet()
obj.fit(train_data)
future=obj.make_future_dataframe(periods=days)
forecast=obj.predict(future)
st.write(forecast.tail())

st.header(f"Forecast Plot for {n_years} Years :chart_with_downwards_trend:")

fig1=plot_plotly(obj,forecast)

st.plotly_chart(fig1)

st.header("Forecast Components")

fig2=obj.plot_components(forecast)
st.write(fig2)
