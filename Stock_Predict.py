import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .main{
        background-color:#F5F5F5;
    }
            footer {
	
	        visibility: hidden;
	
	}
        footer:after {
            content:'© 2021 Aman Rath, All Rights Reserved'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# App title
st.markdown('''
# Stock Forecast App
Shown are the stock price data for User selected company!''')



# Sidebar
st.sidebar.subheader('Stock Specification')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = datetime.date.today().strftime("%Y-%m-%d")
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Select Stock Ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
data = load_data(tickerSymbol)


# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker Raw Data**')
st.write(tickerDf.tail(30))

# Plot raw data
st.header('**Company History from Start date**')
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.header('**Forecast data**')
st.write(forecast.tail())
    
st.header(f'**Forecast plot for {n_years} years**')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.header("**Forecast components**")
fig2 = m.plot_components(forecast)
st.write(fig2)


st.write('---')
st.write('Company Info and Details (For Analysis)')
st.write(tickerData.info)

st.write('---')
# App title
st.markdown('''
**Credits**
- App built by [Aman Rath](https://rathaman1711.github.io/Myportfolio/). Check my portfolio for other projects.
- Built in `Python` using `fbprophet`,`streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')

