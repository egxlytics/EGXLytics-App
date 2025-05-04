import streamlit as st
from datetime import datetime as dt
import pandas as pd
import numpy as np
import io
from utils.download import get_EGXdata, get_EGX_intraday_data, get_OHLCV_data
from utils.download import _get_intraday_close_price_data
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure the chart axes.
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.reset_index().to_excel(writer, index=False, sheet_name='Portfolio')  # Reset index to include 'Date'
        
        sheet_name = "Portfolio"
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:A', 20)
        chart = workbook.add_chart({'type': 'line'})
        max_row = len(df) + 1
        for i, ticker in enumerate(df.columns.to_list()):
            col = i + 1  # Assuming first column (0) is Date
            chart.add_series({
                'name':[sheet_name, 0, col],
                'categories':[sheet_name, 2, 0, max_row, 0],
                'values':[sheet_name, 2, col, max_row, col],
                'line':{'width': 1.00},
            })
        chart.set_x_axis({'name': 'Date', 'date_axis': True})
        chart.set_y_axis({'name': 'Price', 'major_gridlines': {'visible': False}})
        chart.set_legend({'position': 'top'})
        worksheet.insert_chart('H2', chart)

    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def eod_cache_func(tickers, interval, start, end, date):
    def fetch_single_ticker(ticker):
        df = get_EGXdata([ticker], interval, start, end)
        if isinstance(df, pd.DataFrame):
            df.columns = [ticker]  # Rename to match ticker
        return df

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_single_ticker, ticker) for ticker in tickers]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error fetching ticker: {e}")

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame()  # fallback if no data


st.set_page_config(page_title="Download Data", layout='wide')



st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    /* Global app styles */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Title styling */
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }

    .title {
        font-size: 32px;
        font-weight: 700;
        color: #FFD700; /* Gold */
        margin-left: 10px; /* Space between logo and title */
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A1F29 !important;
        color: #EAEAEA !important;
    }
    
    /* Main content styling */
    .main-container {
        background-color: #10141A !important;
        color: #EAEAEA !important;
    }

    /* Gold buttons with hover effect */
    .stButton>button {
        background-color: #FFD700 !important;
        color: #10141A !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 10px rgba(255, 215, 0, 0.3);
    }

    .stButton>button:hover {
        background-color: #E6C200 !important;
        transform: scale(1.05);
        box-shadow: 0px 6px 15px rgba(255, 215, 0, 0.6);
    }

    /* Tables */
    .stTable {
        background-color: #1A1F29;
        color: #EAEAEA;
        border-radius: 10px;
        padding: 10px;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #B0B0B0;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.page_link("pages/Contact_API.py", label="📞 Contact & API")


# Footer
st.markdown("<p class='footer'> &copy EGXLytics | 100% Free & Open Source</p>", unsafe_allow_html=True)

st.title('Download Data')


##############################
#inputs
##########################
#Tickers
tickers = st.text_input(label='Ticker(s): Enter all Caps',
                       key = 'tickers',
                       value='ABUK',
                      )
tickers = st.session_state.tickers.upper()

interval = st.selectbox(label='Interval',
                       options = ['Daily','Weekly','Monthly','1 Minute','5 Minute','30 Minute'],
                       key='interval',
                      )
interval = st.session_state.interval

start = st.date_input(label='Start date:',
              key='start')
start = st.session_state.start

end = st.date_input(label='End date:',
              key='end')
end = st.session_state.end

date = dt.today().date()

if start < end:
    start_time = time.time()
    if interval in ['1 Minute','5 Minute','30 Minute']:
            df = get_EGX_intraday_data(tickers.split(" "),interval,start,end)

    else:
        df = get_EGXdata(tickers.split(" "),interval,start,end,date)
        df.index = df.index.date
    end_time = time.time()
    st.write(f"Fetched in: {end_time - start_time:.2f} seconds")
    st.write(df)
# Download Button
    st.download_button(
        label="Download Data",
        data=df.to_csv(),
        file_name="Data.csv",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    

else:
    pass


st.write("Note: Intraday data is delayed by 20 minutes.")

