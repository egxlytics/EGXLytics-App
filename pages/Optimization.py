import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from utils.optimize import Portfolio

def to_excel(df, weights_df, cor_matrix, optimization_type):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        
        sheet1_name = "Weights & Correlations"
        weights_df.to_excel(writer, sheet_name=sheet1_name, startrow=1, index=False)
        cor_matrix.to_excel(writer, sheet_name=sheet1_name, startrow=len(weights_df)+5, index=True)
        workbook = writer.book
        worksheet = writer.sheets[sheet1_name]

        percent_format = workbook.add_format({'num_format': '0.000%'})
        worksheet.set_column(1, 1, 15, percent_format)
        chart = workbook.add_chart({'type': 'pie'})

        if optimization_type == 'Index Tracking':
            
            chart.add_series({
                'name':       'Portfolio Allocation',
                'categories': [sheet1_name, 2, 0, 1 + len(weights_df), 0],  # Ticker names (index)
                'values':     [sheet1_name, 2, 1, 1 + len(weights_df), 1],  # Weights column
                'data_labels': {'percentage': True, 'category': True},     # Show category and % in chart
            })
        else:
                        chart.add_series({
                'name':       'Portfolio Allocation',
                'categories': [sheet1_name, 2, 0, 1 + len(weights_df), 0],  # Ticker names (index)
                'values':     [sheet1_name, 2, 2, 1 + len(weights_df), 2],  # Weights column
                'data_labels': {'percentage': True, 'category': True},     # Show category and % in chart
            })


        chart.set_title({'name': 'Portfolio Allocation'})
        worksheet.insert_chart('D2', chart)
        
        
        df.reset_index().to_excel(writer, index=False, sheet_name='Portfolio Vs Benchmark')  # Reset index to include 'Date'
        
        sheet2_name = "Portfolio Vs Benchmark"
        workbook = writer.book
        worksheet = writer.sheets[sheet2_name]
        worksheet.set_column('A:A', 20)
        chart = workbook.add_chart({'type': 'line'})
        max_row = len(df) + 1
        for i, ticker in enumerate(df.columns.to_list()):
            col = i + 1  # Assuming first column (0) is Date
            chart.add_series({
                'name':[sheet2_name, 0, col],
                'categories':[sheet2_name, 2, 0, max_row, 0],
                'values':[sheet2_name, 2, col, max_row, col],
                'line':{'width': 1.00},
            })
        chart.set_x_axis({'name': 'Date', 'date_axis': True})
        chart.set_y_axis({'name': 'Price', 'major_gridlines': {'visible': False}})
        chart.set_legend({'position': 'top'})
        worksheet.insert_chart('D2', chart)


        sheet3_name = "Returns Distribution"
        returns = df.pct_change().dropna()
        
        hist, bin_edges = np.histogram(returns, bins=100)
        # Create a DataFrame for plotting
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # use midpoints for better representation

        # DataFrame with numeric bin midpoints
        hist_df = pd.DataFrame({'Return Bin': bin_midpoints,
                                'Frequency': hist
                               })
        hist_df.to_excel(writer, sheet_name=sheet3_name, startrow=1, index=False)
        # Access workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet3_name]
        
        percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column(0, 0, 15, percent_format)
        
        # Create a column chart
        chart = workbook.add_chart({'type': 'column'})

        # Add histogram series
        chart.add_series({
            'name':       'Return Distribution',
            'categories': [sheet3_name, 2, 0, 1 + len(hist), 0],  # Bins
            'values':     [sheet3_name, 2, 1, 1 + len(hist), 1],  # Frequencies
            'gap':        1
        })
        chart.set_title({'name': 'Portfolio Returns Distribution'})
        chart.set_x_axis({'name': 'Return Range'})
        chart.set_y_axis({'name': 'Frequency'})
        worksheet.insert_chart('D2', chart)

    processed_data = output.getvalue()
    return processed_data






##################
#Streamlit app
####################

st.set_page_config(page_title='Portfolio Optimization', layout='wide')
st.title('Portfolio Optimization')


#########
#inputs
##########

close = st.file_uploader(label='Upload CSV:',
                 type='csv',
                 key='close')
close = st.session_state.close

optimization_input_cols = st.columns([0.25,0.25,0.25,0.25])
with optimization_input_cols[0]:
    optimization_type = st.selectbox(label='Optimization Type',
                                     options=['Sharpe','Minimize Risk', 'Maximum Return', 'Index Tracking'])

if optimization_type == 'Index Tracking':
    with optimization_input_cols[1]:
        equity_constraint = st.number_input(label='Maximum Equity Allocation',
                                                value = 0.5,
                                                key='equity_constraint')
        equity_constraint = st.session_state.equity_constraint

    index_tracking_options_col = st.columns([0.5,0.5])
    with index_tracking_options_col[0]:
        risk_free_rate = st.number_input(label='Risk Free Rate',
                                        value = 0.02,
                                        key='risk_free_rate')
    risk_free_rate = st.session_state.risk_free_rate

    with index_tracking_options_col[1]:
        holding_period = st.number_input(label='Holding Period Days',
                                        value = 252,
                                        key='Holding Period:')



else:
    with optimization_input_cols[1]:
        risk_measure = st.selectbox(label='Risk Measure:',
                                    options=['MV','CVaR'])

    with optimization_input_cols[2]:
        equity_constraint = st.number_input(label='Maximum Equity Allocation',
                                            value = 0.5,
                                            key='equity_constraint')
    equity_constraint = st.session_state.equity_constraint

    with optimization_input_cols[3]:
        sector_constraint = st.number_input(label='Maximum Sector Allocation',
                                            value = 0.5,
                                            key='sector_constraint')
    sector_constraint = st.session_state.sector_constraint


    other_input_cols = st.columns([0.5,0.5])
    with other_input_cols[0]:
        risk_free_rate = st.number_input(label='Risk Free Rate',
                                        value = 0.02,
                                        key='risk_free_rate')
    risk_free_rate = st.session_state.risk_free_rate

    with other_input_cols[1]:
        holding_period = st.number_input(label='Holding Period Days',
                                        value = 252,
                                        key='Holding Period:')



benchmark_path = st.file_uploader(label='Upload CSV Of Desired Benchmark',
                 type='csv',
                 key='benchmark_path')
benchmark_path = st.session_state.benchmark_path
if benchmark_path:
    benchmark = pd.read_csv(benchmark_path, index_col=0, header=0)


###########
#############

if close:
    close = pd.read_csv(close, index_col=0, header=0)

    market_info = pd.read_csv('egx_companies.csv',index_col=None)
    market_info.columns = ['Ticker','Company Name', 'Sector','Exchange']
    market_info.set_index('Ticker',inplace=True, drop=False)
    ticker_sector_df = market_info.loc[close.columns,['Ticker','Sector']].reset_index(drop=True)
    ###############

    optimization_type_dic = {'Sharpe':'Sharpe', 'Minimize Risk':'MinRisk', 'Maximum Return':'MaxRet'}
        
    pf = Portfolio(close)

    if optimization_type=='Index Tracking':


        if benchmark_path:
            portfolio_allocation, portfolio = pf.index_tracking(equity_constraint=equity_constraint,benchmark=benchmark)
        else:
            e = RuntimeError("Benchmark File not found")
            st.error("Upload Benchmark CSV")
            st.exception(e)
            st.stop()

    else:
        portfolio_allocation, portfolio = pf.optimize(ticker_sector_df=ticker_sector_df,
                                                    risk_measure=risk_measure,
                                                    objective_function=optimization_type_dic[optimization_type],
                                                    sector_constraint=sector_constraint,
                                                    equity_constraint=equity_constraint)

    
    portfolio_allocation = (portfolio_allocation[portfolio_allocation.Weight.round(3)>0.00]).sort_values(by='Weight',ascending=False).reset_index()
        
    cols = st.columns([0.33,0.33,0.33])
    with cols[0]:
        '''### Portfolio Allocation'''
        portfolio_weights = st.data_editor(portfolio_allocation, disabled=["Ticker","Sector"]) 
        
        portfolio = close.loc[:,portfolio_weights.Ticker] @ portfolio_weights.Weight.values.reshape((-1,1))
        portfolio.dropna(inplace=True)
        portfolio.columns=["Portfolio"]

        portfolio_expected_return, portfolio_risk, cvar, sharpe = pf.portfolio_performance(portfolio.pct_change().dropna(), n=holding_period, risk_free_rate=risk_free_rate)  

        st.write(f'''Expected Return: {portfolio_expected_return:.2%} Sharpe Ratio: {sharpe:.2f} \n
                 Risk: {portfolio_risk:.2%}  CVaR 95% : {cvar:.2%}''')
     
        if portfolio_weights.Weight.sum().round(2) < 1.0:
            st.warning(
                '''Warning: Portfolio allocation must sum to 1.0. Relax diversification threshold, use Pie Chart proportionate allocation for Unused Cash, or modify manually from table.
                **Current Unused Cash {:.2f}**'''.format(1-portfolio_weights.Weight.sum())
                      )
        
        elif portfolio_weights.Weight.sum().round(2) > 1.0:
            st.warning(
                '''Warning: Portfolio allocation must sum to 1.0. Use Pie Chart proportionate allocation, or modify manually from the table.
                **Current Leverage {:.2f}**'''.format(portfolio_weights.Weight.sum())
                      )
            
    with cols[1]:
        fig = px.pie(portfolio_weights, values='Weight', names='Ticker', title='Portfolio Weights')
        st.plotly_chart(fig)
    
    with cols[2]:
        if benchmark_path:
            portfolio["Benchmark"] = benchmark
            portfolio.dropna(inplace=True)


            
            comparison = portfolio/portfolio.values[0]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=comparison.index, y=comparison["Portfolio"], name="Portfolio", marker=dict(color='deepskyblue'))
            )
            fig.add_trace(
                go.Scatter(x=comparison.index, y=comparison["Benchmark"], name="Benchmark",marker=dict(color='Red'))

            )
            st.plotly_chart(fig)
        else:
            comparison=portfolio

        
    '''### Correlation Matrix'''
    last_n = st.slider(label='Last N Days',
              min_value=5,
              max_value=252)
    f'''##### Last {last_n} Day Portfolio Correlation Matrix'''
    
    if benchmark_path:
        df = pd.merge(left=benchmark,right=close, left_on=benchmark.index, right_on=close.index).set_index('key_0',drop=True)
        df.index.name = 'Date'
        cor_matrix = df.pct_change().dropna().tail(last_n).corr()
    else:
        cor_matrix =close.pct_change().dropna().tail(last_n).corr()

    mask = np.zeros_like(cor_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Viz
    cor_matrix_viz = cor_matrix.mask(mask).dropna(how='all')
    fig = px.imshow(cor_matrix_viz.iloc[:,:-1].round(2),labels=dict(x="", y="", color="Correlation"),
                    color_continuous_scale='RdBu_r', text_auto=True, aspect = 'auto')
    
    fig.layout.height = 800
    fig.layout.width = 1200
    st.plotly_chart(fig)

    st.download_button(
    label="Download Report",
    data=to_excel(comparison, portfolio_weights,cor_matrix,optimization_type),
    file_name="Comparison.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
