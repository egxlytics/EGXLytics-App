import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sc
import streamlit as st
import io

def to_excel(df, weights_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        
        sheet1_name = "Portfolio Weights"
        weights_df.to_excel(writer, sheet_name=sheet1_name, startrow=1, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet1_name]

        percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column(1, 1, 15, percent_format)
        chart = workbook.add_chart({'type': 'pie'})

        chart.add_series({
            'name':       'Portfolio Allocation',
            'categories': [sheet1_name, 2, 0, 1 + len(weights_df), 0],  # Ticker names (index)
            'values':     [sheet1_name, 2, 1, 1 + len(weights_df), 1],  # Weights column
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



def stock_performance(close):
    returns = close.pct_change()
    mean_returns = returns.mean()
    cov = returns.cov()
    return mean_returns, cov

def portfolio_performance(W, mean_returns, cov, n):
    W = np.asarray(W)
    portfolio_returns = (np.dot(W, mean_returns) * n)
    portfolio_risk = np.sqrt(W.T @ cov @ W) * np.sqrt(n)
    return portfolio_returns, portfolio_risk

############################
def negative_sharpe_ratio(W, mean_returns, cov, risk_free_rate,n):
    portfolio_return, portfolio_risk = portfolio_performance(W, mean_returns, cov, n)
    neg_sharpe_ratio = -(portfolio_return - risk_free_rate)/portfolio_risk
    return neg_sharpe_ratio

def optimize_portfolio(mean_returns, cov, upper_bound, risk_free_rate,n):
    """
    returns
    -------
    sharpe_ratio, optimal_weights
    """
    #assign random weights
    np.random.seed(1)
    W = np.random.random(len(mean_returns))
    W = [weight/ np.sum(W) for weight in W]

    #add bounds
    bound = (0,upper_bound)
    bounds = tuple(bound for w in range(len(W)))

    #constraint
    def constraint(W):
        return np.sum(W) - 1


    constraint_set = [{'type': 'eq', 'fun': constraint}]
    #minimize negative SharpeRatio
    result = sc.minimize(negative_sharpe_ratio,
                        W,
                        args=(mean_returns, cov, risk_free_rate,n),
                        method='SLSQP',
                        bounds= bounds,
                        constraints=constraint_set)
    neg_sharpe_ratio, optimal_weights = result['fun'], result['x'].round(4)
    return -neg_sharpe_ratio, optimal_weights

def minimum_risk_portfolio(mean_returns, cov, upper_bound, risk_free_rate,n):
    """
    returns
    -------
    sharpe_ratio, optimal_weights"""
     #assign random weights
    np.random.seed(1)
    W = np.random.random(len(mean_returns))
    W = [weight/ np.sum(W) for weight in W]

    #add bounds
    bound = (0,upper_bound)
    bounds = tuple(bound for w in range(len(W)))

    #constraint 
    def constraint(W):
        return np.sum(W) - 1
    constraint_set = [{'type': 'eq', 'fun': constraint}]
    
    def portfolio_variance(W,cov,n):
        return (np.sqrt(W.T @ cov @ W) * np.sqrt(n))

    result = sc.minimize(portfolio_variance,
                        W,
                        args = (cov,n),
                        bounds = bounds,
                        constraints = constraint_set,
                        method = 'SLSQP')

    min_risk, optimal_weights = result['fun'], result['x'].round(4)
    return min_risk, optimal_weights


def main(type, close, n, risk_free_rate:float,  upper_bound:float):
    mean_returns, cov = stock_performance(close)

    #maximum Sharpe Ratio portfolio
    if type == 'Sharpe Ratio':
        metric, optimal_weights = optimize_portfolio(mean_returns, cov, upper_bound, risk_free_rate,n)
    else:
        metric, optimal_weights = minimum_risk_portfolio(mean_returns, cov, upper_bound, risk_free_rate,n)
                               
    portfolio_returns, portfolio_risk = portfolio_performance(optimal_weights, mean_returns, cov,n)
    st.write(f'Expected return: {portfolio_returns:.2%}, Risk: {portfolio_risk:.2%} with {type}:{metric.round(3)}\n')

    df = pd.DataFrame({"ticker":close.columns.to_list(), "weight": optimal_weights})
    best_weights = df.loc[df['weight']>0,:].reset_index(drop=True)

    return best_weights




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

upper_bound = st.number_input(label='Diversification Threshold:',
                              value = 0.1,
                              key='upper_bound')
upper_bound = st.session_state.upper_bound


risk_free_rate = st.number_input(label='Risk Free Rate:',
                              value = 0.2,
                              key='risk_free_rate')
risk_free_rate = st.session_state.risk_free_rate

holding_period = st.number_input(label='Holding Period:',
                              value = 252,
                              key='Holding Period:')

type = st.selectbox(label='Optimization Type',
                   options=['Sharpe Ratio','Minimum Risk'])

benchmark = st.file_uploader(label='Upload CSV Of Desired Benchmark (Optional)',
                 type='csv',
                 key='benchmark')
benchmark = st.session_state.benchmark
###########
#############

if close:
    close = pd.read_csv(close, index_col=0, header=0)
    portfolio_weights = main(type, close,holding_period,
                             risk_free_rate=risk_free_rate,
                             upper_bound=upper_bound)
    cols = st.columns([0.3,0.7])
    with cols[0]:
        portfolio_weights = st.data_editor(portfolio_weights, disabled=["ticker"])        
        if portfolio_weights.weight.sum().round(2) < 1.0:
            st.warning(
                '''Warning: Portfolio allocation must sum to 1.0.
                Relax diversification threshold, use Pie Chart proportionate allocation for Unused Cash, or modify manually from table.
                **Current Unused Cash {:.2f}**'''.format(1-portfolio_weights.weight.sum())
                      )
        
        elif portfolio_weights.weight.sum().round(2) > 1.0:
            st.warning(
                '''Warning: Portfolio allocation must sum to 1.0. Use Pie Chart proportionate allocation, or modify manually from the table.
                **Current Leverage {:.2f}**'''.format(portfolio_weights.weight.sum())
                      )
            
    with cols[1]:
        fig = px.pie(portfolio_weights, values='weight', names='ticker', title='Portfolio Weights')
        st.plotly_chart(fig)

    portfolio = close.loc[:,portfolio_weights.ticker] @ portfolio_weights.weight.values.reshape((-1,1))
    portfolio.columns=["Portfolio"]
    #portfolio = portfolio.iloc[-70:,:]

    if benchmark:
        benchmark = pd.read_csv(benchmark, index_col=0, header=0)
        #benchmark = benchmark.iloc[-70:,:]
        portfolio["Benchmark"] = benchmark
        comparison = portfolio/portfolio.values[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=comparison.index, y=comparison["Portfolio"], name="Portfolio")
        )
        fig.add_trace(
            go.Scatter(x=comparison.index, y=comparison["Benchmark"], name="Benchmark")
        )
        st.plotly_chart(fig)
    else:
        comparison=portfolio

        
    st.download_button(
        label="Download Report",
        data=to_excel(comparison, portfolio_weights),
        file_name="Comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
