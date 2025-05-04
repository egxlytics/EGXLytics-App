import pandas as pd
import numpy as np
from cvxpy import *
import riskfolio as rp
import riskfolio.src.RiskFunctions as rk


def cointegration_tracking(y,X, upper_bound):
    """Track a specific Index-a Benchmark.

    Args:
        y (pd.Dataframe): Benchmark
        X (pd.Dataframe): Prices Dataframe
        upper_bound (float): Maximum allocation for a stock

    Returns:
        tuple: A Tuple containing:
            portfolio_allocation (pd.DataFrame): DataFrame with tickers and their optimal weights.
            portfolio (pd.DataFrame): Portfolio Series
    """
    if y.shape[0] != X.shape[0]:
        raise ValueError(f'y and X do not have the same shape. {y.shape[0]}, {X.shape[0]}')

    x = Variable((X.shape[1],1), pos=True)
    constraints = [sum(x)<=1, x <= upper_bound]
    objective = Minimize(sum_squares(np.matrix(X) @ x -np.matrix(y).reshape((-1,1))))
    prob = Problem(objective, constraints)
    prob.solve()

    portfolio_weights  = pd.DataFrame({'Weight': x.value.reshape((-1,)).tolist()}, index=X.columns)
    portfolio_weights.index.name='Ticker' 
    return portfolio_weights


class Portfolio():
    def __init__(self, prices_df: pd.DataFrame):
        self.prices= prices_df
        self.tickers = prices_df.columns.to_list()
        self.returns = prices_df.pct_change().dropna()
        self.cov = prices_df.cov()
        self.cor = prices_df.corr()
        self.mean_vector = prices_df.pct_change().dropna().mean()

    

    def optimize(self,
                ticker_sector_df: pd.DataFrame,
                risk_measure: str,
                objective_function: str,
                sector_constraint: float,
                equity_constraint: float,
                **kwargs):
        """Optimize portfolio weights based on the selected objective function (Sharpe or MinRisk).
        
        Args:
            ticker_sector_df (df): Dataframe (2, n) where n is number of Ticker, and with column names ['Ticker','Sector']
            risk_measure (str): ['MV', 'CVaR'] - Minimum Variance or Conditional Value at Risk 
            objective_function (str): ['Sharpe','MinRisk','MaxRet']
            sector_constraint (float): Maximum Allocation For Sector (0 to 1).
            equity_constraint (float): Maximum Allocation For Stock (0 to 1).
            holding_period (int): Holding Period of the Portoflio
            risk_free_rate (float): Risk Free Rate
            **kwargs

        Returns:
            tuple: A tuple containing:

            portfolio_allocation (pd.DataFrame): DataFrame with tickers and their optimal weights.
            portfolio (pd.DataFrame): Portfolio Series
    """

        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("`returns_df` must be a pandas DataFrame.")

        if not isinstance(ticker_sector_df, pd.DataFrame):
            raise ValueError("`ticker_sector_df` must be a pandas DataFrame.")

        required_cols = {'Ticker', 'Sector'}
        if not required_cols.issubset(ticker_sector_df.columns):
            raise ValueError(f"`ticker_sector_df` must contain columns: {required_cols}")

        if risk_measure not in ['MV', 'CVaR']:
            raise ValueError("`risk_measure` must be one of: 'MV', 'CVaR'.")

        if objective_function not in ['Sharpe', 'MinRisk', 'MaxRet']:
            raise ValueError("`objective_function` must be one of: 'Sharpe', 'MinRisk', 'MaxRet'.")

        if not (0 <= sector_constraint <= 1):
            raise ValueError("`sector_constraint` must be between 0 and 1.")

        if not (0 <= equity_constraint <= 1):
            raise ValueError("`equity_constraint` must be between 0 and 1.")
        
        
        self.ticker_sector_df = ticker_sector_df
        self.equity_constraint = equity_constraint
        self.sector_constraint = sector_constraint

        port = rp.Portfolio(returns=self.returns)

        self.ticker_sector_df = self.ticker_sector_df.sort_values(by='Ticker')
        self.sectors = ticker_sector_df.Sector.unique()
        n_sectors = len(self.sectors)

        constraints = {
            "Disabled": [False] + [False]*n_sectors,
            "Type": ["All Assets"] + ["Classes"]*n_sectors,
            "Set": [""] + ['Sector']*n_sectors,
            "Position": [""] + list(self.sectors),
            "Sign": ["<="] + ["<="] * n_sectors,
            "Weight": [equity_constraint] + [sector_constraint]*n_sectors,
            "Type Relative": [""] + [""]*n_sectors,
            "Relative Set":  [""] + [""]*n_sectors,
            "Relative": [""] + [""]*n_sectors,
            "Factor":  [""] + [""]*n_sectors
        }

        constraints = pd.DataFrame(constraints)

        A, B = rp.assets_constraints(constraints, ticker_sector_df)

        port.ainequality = A
        port.binequality = B

        port.assets_stats(method_mu="hist", method_cov='hist')



        if kwargs:
            port.kindbench = False
            port.allowTE = True
            port.TE = kwargs.get('tracking_error')
            port.benchindex = kwargs.get('benchmark')


        model = "Classic"
        rm = risk_measure
        obj = objective_function

        self.weights = port.optimization(model=model, rm=rm, obj=obj)
        self.weights.columns = ['Weight']

        self.portfolio_returns = self.returns[self.weights.index] @ self.weights.values.reshape((-1,1))

        self.portfolio_allocation = pd.merge(left=self.weights, right=ticker_sector_df,
                                             left_on=self.weights.index, right_on='Ticker')[["Ticker", "Sector", "Weight"]].set_index("Ticker", drop=True)
        self.portfolio = self.prices[self.portfolio_allocation.index.to_list()] @ self.weights.values.reshape((-1,1))
        self.portfolio.dropna(inplace=True)
        self.portfolio.columns = ['Portfolio']

        return self.portfolio_allocation, self.portfolio
    
    def index_tracking(self,
                       benchmark:pd.DataFrame,
                       equity_constraint: float):
        
        self.equity_constraint = equity_constraint
        self.benchmark = benchmark
        benchmark_returns = benchmark.pct_change().dropna()

        self.benchmark_returns = benchmark_returns.reindex(self.returns.index)

        portfolio_allocation = cointegration_tracking(self.benchmark_returns,self.returns,self.equity_constraint)
        self.weights = portfolio_allocation.Weight
        self.portfolio_returns = self.returns.loc[:,portfolio_allocation.index] @ self.weights.values.reshape((-1,1))
        

        portfolio = self.prices.loc[:,portfolio_allocation.index.to_list()] @ self.weights.values.reshape((-1,1))
        portfolio.dropna(inplace=True)
        portfolio.columns = ['portfolio']
        self.portfolio = portfolio

        self.tracking_error = mean(self.benchmark_returns-self.portfolio_returns)

        return portfolio_allocation, portfolio
    
    
    def portfolio_performance(self,portfolio_returns:pd.Series, n:int,risk_free_rate:float):
        """Portfolio performance: Computes the Expected portfolio returns, standard deviation, CVaR, and Sharpe Ratio .

        Args:
            portfolio_returns (pd.Series or pd.Dataframe): Portfolio Returns.
            n (int): Holding period.
            risk_free_rate (float): Risk Free Rate.

        Returns:
            tuple: tuple: A tuple containing:
            portfolio_expected_returns (float): Expected returns
            portfolio_risk (float): Expected Risk 
            cvar (float): CVaR at 5%
            sharpe (float): Sharpe Ratio
        """

        self.risk_free_rate = float(risk_free_rate)
        self.portfolio_expected_returns = float(portfolio_returns.mean() * n)
        self.portfolio_risk = float(portfolio_returns.std() * np.sqrt(n))


        self.cvar = float(rk.CVaR_Hist(portfolio_returns, 0.05))
        self.sharpe = float((self.portfolio_expected_returns-self.risk_free_rate)/self.portfolio_risk)



        return self.portfolio_expected_returns, self.portfolio_risk,self.cvar,self.sharpe
