import numpy as np
from scipy.stats import truncnorm, norm
import pandas as pd
from monte_carlo_portfolio import AssetClass, get_truncated_normal

class PortfolioAnalyzer:
    def __init__(self):
        self.asset_classes = []
        self.correlation_matrix = None
        self.tax_rates = {
            'taxable': {'income': 0.35, 'capital_gains': 0.20},
            'traditional': {'income': 0.35, 'capital_gains': 0.00},
            'roth': {'income': 0.00, 'capital_gains': 0.00}
        }
        self.account_types = {
            'taxable': 0.33,
            'traditional': 0.33,
            'roth': 0.34
        }
        
    def set_account_allocation(self, taxable=0.33, traditional=0.33, roth=0.34):
        """Set the allocation between different account types"""
        if abs(taxable + traditional + roth - 1.0) > 0.0001:
            raise ValueError("Account allocations must sum to 1.0")
            
        self.account_types = {
            'taxable': taxable,
            'traditional': traditional,
            'roth': roth
        }
    
    def set_correlation_matrix(self, correlation_matrix=None):
        """Set correlation matrix between asset classes"""
        if correlation_matrix is None:
            # Default correlation matrix (identity = no correlation)
            n_assets = len(self.asset_classes)
            self.correlation_matrix = np.identity(n_assets)
        else:
            # Verify correlation matrix size
            if correlation_matrix.shape != (len(self.asset_classes), len(self.asset_classes)):
                raise ValueError("Correlation matrix size must match number of asset classes")
            
            # Verify correlation matrix is symmetric
            if not np.allclose(correlation_matrix, correlation_matrix.T):
                raise ValueError("Correlation matrix must be symmetric")
            
            # Verify correlation matrix has ones on diagonal
            if not np.allclose(np.diag(correlation_matrix), np.ones(len(self.asset_classes))):
                raise ValueError("Correlation matrix must have ones on diagonal")
            
            self.correlation_matrix = correlation_matrix
    
    def generate_correlated_returns(self, num_simulations, years):
        """Generate correlated returns for all asset classes"""
        n_assets = len(self.asset_classes)
        
        # Generate uncorrelated standard normal random variables
        uncorrelated = np.random.standard_normal((n_assets, num_simulations, years))
        
        # Compute Cholesky decomposition
        L = np.linalg.cholesky(self.correlation_matrix)
        
        # Transform uncorrelated variables to correlated variables
        # Reshape for matrix multiplication
        reshaped = uncorrelated.reshape(n_assets, -1)
        correlated = L @ reshaped
        correlated = correlated.reshape(n_assets, num_simulations, years)
        
        # Convert to truncated normal returns for each asset class
        asset_returns = {}
        for i, asset in enumerate(self.asset_classes):
            # Convert correlated standard normals to uniform [0,1]
            uniform = norm.cdf(correlated[i])
            
            # Convert uniform to truncated normal with asset's parameters
            a = (asset.min_return - asset.mean_return) / asset.std_dev
            b = (asset.max_return - asset.mean_return) / asset.std_dev
            
            returns = truncnorm.ppf(
                uniform, 
                a, b, 
                loc=asset.mean_return, 
                scale=asset.std_dev
            )
            
            # Store returns with asset name
            asset_returns[asset.name] = returns
            
        return asset_returns
    
    def calculate_rebalanced_portfolio(self, initial_investment, years, num_simulations, 
                                       initial_allocation, rebalance_frequency=1, 
                                       rebalance_threshold=0.05):
        """Calculate portfolio with periodic rebalancing"""
        # Initialize portfolio value
        portfolio_values = np.zeros((num_simulations, years + 1))
        portfolio_values[:, 0] = initial_investment
        
        # Generate returns for each asset class
        asset_returns = self.generate_correlated_returns(num_simulations, years)
        
        # Convert initial allocation to numpy array
        target_allocation = np.array([asset.allocation for asset in self.asset_classes])
        
        # Initialize asset values
        asset_values = np.zeros((num_simulations, len(self.asset_classes), years + 1))
        for i, asset in enumerate(self.asset_classes):
            asset_values[:, i, 0] = initial_investment * asset.allocation
        
        for year in range(years):
            # Apply returns for each asset
            for i, asset in enumerate(self.asset_classes):
                asset_values[:, i, year + 1] = asset_values[:, i, year] * (1 + asset_returns[asset.name][:, year])
            
            # Calculate new portfolio value and allocation
            portfolio_values[:, year + 1] = np.sum(asset_values[:, :, year + 1], axis=1)
            
            # Calculate drift from target allocation
            current_allocation = asset_values[:, :, year + 1] / portfolio_values[:, year + 1].reshape(-1, 1)
            drift = np.abs(current_allocation - target_allocation)
            
            # Rebalance if needed
            if (year + 1) % rebalance_frequency == 0:
                # Determine which simulations need rebalancing (if any exceed threshold)
                needs_rebalance = np.any(drift > rebalance_threshold, axis=1)
                
                # Rebalance those that need it
                for sim in range(num_simulations):
                    if needs_rebalance[sim]:
                        for i, asset in enumerate(self.asset_classes):
                            asset_values[sim, i, year + 1] = portfolio_values[sim, year + 1] * target_allocation[i]
        
        return portfolio_values, asset_values, asset_returns

    def apply_withdrawal_strategy(self, portfolio_values, withdrawal_strategy, initial_withdrawal,
                                 inflation_rate=0.03, min_withdrawal=None, max_withdrawal=None):
        """
        Apply different withdrawal strategies to the portfolio
        
        Parameters:
        - portfolio_values: Portfolio values over time (num_simulations x years)
        - withdrawal_strategy: Type of withdrawal strategy ('fixed', 'percent', 'dynamic')
        - initial_withdrawal: Initial annual withdrawal amount or percentage
        - inflation_rate: Annual inflation rate
        - min_withdrawal: Minimum withdrawal amount (for dynamic strategies)
        - max_withdrawal: Maximum withdrawal amount (for dynamic strategies)
        
        Returns:
        - updated_portfolio: Updated portfolio values after withdrawals
        - withdrawals: Annual withdrawal amounts
        """
        num_simulations, years = portfolio_values.shape
        years -= 1  # Adjust for initial year
        
        # Initialize withdrawals and updated portfolio
        withdrawals = np.zeros((num_simulations, years))
        updated_portfolio = portfolio_values.copy()
        
        # Apply different withdrawal strategies
        if withdrawal_strategy == 'fixed':
            # Fixed inflation-adjusted withdrawal
            base_withdrawals = np.array([initial_withdrawal * (1 + inflation_rate)**year 
                                       for year in range(years)])
            for sim in range(num_simulations):
                withdrawals[sim, :] = base_withdrawals
                
        elif withdrawal_strategy == 'percent':
            # Percentage-based withdrawal (e.g., 4% rule)
            for sim in range(num_simulations):
                for year in range(years):
                    withdrawals[sim, year] = updated_portfolio[sim, year] * initial_withdrawal
                    
        elif withdrawal_strategy == 'dynamic':
            # Dynamic withdrawal based on portfolio performance
            base_withdrawals = np.array([initial_withdrawal * (1 + inflation_rate)**year 
                                       for year in range(years)])
            for sim in range(num_simulations):
                for year in range(years):
                    # Calculate previous period return (except for first year)
                    if year > 0:
                        prev_return = (updated_portfolio[sim, year] / 
                                      (updated_portfolio[sim, year-1] - withdrawals[sim, year-1])) - 1
                        
                        # Adjust withdrawal based on return
                        if prev_return > 0.05:  # Good year
                            withdrawals[sim, year] = min(
                                withdrawals[sim, year-1] * (1 + inflation_rate) * 1.1,
                                base_withdrawals[year] * 1.5
                            )
                        elif prev_return < -0.05:  # Bad year
                            withdrawals[sim, year] = max(
                                withdrawals[sim, year-1] * (1 + inflation_rate) * 0.9,
                                base_withdrawals[year] * 0.75
                            )
                        else:  # Normal year
                            withdrawals[sim, year] = withdrawals[sim, year-1] * (1 + inflation_rate)
                    else:
                        withdrawals[sim, year] = base_withdrawals[year]
                    
                    # Apply min/max constraints if provided
                    if min_withdrawal is not None:
                        withdrawals[sim, year] = max(withdrawals[sim, year], min_withdrawal)
                    if max_withdrawal is not None:
                        withdrawals[sim, year] = min(withdrawals[sim, year], max_withdrawal)
        
        # Apply withdrawals to portfolio
        for sim in range(num_simulations):
            for year in range(years):
                updated_portfolio[sim, year+1] -= withdrawals[sim, year]
                updated_portfolio[sim, year+1] = max(updated_portfolio[sim, year+1], 0)  # No negative values
        
        return updated_portfolio, withdrawals

    def calculate_risk_metrics(self, portfolio_values, withdrawals=None):
        """Calculate advanced risk metrics for the portfolio"""
        num_simulations, years = portfolio_values.shape
        
        # Calculate returns for each period
        returns = np.zeros((num_simulations, years-1))
        for year in range(1, years):
            # If withdrawals provided, include them in return calculation
            if withdrawals is not None:
                # Ensure no division by zero
                prev_values = portfolio_values[:, year-1].copy()
                prev_values[prev_values == 0] = 1e-10  # Small value to avoid division by zero
                returns[:, year-1] = (portfolio_values[:, year] + withdrawals[:, year-1]) / prev_values - 1
            else:
                # Ensure no division by zero
                prev_values = portfolio_values[:, year-1].copy()
                prev_values[prev_values == 0] = 1e-10  # Small value to avoid division by zero
                returns[:, year-1] = portfolio_values[:, year] / prev_values - 1
        
        # Calculate metrics
        metrics = {}
        
        # Calculate mean return and standard deviation
        mean_return = np.mean(returns, axis=1)
        std_dev = np.std(returns, axis=1)
        
        # Ensure no division by zero for Sharpe ratio
        std_dev_safe = std_dev.copy()
        std_dev_safe[std_dev_safe == 0] = 1e-10  # Small value to avoid division by zero
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev_safe
        metrics['sharpe_ratio'] = {
            'mean': np.mean(sharpe_ratio),
            'median': np.median(sharpe_ratio),
            'p10': np.percentile(sharpe_ratio, 10),
            'p90': np.percentile(sharpe_ratio, 90)
        }
        
        # Calculate maximum drawdown
        max_drawdown = np.zeros(num_simulations)
        for sim in range(num_simulations):
            peak = max(portfolio_values[sim, 0], 1e-10)  # Ensure non-zero peak
            drawdown = 0
            for year in range(1, years):
                if portfolio_values[sim, year] > peak:
                    peak = portfolio_values[sim, year]
                else:
                    # Ensure non-zero peak for division
                    if peak > 0:
                        current_drawdown = (peak - portfolio_values[sim, year]) / peak
                        if current_drawdown > drawdown:
                            drawdown = current_drawdown
            max_drawdown[sim] = drawdown
        
        metrics['max_drawdown'] = {
            'mean': np.mean(max_drawdown),
            'median': np.median(max_drawdown),
            'p10': np.percentile(max_drawdown, 10),
            'p90': np.percentile(max_drawdown, 90)
        }
        
        # Calculate Sortino Ratio (downside deviation)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_deviation = np.std(downside_returns, axis=1)
        # Replace zeros with small value to avoid division by zero
        downside_deviation[downside_deviation == 0] = 1e-10
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        
        metrics['sortino_ratio'] = {
            'mean': np.mean(sortino_ratio),
            'median': np.median(sortino_ratio),
            'p10': np.percentile(sortino_ratio, 10),
            'p90': np.percentile(sortino_ratio, 90)
        }
        
        # Calculate conditional value at risk (CVaR)
        # (average of worst 5% outcomes)
        cvar_cutoff = 0.05
        final_values = portfolio_values[:, -1]
        sorted_values = np.sort(final_values)
        cvar_index = int(num_simulations * cvar_cutoff)
        cvar = np.mean(sorted_values[:cvar_index])
        metrics['cvar'] = cvar
        
        # Sequence of returns risk (effect of bad returns in early years)
        early_bad_sims = []
        for sim in range(num_simulations):
            # Check if first 5 years have negative average return
            if np.mean(returns[sim, :5]) < 0:
                early_bad_sims.append(sim)
        
        if early_bad_sims:
            early_bad_final = portfolio_values[early_bad_sims, -1]
            metrics['sequence_risk'] = {
                'probability': len(early_bad_sims) / num_simulations,
                'median_final_value': np.median(early_bad_final),
                'impact': (np.median(final_values) - np.median(early_bad_final)) / np.median(final_values)
            }
        else:
            metrics['sequence_risk'] = {
                'probability': 0,
                'median_final_value': 0,
                'impact': 0
            }
        
        return metrics

    def calculate_tax_impact(self, portfolio_values, asset_values, asset_returns, withdrawal_amount):
        """Calculate tax impact on portfolio over time"""
        num_simulations, years = portfolio_values.shape
        years -= 1  # Adjust for initial year
        
        # Initialize tax amounts and after-tax portfolio
        tax_paid = np.zeros((num_simulations, years))
        after_tax_portfolio = portfolio_values.copy()
        
        for sim in range(num_simulations):
            for year in range(years):
                # Calculate proportion of withdrawal from each account type
                taxable_withdrawal = withdrawal_amount * self.account_types['taxable']
                traditional_withdrawal = withdrawal_amount * self.account_types['traditional']
                roth_withdrawal = withdrawal_amount * self.account_types['roth']
                
                # Calculate tax on income (traditional accounts)
                income_tax = traditional_withdrawal * self.tax_rates['traditional']['income']
                
                # Calculate capital gains tax on taxable accounts
                # Assume half of taxable withdrawal is basis, half is gains
                taxable_gains = taxable_withdrawal * 0.5
                capital_gains_tax = taxable_gains * self.tax_rates['taxable']['capital_gains']
                
                # Total tax
                tax_paid[sim, year] = income_tax + capital_gains_tax
                
                # Reduce portfolio by tax paid
                after_tax_portfolio[sim, year+1] -= tax_paid[sim, year]
        
        return after_tax_portfolio, tax_paid

def create_default_asset_classes():
    """Create default asset classes"""
    asset_classes = [
        AssetClass("Stocks", 0.10, 0.16, -0.25, 0.25, 0.30),          # 30% stocks
        AssetClass("Bonds/MMF", 0.04, 0.04, -0.01, 0.05, 0.20),       # 20% bonds/money market
        AssetClass("Alternatives", 0.11, 0.12, -0.15, 0.30, 0.25),    # 25% alternatives
        AssetClass("Private Placements", 0.14, 0.20, -0.30, 0.30, 0.20), # 20% private placements
        AssetClass("Cash", 0.02, 0.01, 0.01, 0.04, 0.05)              # 5% cash
    ]
    return asset_classes

def create_default_correlation_matrix():
    """Create default correlation matrix for the five asset classes"""
    # Default correlation matrix (reasonable values)
    corr_matrix = np.array([
        [1.00, 0.30, 0.50, 0.60, 0.10],  # Stocks
        [0.30, 1.00, 0.20, 0.15, 0.40],  # Bonds/MMF
        [0.50, 0.20, 1.00, 0.70, 0.05],  # Alternatives
        [0.60, 0.15, 0.70, 1.00, 0.00],  # Private Placements
        [0.10, 0.40, 0.05, 0.00, 1.00]   # Cash
    ])
    return corr_matrix

# Example historical data - annualized returns for major asset classes
# Format: year, stocks, bonds, alternatives, private, cash
HISTORICAL_DATA = [
    [2000, -0.091, 0.112, 0.049, 0.212, 0.058],
    [2001, -0.119, 0.084, 0.043, 0.092, 0.032],
    [2002, -0.220, 0.102, 0.031, 0.023, 0.017],
    [2003, 0.287, 0.041, 0.152, 0.272, 0.010],
    [2004, 0.109, 0.043, 0.089, 0.195, 0.013],
    [2005, 0.049, 0.028, 0.097, 0.267, 0.030],
    [2006, 0.157, 0.042, 0.125, 0.282, 0.047],
    [2007, 0.055, 0.070, 0.100, 0.303, 0.046],
    [2008, -0.370, 0.051, -0.198, -0.252, 0.015],
    [2009, 0.265, 0.058, 0.112, 0.162, 0.001],
    [2010, 0.150, 0.065, 0.106, 0.199, 0.001],
    [2011, 0.021, 0.078, -0.052, 0.117, 0.001],
    [2012, 0.160, 0.042, 0.063, 0.142, 0.001],
    [2013, 0.323, -0.020, 0.089, 0.162, 0.001],
    [2014, 0.135, 0.060, 0.031, 0.171, 0.001],
    [2015, 0.013, 0.001, -0.018, 0.108, 0.002],
    [2016, 0.119, 0.026, 0.042, 0.122, 0.005],
    [2017, 0.218, 0.035, 0.088, 0.152, 0.010],
    [2018, -0.043, 0.001, -0.034, 0.102, 0.020],
    [2019, 0.315, 0.086, 0.107, 0.142, 0.022],
    [2020, 0.183, 0.076, 0.101, 0.095, 0.004],
    [2021, 0.286, -0.015, 0.152, 0.273, 0.001],
    [2022, -0.186, -0.130, -0.074, -0.011, 0.016],
    [2023, 0.264, 0.059, 0.087, 0.152, 0.052]
]

def get_historical_returns():
    """Return historical returns dataframe"""
    return pd.DataFrame(
        HISTORICAL_DATA, 
        columns=['year', 'stocks', 'bonds', 'alternatives', 'private', 'cash']
    )

def run_historical_backtest(initial_investment, years_to_simulate, initial_withdrawal, asset_classes, withdrawal_strategy='fixed'):
    """
    Run a historical backtest using actual historical returns
    Uses sequential returns from historical data with rolling windows
    """
    # Get historical data
    historical_df = get_historical_returns()
    available_years = len(historical_df)
    
    if years_to_simulate > available_years:
        print(f"Warning: Requested {years_to_simulate} years but only {available_years} years of data available")
        print(f"Will use repeated data to fulfill request")
    
    # Number of possible starting years
    start_indices = available_years - min(years_to_simulate, available_years) + 1
    
    # Initialize results
    results = np.zeros((start_indices, years_to_simulate + 1))
    results[:, 0] = initial_investment
    
    # Set up withdrawal strategy
    if withdrawal_strategy == 'fixed':
        withdrawals = np.array([initial_withdrawal * (1.03)**year 
                              for year in range(years_to_simulate)])
    else:
        # For other strategies, initialize with placeholder
        withdrawals = np.zeros((start_indices, years_to_simulate))
    
    # Run backtest for each possible starting year
    for start_idx in range(start_indices):
        portfolio_value = initial_investment
        
        for year in range(years_to_simulate):
            # Get index in historical data (with wraparound if needed)
            hist_idx = (start_idx + year) % available_years
            
            # Get returns for this year
            year_returns = {
                'Stocks': historical_df.iloc[hist_idx]['stocks'],
                'Bonds/MMF': historical_df.iloc[hist_idx]['bonds'],
                'Alternatives': historical_df.iloc[hist_idx]['alternatives'],
                'Private Placements': historical_df.iloc[hist_idx]['private'],
                'Cash': historical_df.iloc[hist_idx]['cash']
            }
            
            # Calculate portfolio return
            portfolio_return = sum(asset.allocation * year_returns[asset.name] 
                                  for asset in asset_classes)
            
            # Apply return
            portfolio_value *= (1 + portfolio_return)
            
            # Apply withdrawal
            if withdrawal_strategy == 'percent':
                # Percentage-based withdrawal
                withdrawals[start_idx, year] = portfolio_value * initial_withdrawal
            elif withdrawal_strategy == 'dynamic':
                # Dynamic withdrawal (simplified)
                if year == 0:
                    withdrawals[start_idx, year] = initial_withdrawal
                else:
                    if portfolio_return > 0.05:  # Good year
                        withdrawals[start_idx, year] = withdrawals[start_idx, year-1] * 1.1
                    elif portfolio_return < -0.05:  # Bad year
                        withdrawals[start_idx, year] = withdrawals[start_idx, year-1] * 0.9
                    else:  # Normal year
                        withdrawals[start_idx, year] = withdrawals[start_idx, year-1] * 1.03
            
            # Apply withdrawal
            if withdrawal_strategy == 'fixed':
                portfolio_value -= withdrawals[year]
            else:
                portfolio_value -= withdrawals[start_idx, year]
            
            # No negative values
            portfolio_value = max(portfolio_value, 0)
            
            # Store result
            results[start_idx, year + 1] = portfolio_value
    
    # Return results and withdrawals
    if withdrawal_strategy == 'fixed':
        return results, withdrawals
    else:
        return results, withdrawals