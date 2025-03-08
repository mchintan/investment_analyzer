# Investment Portfolio Analyzer

A comprehensive Monte Carlo simulation tool for investment portfolio analysis. This application helps investors, financial advisors, and wealth managers make data-driven decisions about portfolio allocation, sustainable withdrawal rates, and long-term financial planning.

## Features

- **Portfolio Simulation**: Run Monte Carlo simulations to project investment outcomes over time
- **Asset Allocation**: Customize allocation across 5 asset classes (Stocks, Bonds/MMF, Alternatives, Private Placements, Cash)
- **Flexible Withdrawal Strategies**:
  - Fixed (inflation-adjusted)
  - Percentage-based (e.g., 4% rule)
  - Dynamic (market-responsive)
- **Portfolio Rebalancing**: Set rebalancing frequency and drift tolerance
- **Advanced Risk Metrics**:
  - Sharpe and Sortino ratios
  - Maximum drawdown analysis
  - Sequence of returns risk
  - Conditional Value at Risk (CVaR)
- **Historical Backtesting**: Test strategies against actual historical returns
- **Correlation Analysis**: Visualize relationships between different asset classes
- **Allocation Optimization**: Explore how different allocations affect risk and return
- **Investor Profiles**: Save and load different investor profiles
- **Data Export**: Download simulation results for further analysis

## Installation

```bash
git clone https://github.com/yourusername/investment_analyzer.git
cd investment_analyzer
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

This will launch the web application in your default browser.

## Required Dependencies

- Python 3.9+
- Streamlit
- NumPy
- SciPy
- Pandas
- Plotly
- Matplotlib

## Configuration

Adjust the simulation parameters using the sidebar controls:

1. **Investment Settings**:
   - Initial investment amount
   - Initial withdrawal amount/rate
   - Simulation time horizon
   - Number of simulations
   - Inflation rate

2. **Asset Allocation**:
   - Percentage allocation to each asset class
   - Expected return, standard deviation, min/max returns for each asset class

3. **Advanced Settings**:
   - Portfolio rebalancing frequency
   - Withdrawal strategy
   - Correlation between asset classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Based on Monte Carlo simulation techniques commonly used in financial planning
- Historical data is representative and should be updated for serious investment decisions
- Not financial advice - consult a qualified professional for personalized guidance
