import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

# Use a different style for more vibrant colors
plt.style.use('default')  # Reset to default style for more control over colors

class AssetClass:
    def __init__(self, name, mean_return, std_dev, min_return, max_return, allocation):
        self.name = name
        self.mean_return = mean_return
        self.std_dev = std_dev
        self.min_return = min_return
        self.max_return = max_return
        self.allocation = allocation

def get_truncated_normal(mean, sd, low, upp, size):
    """Generate truncated normal distribution"""
    a = (low - mean) / sd
    b = (upp - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def run_simulation(initial_investment=6000000, years=30, num_simulations=5000, 
                  initial_withdrawal=300000, inflation_rate=0.03, asset_classes=None):
    """
    Run Monte Carlo simulation for a diversified portfolio
    """
    # Use default asset classes if none provided
    if asset_classes is None:
        asset_classes = [
            AssetClass("Stocks", 0.10, 0.16, -0.25, 0.25, 0.30),          # 30% stocks (higher volatility)
            AssetClass("Bonds/MMF", 0.04, 0.04, -0.01, 0.05, 0.20),       # 20% bonds/money market
            AssetClass("Alternatives", 0.11, 0.12, -0.15, 0.30, 0.25),    # 25% alternatives (hedge funds, commodities)
            AssetClass("Private Placements", 0.14, 0.20, -0.30, 0.30, 0.20), # 20% private placements (highest risk/return)
            AssetClass("Cash", 0.02, 0.01, 0.01, 0.04, 0.05)              # 5% cash
        ]
    
    results = np.zeros((num_simulations, years + 1))
    results[:, 0] = initial_investment
    
    # Calculate inflation-adjusted withdrawals
    withdrawals = np.array([initial_withdrawal * (1 + inflation_rate)**year 
                           for year in range(years)])
    
    # Generate returns for each asset class
    asset_returns = {}
    for asset in asset_classes:
        asset_returns[asset.name] = get_truncated_normal(
            asset.mean_return,
            asset.std_dev,
            asset.min_return,
            asset.max_return,
            (num_simulations, years)
        )
    
    # Calculate portfolio returns
    portfolio_returns = np.zeros((num_simulations, years))
    for asset in asset_classes:
        portfolio_returns += asset_returns[asset.name] * asset.allocation
    
    # Calculate cumulative portfolio value
    for i in range(years):
        results[:, i + 1] = results[:, i] * (1 + portfolio_returns[:, i])
        results[:, i + 1] -= withdrawals[i]
        results[:, i + 1] = np.maximum(results[:, i + 1], 0)
    
    return results, portfolio_returns, withdrawals, asset_returns, asset_classes

def plot_simulations(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_withdrawal, initial_investment=7000000):
    """
    Added initial_investment parameter with default value matching the simulation
    """
    years = np.arange(results.shape[1])
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(15, 20), facecolor='#f0f0f0')
    
    # Portfolio value plot
    ax1 = plt.subplot(4, 1, 1)
    ax1.set_facecolor('#f0f0f0')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i in range(len(results)):
        ax1.plot(years, results[i], alpha=0.05, color=colors[i])
    
    median = np.median(results, axis=0)
    percentile_10 = np.percentile(results, 10, axis=0)
    percentile_90 = np.percentile(results, 90, axis=0)
    
    final_values = results[:, -1]
    # Proper risk of depletion: percentage of simulations that hit zero at any point
    depleted_simulations = np.any(results <= 0, axis=1)
    risk_of_depletion = np.mean(depleted_simulations) * 100
    
    ax1.plot(years, median, color='#FF1E1E', linewidth=3.5, label='Median')
    ax1.plot(years, percentile_10, color='#4CAF50', linewidth=3.5, label='10th Percentile')
    ax1.plot(years, percentile_90, color='#2196F3', linewidth=3.5, label='90th Percentile')
    
    ax1.grid(True, alpha=0.2, color='white', linewidth=2)
    ax1.set_title('Diversified Portfolio Monte Carlo Simulation\n40-Year Projection with Inflation-Adjusted Withdrawals', 
                  fontsize=14, pad=20, color='#2C3E50', fontweight='bold')
    ax1.set_xlabel('Years', fontsize=12, color='#2C3E50', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='#2C3E50', fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    legend = ax1.legend(loc='upper right', fontsize=10, 
                       facecolor='white', edgecolor='#2C3E50', 
                       framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_linewidth(2)
    
    # Add asset allocation and statistics text
    allocation_text = "Asset Allocation:\n" + "\n".join(
        f"{asset.name}: {asset.allocation:.0%} (Return: {asset.mean_return:.1%} ± {asset.std_dev:.1%})"
        for asset in asset_classes
    )
    
    avg_annual_return = np.mean([(final_val/initial_investment)**(1/40) - 1 
                                for final_val in final_values if final_val > 0]) * 100
    
    stats_text = (
        f'Final Portfolio Statistics:\n'
        f'Median: ${np.median(final_values):,.0f}\n'
        f'90th Percentile: ${np.percentile(final_values, 90):,.0f}\n'
        f'10th Percentile: ${np.percentile(final_values, 10):,.0f}\n'
        f'Risk of Depletion: {risk_of_depletion:.1f}%\n'
        f'Initial Withdrawal: ${initial_withdrawal:,}\n'
        f'Final Withdrawal: ${withdrawals[-1]:,.0f}\n'
        f'Avg Annual Return: {avg_annual_return:.1f}%\n\n'
        f'{allocation_text}'
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2C3E50',
                      boxstyle='round,pad=1', linewidth=2),
             verticalalignment='top', fontsize=10, color='#2C3E50',
             fontweight='bold')
    
    # Asset class return distributions
    ax2 = plt.subplot(4, 1, 2)
    ax2.set_facecolor('#f0f0f0')
    
    colors = ['#FF1E1E',    # Red for Stocks
              '#4CAF50',    # Green for Bonds/MMF
              '#2196F3',    # Blue for Alternatives
              '#FFC107',    # Yellow for Private Placements
              '#9C27B0']    # Purple for Cash
              
    for i, asset in enumerate(asset_classes):
        plt.hist(asset_returns[asset.name].flatten(), bins=50, alpha=0.5,
                label=asset.name, color=colors[i])
    
    ax2.grid(True, alpha=0.2, color='white', linewidth=2)
    ax2.set_title('Asset Class Return Distributions', 
                  fontsize=14, color='#2C3E50', fontweight='bold')
    ax2.set_xlabel('Return Rate', fontsize=12, color='#2C3E50', fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, color='#2C3E50', fontweight='bold')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax2.legend()
    
    # Portfolio return distribution
    ax3 = plt.subplot(4, 1, 3)
    ax3.set_facecolor('#f0f0f0')
    
    ax3.hist(portfolio_returns.flatten(), bins=50, color='#2196F3', alpha=0.7,
             edgecolor='white')
    ax3.axvline(x=np.mean([asset.mean_return * asset.allocation for asset in asset_classes]), 
                color='#FF1E1E', linewidth=2, linestyle='--',
                label='Expected Portfolio Return')
    ax3.grid(True, alpha=0.2, color='white', linewidth=2)
    ax3.set_title('Portfolio Return Distribution', 
                  fontsize=14, color='#2C3E50', fontweight='bold')
    ax3.set_xlabel('Return Rate', fontsize=12, color='#2C3E50', fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, color='#2C3E50', fontweight='bold')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax3.legend()
    
    # Withdrawal progression
    ax4 = plt.subplot(4, 1, 4)
    ax4.set_facecolor('#f0f0f0')
    
    ax4.plot(years[1:], withdrawals, color='#FF1E1E', linewidth=3.5)
    ax4.grid(True, alpha=0.2, color='white', linewidth=2)
    ax4.set_title('Annual Withdrawal Amount (4% Inflation Adjusted)', 
                  fontsize=14, color='#2C3E50', fontweight='bold')
    ax4.set_xlabel('Years', fontsize=12, color='#2C3E50', fontweight='bold')
    ax4.set_ylabel('Withdrawal Amount ($)', fontsize=12, color='#2C3E50', fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color('#2C3E50')
            spine.set_linewidth(2)
        ax.tick_params(colors='#2C3E50', which='both', width=2, labelsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
def run_convergence_analysis(initial_investment, years, initial_withdrawal, inflation_rate, asset_classes):
    """Run multiple simulations with increasing sample sizes to show convergence"""
    sample_sizes = [1000, 3000, 5000, 7000, 10000]
    convergence_results = []
    
    # Run simulations for each sample size
    for n_sims in sample_sizes:
        results, portfolio_returns, withdrawals, _, _ = run_simulation(
            initial_investment=initial_investment,
            years=years,
            num_simulations=n_sims,
            initial_withdrawal=initial_withdrawal,
            inflation_rate=inflation_rate,
            asset_classes=asset_classes
        )
        
        final_values = results[:, -1]
        # Check for simulations that hit zero at any point (true risk of depletion)
        depleted_simulations = np.any(results <= 0, axis=1)
        
        # Calculate all metrics with protection against edge cases
        try:
            median_value = np.median(final_values)
        except:
            median_value = 0
            
        try:
            p10 = np.percentile(final_values, 10)
        except:
            p10 = 0
            
        try:
            p90 = np.percentile(final_values, 90)
        except:
            p90 = 0
            
        try:
            depletion_risk = np.mean(depleted_simulations) * 100
        except:
            depletion_risk = 0
        
        convergence_results.append({
            'n_sims': n_sims,
            'median': median_value,
            'percentile_10': p10,
            'percentile_90': p90,
            'risk_of_depletion': depletion_risk
        })
    
    return convergence_results

# Modify the main block to only run if directly executed
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run simulation
    initial_withdrawal = 300000
    initial_investment = 6000000
    results, portfolio_returns, withdrawals, asset_returns, asset_classes = run_simulation(
        initial_withdrawal=initial_withdrawal,
        initial_investment=initial_investment
    )
    
    # Plot results
    plot_simulations(
        results, 
        portfolio_returns, 
        withdrawals, 
        asset_returns, 
        asset_classes, 
        initial_withdrawal,
        initial_investment
    )
    plt.show()  # Only show when running directly 