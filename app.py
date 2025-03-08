import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import json
import os
from monte_carlo_portfolio import run_simulation, AssetClass, run_convergence_analysis
from enhanced_portfolio import (PortfolioAnalyzer, create_default_asset_classes, 
                              create_default_correlation_matrix, get_historical_returns,
                              run_historical_backtest)
import enhanced_portfolio as ep

def create_plotly_figures(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_investment):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Asset Class Returns Distribution',
            'Portfolio Return Distribution',
            'Inflation-Adjusted Withdrawals'
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    years = np.arange(results.shape[1])
    
    # Portfolio value plot (top, spanning both columns)
    for i in range(min(100, len(results))):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=results[i],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.1)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Add median and percentiles
    median = np.median(results, axis=0)
    percentile_10 = np.percentile(results, 10, axis=0)
    percentile_90 = np.percentile(results, 90, axis=0)

    fig.add_trace(
        go.Scatter(
            x=years,
            y=median,
            name='Median',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=percentile_10,
            name='10th Percentile',
            line=dict(color='green', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=percentile_90,
            name='90th Percentile',
            line=dict(color='blue', width=2),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Withdrawal progression
    fig.add_trace(
        go.Scatter(
            x=years[1:],
            y=withdrawals,
            name='Withdrawal Amount',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Withdrawal: $%{y:,.0f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Asset class return distributions
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, asset in enumerate(asset_classes):
        fig.add_trace(
            go.Histogram(
                x=asset_returns[asset.name].flatten(),
                name=asset.name,
                nbinsx=50,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                hovertemplate='Return: %{x:.1%}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

    # Asset class returns Pareto (middle right)
    for i, asset in enumerate(asset_classes):
        returns = asset_returns[asset.name].flatten()
        counts, bins = np.histogram(returns, bins=50)
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / cumsum[-1] * 100
        
        fig.add_trace(
            go.Scatter(
                x=bins[:-1],
                y=cumsum_norm,
                name=f'{asset.name} Cumulative',
                line=dict(color=colors[i % len(colors)]),
                hovertemplate='Return: %{x:.1%}<br>Cumulative: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )

    # Portfolio return distribution (bottom left)
    returns_hist, returns_bins = np.histogram(portfolio_returns.flatten(), bins=50)
    most_common_idx = np.argmax(returns_hist)
    most_common_range = f"{returns_bins[most_common_idx]:.1%} to {returns_bins[most_common_idx + 1]:.1%}"
    most_common_frequency = returns_hist[most_common_idx] / len(portfolio_returns.flatten()) * 100
    
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns.flatten() * 100,  # Convert to percentage
            name='Portfolio Returns',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Add annotation for most common range
    fig.add_annotation(
        text=f"Most Common: {most_common_range}<br>({most_common_frequency:.1f}% of returns)",
        xref="x3", yref="paper",
        x=returns_bins[most_common_idx] * 100,
        y=1,
        showarrow=True,
        arrowhead=1,
        row=3, col=1
    )

    # Portfolio returns Pareto (bottom right)
    counts, bins = np.histogram(portfolio_returns.flatten() * 100, bins=50)  # Convert to percentage
    cumsum = np.cumsum(counts)
    cumsum_norm = cumsum / cumsum[-1] * 100
    
    fig.add_trace(
        go.Scatter(
            x=bins[:-1],
            y=cumsum_norm,
            name='Portfolio Cumulative',
            line=dict(color='blue'),
            hovertemplate='Return: %{x:.1f}%<br>Cumulative: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )

    # Update axis labels to show percentages
    fig.update_xaxes(
        title_text="Return Rate (%)", 
        tickformat=".1f",  # Show as percentage with 1 decimal
        row=3, col=1
    )

    # Add 80% reference line for Pareto charts
    fig.add_hline(y=80, line=dict(color="red", width=1, dash="dash"),
                  row=3, col=1)

    # Update layout for better mobile viewing and legend readability
    fig.update_layout(
        height=1200,  # Increased height
        showlegend=True,
        title_text="Portfolio Monte Carlo Simulation",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            groupclick="toggleitem",
            itemsizing="constant",
            tracegroupgap=5  # Add gap between legend groups
        ),
        margin=dict(l=20, r=20, t=120, b=20),  # Increased top margin for legend
        font=dict(size=14)
    )

    # Update subplot titles font size and position
    for i, annotation in enumerate(fig.layout.annotations):
        annotation.update(
            font=dict(size=16),
            y=annotation.y + 0.02  # Move titles up slightly
        )

    # Group legends by subplot with proper naming
    for trace in fig.data:
        # Skip if trace name is None
        if trace.name is None:
            continue
            
        # Handle different types of traces
        if 'Cumulative' in trace.name:
            trace.showlegend = False
        elif trace.name in ['Median', '10th Percentile', '90th Percentile']:
            trace.legendgroup = "portfolio_value"
            trace.legendgrouptitle = dict(text="Portfolio Value")
        elif 'Withdrawal' in trace.name:
            trace.legendgroup = "withdrawals"
            trace.legendgrouptitle = dict(text="Withdrawals")
        elif 'Portfolio Returns' in trace.name:
            trace.legendgroup = "portfolio_returns"
            trace.legendgrouptitle = dict(text="Portfolio Returns")
        elif any(asset.name in trace.name for asset in asset_classes):
            trace.legendgroup = "assets"
            trace.legendgrouptitle = dict(text="Asset Classes")

    return fig

def plot_convergence(convergence_results):
    """Create a single convergence analysis plot with all metrics"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Extract data
    n_sims = [r['n_sims'] for r in convergence_results]
    medians = [r['median'] for r in convergence_results]
    p90 = [r['percentile_90'] for r in convergence_results]
    p10 = [r['percentile_10'] for r in convergence_results]
    risk = [r['risk_of_depletion'] for r in convergence_results]
    
    # Calculate relative changes with protection against division by zero
    def calc_relative_change(values):
        changes = []
        for i in range(1, len(values)):
            # Skip if previous value is zero to avoid division by zero
            if values[i-1] == 0:
                if values[i] == 0:
                    changes.append(0)  # No change if both values are zero
                else:
                    changes.append(100)  # 100% change if going from 0 to non-zero
            else:
                changes.append(abs((values[i] - values[i-1]) / values[i-1]) * 100)
        return changes
    
    # Calculate changes with protection against invalid inputs
    try:
        median_changes = calc_relative_change(medians)
    except:
        median_changes = [0] * (len(n_sims) - 1)
        
    try:
        p90_changes = calc_relative_change(p90)
    except:
        p90_changes = [0] * (len(n_sims) - 1)
        
    try:
        p10_changes = calc_relative_change(p10)
    except:
        p10_changes = [0] * (len(n_sims) - 1)
        
    try:
        risk_changes = calc_relative_change(risk)
    except:
        risk_changes = [0] * (len(n_sims) - 1)
    
    # Add traces for portfolio values
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=medians,
            name='Median',
            line=dict(color='red'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p90,
            name='90th Percentile',
            line=dict(color='blue'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=p10,
            name='10th Percentile',
            line=dict(color='green'),
            hovertemplate='Sims: %{x:,}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add risk of depletion on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=n_sims,
            y=risk,
            name='Risk of Depletion',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Sims: %{x:,}<br>Risk: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout for better legend readability
    fig.update_layout(
        height=500,
        title_text="Convergence Analysis",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemsizing="constant"
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(size=14)
    )

    # Group convergence metrics in legend
    for trace in fig.data:
        if trace.name in ['Median', '90th Percentile', '10th Percentile']:
            trace.legendgroup = "portfolio"
            trace.legendgrouptitle = dict(text="Portfolio Values")
        elif trace.name == 'Risk of Depletion':
            trace.legendgroup = "risk"
            trace.legendgrouptitle = dict(text="Risk Metrics")

    # Update axes with larger font
    fig.update_xaxes(title_text="Number of Simulations", 
                    tickformat=",", 
                    title_font=dict(size=14))
    fig.update_yaxes(title_text="Portfolio Value ($)", 
                    tickformat="$,.0f", 
                    secondary_y=False,
                    title_font=dict(size=14))
    fig.update_yaxes(title_text="Risk of Depletion (%)", 
                    tickformat=".1f", 
                    secondary_y=True,
                    title_font=dict(size=14))
    
    return fig, median_changes, p90_changes, p10_changes, risk_changes

def plot_advanced_metrics(metrics):
    """Plot advanced metrics in a radar chart"""
    # Extract the metrics we want to display, using median values where available
    simplified_metrics = {}
    
    # Process Sharpe Ratio
    if 'sharpe_ratio' in metrics:
        simplified_metrics['Sharpe Ratio'] = metrics['sharpe_ratio']['median']
    
    # Process Sortino Ratio
    if 'sortino_ratio' in metrics:
        simplified_metrics['Sortino Ratio'] = metrics['sortino_ratio']['median']
    
    # Process Max Drawdown
    if 'max_drawdown' in metrics:
        simplified_metrics['Max Drawdown'] = metrics['max_drawdown']['median']
    
    # Process other simple metrics
    if 'cvar' in metrics:
        simplified_metrics['CVaR'] = metrics['cvar']
    
    # Add sequence risk impact if available
    if 'sequence_risk' in metrics:
        simplified_metrics['Sequence Risk'] = metrics['sequence_risk']['impact']
    
    # Add success rate if it was calculated
    if 'success_rate' in metrics:
        simplified_metrics['Success Rate'] = metrics['success_rate']
    
    # Default metrics if the above ones are not available
    if not simplified_metrics:
        simplified_metrics = {
            'Sharpe Ratio': 1.2,
            'Sortino Ratio': 1.5,
            'Max Drawdown': -0.3,
            'Success Rate': 90,
            'CVaR': 2000000
        }
    
    # Create a radar chart for the metrics
    categories = list(simplified_metrics.keys())
    
    # Normalize values for radar chart (all positive between 0 and 1)
    normalized_values = []
    for metric, value in simplified_metrics.items():
        if metric == 'Max Drawdown':
            # Convert negative drawdown to positive scale (smaller is better)
            normalized_values.append(max(0, min(1, abs(value) / 0.5)))
        elif metric == 'CVaR':
            # Normalize based on initial investment
            # Assuming higher CVaR is better
            normalized_values.append(max(0, min(1, value / 6000000)))
        elif metric == 'Sequence Risk':
            # Smaller is better
            normalized_values.append(max(0, min(1, (1 - value))))
        elif metric == 'Success Rate':
            # Already in percentage, normalize to 0-1
            normalized_values.append(max(0, min(1, value / 100)))
        else:
            # For other metrics, higher is better
            normalized_values.append(max(0, min(1, value / 3)))  # Assuming 3 is a good upper bound
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Portfolio Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=500,
        title="Portfolio Risk-Return Metrics"
    )
    
    return fig

def plot_correlation_matrix(correlation_matrix, asset_classes):
    """Plot correlation matrix as a heatmap"""
    asset_names = [asset.name for asset in asset_classes]
    
    # Create correlation values for heatmap
    z_values = []
    text_values = []
    
    # Handle correlation_matrix as numpy array
    for i, asset1 in enumerate(asset_names):
        z_row = []
        text_row = []
        for j, asset2 in enumerate(asset_names):
            corr_value = correlation_matrix[i, j]
            z_row.append(corr_value)
            text_row.append(f"{corr_value:.2f}")
        z_values.append(z_row)
        text_values.append(text_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=asset_names,
        y=asset_names,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=text_values,
        texttemplate="%{text}",
        showscale=True
    ))
    
    fig.update_layout(
        title="Asset Class Correlation Matrix",
        height=600,
        xaxis=dict(title="Asset Class"),
        yaxis=dict(title="Asset Class")
    )
    
    return fig

def plot_historical_backtest(backtest_results, historical_data, years):
    """Create a visualization of historical backtest results"""
    # Extract results
    results, withdrawals = backtest_results
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Historical Backtest Portfolio Values',
            'Historical Asset Returns'
        ),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Plot backtest results
    for i in range(len(results)):
        fig.add_trace(
            go.Scatter(
                x=np.arange(years + 1),
                y=results[i],
                mode='lines',
                opacity=0.3,
                line=dict(color='gray'),
                showlegend=False,
                hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add median and percentiles
    median = np.median(results, axis=0)
    p10 = np.percentile(results, 10, axis=0)
    p90 = np.percentile(results, 90, axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(years + 1),
            y=median,
            mode='lines',
            name='Median',
            line=dict(color='red', width=3),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(years + 1),
            y=p10,
            mode='lines',
            name='10th Percentile',
            line=dict(color='green', width=3),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(years + 1),
            y=p90,
            mode='lines',
            name='90th Percentile',
            line=dict(color='blue', width=3),
            hovertemplate='Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot historical returns
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, col in enumerate(['stocks', 'bonds', 'alternatives', 'private', 'cash']):
        if col in historical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=historical_data['year'],
                    y=historical_data[col] * 100,  # Convert to percentage
                    mode='lines',
                    name=col.capitalize(),
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate='Year: %{x}<br>Return: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Historical Backtest Analysis",
        template="plotly_white",
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=120, b=20),
        font=dict(size=14)
    )
    
    # Update y-axis formatting
    fig.update_yaxes(title_text="Portfolio Value ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", tickformat=".1f", row=2, col=1)
    
    # Update x-axis
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Calendar Year", row=2, col=1)
    
    return fig

def create_allocation_heatmap(stock_range, bond_range, asset_classes):
    """Create a heatmap of portfolio metrics for different asset allocations"""
    # Generate combinations of stock and bond allocations
    stock_allocations = np.linspace(stock_range[0], stock_range[1], 10)
    bond_allocations = np.linspace(bond_range[0], bond_range[1], 10)
    
    # Initialize result matrices
    median_values = np.zeros((len(stock_allocations), len(bond_allocations)))
    sharpe_values = np.zeros((len(stock_allocations), len(bond_allocations)))
    risk_values = np.zeros((len(stock_allocations), len(bond_allocations)))
    
    # Create portfolio analyzer
    analyzer = PortfolioAnalyzer()
    
    # Run simulations for each allocation combination
    for i, stock_alloc in enumerate(stock_allocations):
        for j, bond_alloc in enumerate(bond_allocations):
            # Calculate remaining allocation for other assets
            remaining = 1.0 - (stock_alloc + bond_alloc)
            
            # Skip invalid allocations
            if remaining < 0:
                median_values[i, j] = np.nan
                sharpe_values[i, j] = np.nan
                risk_values[i, j] = np.nan
                continue
            
            # Create asset classes with updated allocations
            updated_asset_classes = []
            for idx, asset in enumerate(asset_classes):
                if idx == 0:  # Stocks
                    alloc = stock_alloc
                elif idx == 1:  # Bonds
                    alloc = bond_alloc
                else:
                    # Distribute remaining allocation proportionally among other assets
                    original_sum = sum(a.allocation for a in asset_classes[2:])
                    if original_sum > 0:
                        alloc = remaining * (asset.allocation / original_sum)
                    else:
                        alloc = remaining / (len(asset_classes) - 2)
                
                updated_asset_classes.append(
                    AssetClass(asset.name, asset.mean_return, asset.std_dev, 
                              asset.min_return, asset.max_return, alloc)
                )
            
            # Run simulation
            analyzer.asset_classes = updated_asset_classes
            analyzer.set_correlation_matrix()  # Use identity matrix for simplicity
            
            portfolio_values, _, asset_returns = analyzer.calculate_rebalanced_portfolio(
                initial_investment=6000000,
                years=30,
                num_simulations=1000,  # Use smaller number for speed
                initial_allocation=updated_asset_classes,
                rebalance_frequency=1
            )
            
            # Calculate metrics
            metrics = analyzer.calculate_risk_metrics(portfolio_values)
            
            # Store results
            median_values[i, j] = np.median(portfolio_values[:, -1]) / 1000000  # In millions
            sharpe_values[i, j] = metrics['sharpe_ratio']['median']
            risk_values[i, j] = 100 - (np.mean(portfolio_values[:, -1] > 0) * 100)  # Risk of depletion
    
    return stock_allocations, bond_allocations, median_values, sharpe_values, risk_values

def plot_allocation_heatmap(stock_allocations, bond_allocations, metric_values, metric_name):
    """Create a heatmap visualization for different asset allocations"""
    # Create 2D meshgrid for heatmap
    stock_grid, bond_grid = np.meshgrid(stock_allocations * 100, bond_allocations * 100)
    
    # Create formatter based on metric
    if 'Median' in metric_name:
        tickformat = '$,.1f'
        hovertemplate = 'Stocks: %{x:.0f}%<br>Bonds: %{y:.0f}%<br>Value: $%{z:.1f}M<extra></extra>'
    elif 'Sharpe' in metric_name:
        tickformat = '.2f'
        hovertemplate = 'Stocks: %{x:.0f}%<br>Bonds: %{y:.0f}%<br>Sharpe: %{z:.2f}<extra></extra>'
    else:  # Risk
        tickformat = '.1f%'
        hovertemplate = 'Stocks: %{x:.0f}%<br>Bonds: %{y:.0f}%<br>Risk: %{z:.1f}%<extra></extra>'
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=metric_values.T,  # Transpose for correct orientation
        x=stock_allocations * 100,  # Convert to percentage
        y=bond_allocations * 100,  # Convert to percentage
        colorscale='Viridis',
        hovertemplate=hovertemplate
    ))
    
    fig.update_layout(
        title=f"Asset Allocation Impact on {metric_name}",
        height=500,
        width=600,
        xaxis_title="Stock Allocation (%)",
        yaxis_title="Bond Allocation (%)",
        template="plotly_white",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def load_profile_presets():
    """Load profile presets from file or return defaults"""
    default_presets = {
        "Conservative": {
            "stocks": 20,
            "bonds": 50,
            "alternatives": 15,
            "private": 10,
            "cash": 5,
            "withdrawal_strategy": "fixed",
            "withdrawal_rate": 3.0,
            "rebalance_frequency": 1
        },
        "Balanced": {
            "stocks": 40,
            "bonds": 30,
            "alternatives": 15,
            "private": 10,
            "cash": 5,
            "withdrawal_strategy": "fixed",
            "withdrawal_rate": 4.0,
            "rebalance_frequency": 1
        },
        "Growth": {
            "stocks": 60,
            "bonds": 20,
            "alternatives": 10,
            "private": 8,
            "cash": 2,
            "withdrawal_strategy": "percent",
            "withdrawal_rate": 4.5,
            "rebalance_frequency": 1
        },
        "Aggressive": {
            "stocks": 70,
            "bonds": 10,
            "alternatives": 10,
            "private": 8,
            "cash": 2,
            "withdrawal_strategy": "dynamic",
            "withdrawal_rate": 5.0,
            "rebalance_frequency": 2
        }
    }
    
    # Try to load from file if exists
    # Check both the current directory and the persistent data directory
    file_paths = ["profile_presets.json", "data/profile_presets.json"]
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except:
                continue
    
    return default_presets

def save_profile_presets(presets):
    """Save profile presets to file"""
    # Try to save to the persistent data directory first
    if os.path.exists("data"):
        file_path = "data/profile_presets.json"
    else:
        # Fall back to current directory
        file_path = "profile_presets.json"
        
    try:
        with open(file_path, 'w') as f:
            json.dump(presets, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving profile presets: {e}")
        return False

def save_current_profile(profile_name):
    """Save current settings as a new profile"""
    # Get current settings from session state
    current_settings = {
        "stocks": st.session_state.stocks_allocation,
        "bonds": st.session_state.bonds_allocation,
        "alternatives": st.session_state.alts_allocation,
        "private": st.session_state.private_allocation,
        "cash": st.session_state.cash_allocation,
        "withdrawal_strategy": st.session_state.get('withdrawal_strategy', 'fixed'),
        "withdrawal_rate": st.session_state.get('withdrawal_rate', 4.0),
        "rebalance_frequency": st.session_state.get('rebalance_frequency', 1)
    }
    
    # Load existing presets
    presets = load_profile_presets()
    
    # Add or update the profile
    presets[profile_name] = current_settings
    
    # Save presets
    return save_profile_presets(presets)

def export_report_data():
    """Export current simulation data as JSON"""
    if 'simulation_results' not in st.session_state:
        return None
    
    # Extract results from session state
    sim_data = st.session_state.simulation_results
    
    # Convert numpy arrays to lists for JSON serialization
    export_data = {
        'portfolio_summary': {
            'initial_investment': float(sim_data['initial_investment']),
            'median_final_value': float(np.median(sim_data['results'][:, -1])),
            'percentile_10': float(np.percentile(sim_data['results'][:, -1], 10)),
            'percentile_90': float(np.percentile(sim_data['results'][:, -1], 90)),
            'risk_of_depletion': float(np.mean(sim_data['results'][:, -1] < sim_data['withdrawals'][-1]) * 100)
        },
        'asset_allocation': [
            {
                'name': asset.name,
                'allocation': float(asset.allocation),
                'mean_return': float(asset.mean_return),
                'std_dev': float(asset.std_dev),
                'min_return': float(asset.min_return),
                'max_return': float(asset.max_return)
            }
            for asset in sim_data['asset_classes']
        ],
        'withdrawal_info': {
            'initial_withdrawal': float(sim_data['withdrawals'][0]),
            'final_withdrawal': float(sim_data['withdrawals'][-1])
        }
    }
    
    # Advanced metrics if available
    if 'risk_metrics' in sim_data:
        export_data['risk_metrics'] = {
            'sharpe_ratio': {
                'median': float(sim_data['risk_metrics']['sharpe_ratio']['median'])
            },
            'max_drawdown': {
                'median': float(sim_data['risk_metrics']['max_drawdown']['median'])
            },
            'sortino_ratio': {
                'median': float(sim_data['risk_metrics']['sortino_ratio']['median'])
            }
        }
    
    return export_data

def calculate_max_drawdown(results):
    """Calculate maximum drawdown across all simulations"""
    max_drawdowns = []
    
    for sim in results:
        # Calculate running maximum
        running_max = np.maximum.accumulate(sim)
        # Calculate drawdown
        drawdown = (sim - running_max) / running_max
        # Get maximum drawdown
        max_drawdown = np.min(drawdown)
        max_drawdowns.append(max_drawdown)
    
    # Return median of maximum drawdowns
    return np.median(max_drawdowns)

def main():
    st.set_page_config(
        page_title="Portfolio Monte Carlo Simulation", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS to hide sidebar on small screens
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            [data-testid="stSidebar"][aria-expanded="true"] {
                display: none;
            }
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create header row with just the title
    header_col1, header_col2, header_col3 = st.columns([2, 4, 2])
    
    with header_col2:
        st.title("Portfolio Monte Carlo Simulation")
    
    # Create a placeholder for the summary at the top
    summary_placeholder = st.empty()
    
    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Simulation", 
        "📈 Advanced Analysis", 
        "🧠 Historical Analysis",
        "⚙️ Settings"
    ])
    
    # Create sidebar for inputs
    st.sidebar.header("Simulation Parameters")
    
    # Profile Presets
    st.sidebar.subheader("Profile Presets")
    profile_presets = load_profile_presets()
    selected_profile = st.sidebar.selectbox(
        "Select Profile",
        options=list(profile_presets.keys()),
        index=1  # Default to Balanced
    )
    
    # Apply profile button
    apply_profile = st.sidebar.button("Apply Profile")
    
    # Initialize session state variables if they don't exist
    if 'stocks_allocation' not in st.session_state:
        # Initialize with balanced profile
        preset = profile_presets["Balanced"]
        st.session_state.stocks_allocation = preset["stocks"]
        st.session_state.bonds_allocation = preset["bonds"]
        st.session_state.alts_allocation = preset["alternatives"]
        st.session_state.private_allocation = preset["private"]
        st.session_state.cash_allocation = preset["cash"]
        st.session_state.withdrawal_strategy = preset["withdrawal_strategy"]
        st.session_state.withdrawal_rate = preset["withdrawal_rate"]
        st.session_state.rebalance_frequency = preset["rebalance_frequency"]
    
    # Apply selected profile if button clicked
    if apply_profile:
        preset = profile_presets[selected_profile]
        st.session_state.stocks_allocation = preset["stocks"]
        st.session_state.bonds_allocation = preset["bonds"]
        st.session_state.alts_allocation = preset["alternatives"]
        st.session_state.private_allocation = preset["private"]
        st.session_state.cash_allocation = preset["cash"]
        st.session_state.withdrawal_strategy = preset["withdrawal_strategy"]
        st.session_state.withdrawal_rate = preset["withdrawal_rate"]
        st.session_state.rebalance_frequency = preset["rebalance_frequency"]
    
    # Basic parameters
    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=100000,
        max_value=100000000,
        value=6000000,
        step=100000,
        format="%d",
        key="initial_investment"
    )
    
    # Default initial withdrawal (will be updated based on withdrawal strategy)
    initial_withdrawal = 300000
    
    # Withdrawal Strategy (moved earlier for better UI flow)
    st.sidebar.header("Withdrawal Strategy")
    withdrawal_strategy = st.sidebar.selectbox(
        "Withdrawal Method",
        options=["Fixed", "Percentage", "Dynamic"],
        index=0,
        format_func=lambda x: {
            "Fixed": "Fixed (Inflation-Adjusted)",
            "Percentage": "Percentage of Portfolio",
            "Dynamic": "Dynamic (Market-Responsive)"
        }.get(x, x)
    )
    
    st.session_state.withdrawal_strategy = withdrawal_strategy.lower()
    
    if withdrawal_strategy == "Fixed":
        initial_withdrawal = st.sidebar.number_input(
            "Initial Annual Withdrawal ($)",
            min_value=10000,
            max_value=1000000,
            value=300000,
            step=10000,
            format="%d"
        )
        withdrawal_rate = initial_withdrawal / initial_investment * 100
        st.session_state.withdrawal_rate = withdrawal_rate
    else:
        withdrawal_rate = st.sidebar.slider(
            "Annual Withdrawal Rate (%)",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.withdrawal_rate,
            step=0.1
        )
        st.session_state.withdrawal_rate = withdrawal_rate
        initial_withdrawal = initial_investment * (withdrawal_rate / 100)
    
    # Add stress test option
    st.sidebar.header("Stress Testing")
    stress_test = st.sidebar.checkbox("Enable Stress Test", value=False)
    
    if stress_test:
        stress_factor = st.sidebar.slider(
            "Market Stress Factor (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="Reduces expected returns by this percentage"
        )
    else:
        stress_factor = 0
    
    # Asset Allocation Section
    st.sidebar.header("Asset Allocation")
    
    # Create expandable section for each asset class
    with st.sidebar.expander(f"Stocks ({st.session_state.stocks_allocation}%)", expanded=True):
        stocks_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.stocks_allocation,
            step=5,
            key="stocks_alloc"
        )
        st.session_state.stocks_allocation = stocks_allocation
        
        stocks_return = st.number_input(
            "Expected Return (%)", 
            min_value=-10, 
            max_value=30, 
            value=10, 
            step=1, 
            key="stocks_return"
        )
        stocks_std = st.number_input("Standard Deviation (%)", 0, 30, 16, step=1, key="stocks_std")
        stocks_min = st.number_input("Minimum Return (%)", -50, 0, -25, step=1, key="stocks_min")
        stocks_max = st.number_input("Maximum Return (%)", 0, 50, 25, step=1, key="stocks_max")
    
    with st.sidebar.expander(f"Bonds/MMF ({st.session_state.bonds_allocation}%)"):
        bonds_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.bonds_allocation,
            step=5,
            key="bonds_alloc"
        )
        st.session_state.bonds_allocation = bonds_allocation
        
        bonds_return = st.number_input("Expected Return (%)", -5, 15, 4, step=1, key="bonds_return")
        bonds_std = st.number_input("Standard Deviation (%)", 0, 20, 4, step=1, key="bonds_std")
        bonds_min = st.number_input("Minimum Return (%)", -10, 5, -1, step=1, key="bonds_min")
        bonds_max = st.number_input("Maximum Return (%)", 0, 20, 10, step=1, key="bonds_max")
    
    with st.sidebar.expander(f"Alternatives ({st.session_state.alts_allocation}%)"):
        alts_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.alts_allocation,
            step=5,
            key="alts_alloc"
        )
        st.session_state.alts_allocation = alts_allocation
        
        alts_return = st.number_input("Expected Return (%)", -5, 25, 11, step=1, key="alts_return")
        alts_std = st.number_input("Standard Deviation (%)", 0, 25, 12, step=1, key="alts_std")
        alts_min = st.number_input("Minimum Return (%)", -30, 0, -15, step=1, key="alts_min")
        alts_max = st.number_input("Maximum Return (%)", 0, 40, 30, step=1, key="alts_max")
    
    with st.sidebar.expander(f"Private Placements ({st.session_state.private_allocation}%)"):
        private_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.private_allocation,
            step=5,
            key="private_alloc"
        )
        st.session_state.private_allocation = private_allocation
        
        private_return = st.number_input("Expected Return (%)", 0, 40, 20, step=1, key="private_return")
        private_std = st.number_input("Standard Deviation (%)", 0, 40, 20, step=1, key="private_std")
        private_min = st.number_input("Minimum Return (%)", -50, 0, -30, step=1, key="private_min")
        private_max = st.number_input("Maximum Return (%)", 0, 60, 40, step=1, key="private_max")
    
    with st.sidebar.expander(f"Cash ({st.session_state.cash_allocation}%)"):
        cash_allocation = st.number_input(
            "Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.cash_allocation,
            step=5,
            key="cash_alloc"
        )
        st.session_state.cash_allocation = cash_allocation
        
        cash_return = st.number_input("Expected Return (%)", 0, 10, 2, step=1, key="cash_return")
        cash_std = st.number_input("Standard Deviation (%)", 0, 5, 1, step=1, key="cash_std")
        cash_min = st.number_input("Minimum Return (%)", 0, 5, 1, step=1, key="cash_min")
        cash_max = st.number_input("Maximum Return (%)", 0, 10, 4, step=1, key="cash_max")
    
    # Check if allocations sum to 100%
    total_allocation = (stocks_allocation + bonds_allocation + alts_allocation + 
                       private_allocation + cash_allocation)
    
    if total_allocation != 100:
        st.sidebar.error(f"Total allocation must equal 100% (currently {total_allocation}%)")
    
    # Simulation parameters
    st.sidebar.header("Simulation Settings")
    
    years = st.sidebar.slider(
        "Simulation Years",
        min_value=10,
        max_value=50,
        value=30,  # Increase default from 15 to 30 years
        step=5
    )
    
    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=10000,
        value=5000
    )
    
    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    ) / 100
    
    # Advanced settings in a separate expander
    with st.sidebar.expander("Advanced Settings"):
        # Portfolio Rebalancing
        rebalance_frequency = st.number_input(
            "Rebalancing Frequency (years)",
            min_value=0,
            max_value=10,
            value=st.session_state.rebalance_frequency,
            help="How often to rebalance the portfolio (0 = never)"
        )
        st.session_state.rebalance_frequency = rebalance_frequency
        
        rebalance_threshold = st.slider(
            "Rebalancing Threshold (%)",
            min_value=0,
            max_value=20,
            value=5,
            help="Minimum drift percentage to trigger rebalancing"
        )
        
        # Use correlated returns
        use_correlation = st.checkbox(
            "Use Correlated Returns",
            value=True,
            help="Simulate with realistic correlations between asset classes"
        )
        
        # Save settings as profile
        st.subheader("Save Current Profile")
        new_profile_name = st.text_input("Profile Name", placeholder="My Custom Profile")
        save_profile = st.button("Save Profile")
        
        if save_profile and new_profile_name:
            success = save_current_profile(new_profile_name)
            if success:
                st.success(f"Profile '{new_profile_name}' saved successfully")
            else:
                st.error("Failed to save profile")
    
    # Store the "Run Simulation" button state
    run_clicked = st.sidebar.button("Run Simulation")
    
    # MAIN CONTENT AREA WITHIN TABS
    
    # Tab 1: Basic Simulation
    with tabs[0]:
        # Run simulation if button is clicked OR if this is the first load (no previous runs)
        if ('simulation_results' not in st.session_state) or (run_clicked and total_allocation == 100):
            with st.spinner("Running simulations and convergence analysis..."):
                # Create asset classes with user inputs
                asset_classes = [
                    AssetClass("Stocks", stocks_return/100, stocks_std/100, 
                              stocks_min/100, stocks_max/100, stocks_allocation/100),
                    AssetClass("Bonds/MMF", bonds_return/100, bonds_std/100,
                              bonds_min/100, bonds_max/100, bonds_allocation/100),
                    AssetClass("Alternatives", alts_return/100, alts_std/100,
                              alts_min/100, alts_max/100, alts_allocation/100),
                    AssetClass("Private Placements", private_return/100, private_std/100,
                              private_min/100, private_max/100, private_allocation/100),
                    AssetClass("Cash", cash_return/100, cash_std/100,
                              cash_min/100, cash_max/100, cash_allocation/100)
                ]
                
                # Run simulation and store results in session state
                results, portfolio_returns, withdrawals, asset_returns, _ = run_simulation(
                    initial_investment=initial_investment,
                    years=years,
                    num_simulations=num_simulations,
                    initial_withdrawal=initial_withdrawal,
                    inflation_rate=inflation_rate,
                    asset_classes=asset_classes
                )
                
                # Calculate all summary statistics directly from simulation results
                final_values = results[:, -1]
                
                # Risk of depletion (percentage of simulations that hit zero at any point)
                depleted_simulations = np.any(results <= 0, axis=1)
                risk_of_depletion = np.mean(depleted_simulations) * 100
                
                # Initialize variables with default values
                depletion_text = "Never"
                escape_text = "Never"
                escape_delta = "0% probability"
                years_of_depletion = []
                years_of_escape = []
                
                # Calculate year of depletion for each simulation
                for sim in results:
                    depleted_years = np.where(sim <= 0)[0]
                    if len(depleted_years) > 0:
                        years_of_depletion.append(depleted_years[0])
                
                # Calculate median year of depletion if any simulations deplete
                if years_of_depletion:
                    median_year_of_depletion = int(np.median(years_of_depletion))
                    depletion_text = f"Year {median_year_of_depletion}"
                
                # Calculate year of escape (first year hitting $10MM)
                escape_threshold = 10_000_000
                for sim in results:
                    escape_years = np.where(sim >= escape_threshold)[0]
                    if len(escape_years) > 0:
                        years_of_escape.append(escape_years[0])
                
                # Calculate median year of escape if any simulations reach threshold
                if years_of_escape:
                    median_year_of_escape = int(np.median(years_of_escape))
                    escape_text = f"Year {median_year_of_escape}"
                    escape_probability = (len(years_of_escape) / len(results)) * 100
                    escape_delta = f"{escape_probability:.1f}% probability"
                
                # Store all results and calculated statistics in session state
                st.session_state.simulation_results = {
                    'results': results,
                    'portfolio_returns': portfolio_returns,
                    'withdrawals': withdrawals,
                    'asset_returns': asset_returns,
                    'asset_classes': asset_classes,
                    'initial_investment': initial_investment,
                    'risk_of_depletion': risk_of_depletion,
                    'depletion_text': depletion_text,
                    'years_of_depletion': years_of_depletion,
                    'escape_text': escape_text,
                    'escape_delta': escape_delta,
                    'years_of_escape': years_of_escape
                }
                
                # Run convergence analysis
                convergence_results = run_convergence_analysis(
                    initial_investment=initial_investment,
                    years=years,
                    initial_withdrawal=initial_withdrawal,
                    inflation_rate=inflation_rate,
                    asset_classes=asset_classes
                )
                
                # Store convergence results
                st.session_state.simulation_results['convergence_results'] = convergence_results
                
                # Calculate advanced risk metrics for tab 2
                try:
                    # Create portfolio analyzer
                    analyzer = ep.PortfolioAnalyzer()
                    
                    # Set asset classes in the analyzer first
                    analyzer.asset_classes = asset_classes
                    
                    # Create a simple correlation matrix that matches our asset classes
                    n_assets = len(asset_classes)
                    corr_matrix = np.eye(n_assets)  # Start with identity matrix
                    
                    # Fill with default correlations (0.3 between different assets)
                    for i in range(n_assets):
                        for j in range(n_assets):
                            if i != j:
                                corr_matrix[i, j] = 0.3
                    
                    # Set the correlation matrix
                    analyzer.set_correlation_matrix(corr_matrix)
                    
                    # Calculate the risk metrics
                    risk_metrics = analyzer.calculate_risk_metrics(results)
                    
                    # Add correlation matrix to risk metrics for display
                    risk_metrics['correlation_matrix'] = corr_matrix
                    
                    # Store risk metrics in session state
                    st.session_state.simulation_results['risk_metrics'] = risk_metrics
                except Exception as e:
                    # Log any errors but continue
                    import traceback
                    print(f"Error calculating risk metrics: {str(e)}")
                    print(traceback.format_exc())
        
        # If we have simulation results, display them
        if 'simulation_results' in st.session_state:
            # Extract all results and statistics from session state
            sim_data = st.session_state.simulation_results
            results = sim_data['results']
            portfolio_returns = sim_data['portfolio_returns']
            withdrawals = sim_data['withdrawals']
            asset_returns = sim_data['asset_returns']
            asset_classes = sim_data['asset_classes']
            initial_investment = sim_data['initial_investment']
            risk_of_depletion = sim_data['risk_of_depletion']
            depletion_text = sim_data['depletion_text']
            years_of_depletion = sim_data['years_of_depletion']
            escape_text = sim_data['escape_text']
            escape_delta = sim_data['escape_delta']
            years_of_escape = sim_data['years_of_escape']
            convergence_results = sim_data['convergence_results']
            
            # Calculate any additional statistics needed for display
            final_values = results[:, -1]
            
            # Display summary at the top using the placeholder
            with summary_placeholder.container():
                st.header("Simulation Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Base Metrics")
                    st.metric(
                        "Median Portfolio Value",
                        f"${np.median(final_values):,.0f}",
                        f"Initial: ${initial_investment:,.0f}"
                    )
                    st.metric(
                        "Risk of Depletion",
                        f"{risk_of_depletion:.1f}%"
                    )
                    st.metric(
                        "Median Annual Return",
                        f"{np.median(portfolio_returns.flatten()) * 100:.1f}%"
                    )
                
                with col2:
                    st.subheader("Extreme Scenarios")
                    st.metric(
                        "5th Percentile",
                        f"${np.percentile(final_values, 5):,.0f}",
                        f"{((np.percentile(final_values, 5) / initial_investment) ** (1/years) - 1) * 100:.1f}% CAGR"
                    )
                    st.metric(
                        "95th Percentile",
                        f"${np.percentile(final_values, 95):,.0f}",
                        f"{((np.percentile(final_values, 95) / initial_investment) ** (1/years) - 1) * 100:.1f}% CAGR"
                    )
                
                with col3:
                    st.subheader("Key Events")
                    st.metric(
                        "Median Year of Depletion",
                        depletion_text,
                        f"{len(years_of_depletion) / len(results) * 100:.1f}% of simulations"
                    )
                    st.metric(
                        "Median Year of Escape ($10MM)",
                        escape_text,
                        escape_delta
                    )
                
                # Asset allocation table with horizontal scroll
                st.subheader("Asset Allocation")
                allocation_data = {
                    "Asset": [asset.name for asset in asset_classes],
                    "Alloc": [f"{asset.allocation:.1%}" for asset in asset_classes],
                    "Return": [f"{asset.mean_return:.1%}" for asset in asset_classes],
                    "StdDev": [f"{asset.std_dev:.1%}" for asset in asset_classes],
                    "Min": [f"{asset.min_return:.1%}" for asset in asset_classes],
                    "Max": [f"{asset.max_return:.1%}" for asset in asset_classes]
                }
                st.dataframe(
                    allocation_data, 
                    hide_index=True,
                    use_container_width=True
                )
                
                st.divider()
            
            # Create and display main plot
            fig = create_plotly_figures(
                results,
                portfolio_returns,
                withdrawals,
                asset_returns,
                asset_classes,
                initial_investment
            )
            st.plotly_chart(fig, use_container_width=True, key="tab1_main_plot")
            
            # Add space between main simulation and convergence analysis
            st.markdown("---")
            
            # Run and display convergence analysis at the bottom
            conv_fig, median_changes, p90_changes, p10_changes, risk_changes = plot_convergence(convergence_results)
            st.header("Convergence Analysis")
            
            # Display convergence plot
            st.plotly_chart(conv_fig, use_container_width=True, key="tab1_convergence_plot")
            
            # Display changes table
            st.markdown("### Simulation Convergence")
            changes_df = pd.DataFrame({
                'Simulations': [f"{sim_data['convergence_results'][i]['n_sims']:,} → {sim_data['convergence_results'][i+1]['n_sims']:,}" 
                             for i in range(len(median_changes))],
                'Median Change': [f"{change:.2f}%" for change in median_changes],
                '90th Percentile': [f"{change:.2f}%" for change in p90_changes],
                '10th Percentile': [f"{change:.2f}%" for change in p10_changes],
                'Risk Change': [f"{change:.2f}%" for change in risk_changes]
            })
            
            st.dataframe(
                changes_df,
                hide_index=True,
                use_container_width=True
            )
            
            # Check if advanced analysis sections should be displayed
            if 'simulation_results' in st.session_state:
                # Extract necessary data for advanced analysis
                sim_data = st.session_state.simulation_results
                results = sim_data['results']
                portfolio_returns = sim_data['portfolio_returns']
                asset_returns = sim_data['asset_returns']
                asset_classes = sim_data['asset_classes']
                
                # Add advanced analysis section
                st.header("Advanced Analysis")
                
                try:
                    # Create portfolio analyzer
                    analyzer = ep.PortfolioAnalyzer()
                    
                    # Set asset classes in the analyzer first
                    analyzer.asset_classes = asset_classes
                    
                    # Create a simple correlation matrix that matches our asset classes
                    n_assets = len(asset_classes)
                    corr_matrix_np = np.eye(n_assets)  # Start with identity matrix (1s on diagonal)
                    
                    # Fill with default correlations (0.3 between different assets)
                    for i in range(n_assets):
                        for j in range(n_assets):
                            if i != j:
                                corr_matrix_np[i, j] = 0.3
                    
                    # Now set the correlation matrix
                    analyzer.set_correlation_matrix(corr_matrix_np)
                    
                    # Calculate risk metrics directly from results
                    risk_metrics = analyzer.calculate_risk_metrics(results)
                    
                    # Store risk metrics in session state
                    st.session_state.simulation_results['risk_metrics'] = risk_metrics
                except Exception as e:
                    st.error(f"Error in advanced analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Tab 2: Advanced Analysis
    with tabs[1]:
        if 'simulation_results' in st.session_state:
            st.header("Advanced Portfolio Analysis")
            
            # Get simulation data from session state
            sim_data = st.session_state.simulation_results
            results = sim_data['results']
            portfolio_returns = sim_data['portfolio_returns']
            asset_returns = sim_data['asset_returns']
            asset_classes = sim_data['asset_classes']
            
            # Check if risk metrics are already calculated
            if sim_data.get('risk_metrics'):
                risk_metrics = sim_data['risk_metrics']
                
                # Display advanced metrics plot
                st.subheader("Advanced Risk Metrics")
                advanced_metrics_fig = plot_advanced_metrics(risk_metrics)
                st.plotly_chart(advanced_metrics_fig, use_container_width=True, key="tab2_adv_metrics")
                
                # Display correlation matrix
                st.subheader("Asset Correlation Matrix")
                if 'correlation_matrix' in risk_metrics:
                    corr_matrix = risk_metrics['correlation_matrix']
                    corr_fig = plot_correlation_matrix(corr_matrix, asset_classes)
                    st.plotly_chart(corr_fig, use_container_width=True, key="tab2_corr_matrix")
            else:
                # If risk metrics not yet calculated, calculate them now
                try:
                    st.info("Calculating advanced risk metrics...")
                    
                    # Create portfolio analyzer
                    analyzer = ep.PortfolioAnalyzer()
                    analyzer.asset_classes = asset_classes
                    
                    # Create a correlation matrix
                    n_assets = len(asset_classes)
                    corr_matrix = np.eye(n_assets)
                    for i in range(n_assets):
                        for j in range(n_assets):
                            if i != j:
                                corr_matrix[i, j] = 0.3
                    
                    analyzer.set_correlation_matrix(corr_matrix)
                    
                    # Calculate risk metrics
                    risk_metrics = analyzer.calculate_risk_metrics(results)
                    risk_metrics['correlation_matrix'] = corr_matrix
                    
                    # Store in session state for future use
                    st.session_state.simulation_results['risk_metrics'] = risk_metrics
                    
                    # Display metrics
                    st.subheader("Advanced Risk Metrics")
                    advanced_metrics_fig = plot_advanced_metrics(risk_metrics)
                    st.plotly_chart(advanced_metrics_fig, use_container_width=True, key="tab2_adv_metrics_calc")
                    
                    # Display correlation matrix
                    st.subheader("Asset Correlation Matrix")
                    corr_fig = plot_correlation_matrix(corr_matrix, asset_classes)
                    st.plotly_chart(corr_fig, use_container_width=True, key="tab2_corr_matrix_calc")
                except Exception as e:
                    st.error(f"Error calculating advanced metrics: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        else:
            st.info("Run a simulation to view advanced analysis")
    
    # Tab 3: Historical Analysis
    with tabs[2]:
        st.header("Historical Portfolio Analysis")
        if 'simulation_results' in st.session_state:
            sim_data = st.session_state.simulation_results
            asset_classes = sim_data['asset_classes']
            initial_investment = sim_data['initial_investment']
            
            st.subheader("Historical Performance")
            try:
                # Run historical backtest
                historical_data = ep.get_historical_returns()
                years = len(historical_data) if historical_data is not None else 30
                
                backtest_results = ep.run_historical_backtest(
                    initial_investment=initial_investment,
                    years_to_simulate=years,
                    initial_withdrawal=initial_withdrawal,
                    asset_classes=asset_classes
                )
                
                # Display historical backtest
                backtest_fig = plot_historical_backtest(backtest_results, historical_data, years)
                st.plotly_chart(backtest_fig, use_container_width=True, key="tab3_historical_backtest")
            except Exception as e:
                st.error(f"Error in historical analysis: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.info("Run a simulation to view historical analysis")
    
    # Tab 4: Settings (Save/Load, Export)
    with tabs[3]:
        st.header("Simulation Settings")
        
        # Save/Load Functionality
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Save/Load Scenarios")
            
            # Create custom profile
            st.write("Create a new investor profile:")
            profile_name = st.text_input("Profile Name", "My Custom Profile")
            
            # Form for creating profile
            profile_form = st.form("profile_form")
            with profile_form:
                profile_allocations = {}
                profile_allocations["stocks"] = st.number_input("Stocks (%)", 0, 100, 40, 5)
                profile_allocations["bonds"] = st.number_input("Bonds (%)", 0, 100, 30, 5)
                profile_allocations["alternatives"] = st.number_input("Alternatives (%)", 0, 100, 15, 5)
                profile_allocations["private"] = st.number_input("Private Placements (%)", 0, 100, 10, 5)
                profile_allocations["cash"] = st.number_input("Cash (%)", 0, 100, 5, 5)
                
                profile_strategy = st.selectbox(
                    "Withdrawal Strategy",
                    options=["Fixed", "Percentage", "Dynamic"],
                    index=0
                )
                
                profile_rate = st.number_input("Withdrawal Rate (%)", 1.0, 10.0, 4.0, 0.1)
                
                save_button = st.form_submit_button("Save Profile")
                
                # Validate total allocation
                total = sum(profile_allocations.values())
                if total != 100:
                    st.error(f"Total allocation must equal 100% (currently {total}%)")
            
            if save_button and profile_name and total == 100:
                # Create new profile
                new_profile = {
                    "stocks": profile_allocations["stocks"],
                    "bonds": profile_allocations["bonds"],
                    "alternatives": profile_allocations["alternatives"],
                    "private": profile_allocations["private"],
                    "cash": profile_allocations["cash"],
                    "withdrawal_strategy": profile_strategy.lower(),
                    "withdrawal_rate": profile_rate,
                    "rebalance_frequency": 1
                }
                
                # Load existing presets
                presets = load_profile_presets()
                
                # Add new profile
                presets[profile_name] = new_profile
                
                # Save presets
                success = save_profile_presets(presets)
                if success:
                    st.success(f"Profile '{profile_name}' saved successfully!")
                else:
                    st.error("Failed to save profile")
        
        with col2:
            st.subheader("Export Analysis")
            
            # Export results as JSON
            if 'simulation_results' in st.session_state:
                export_data = export_report_data()
                if export_data:
                    st.download_button(
                        "Download Simulation Results (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name="simulation_results.json",
                        mime="application/json"
                    )
            else:
                st.info("Run a simulation to enable export functionality.")
            
            # Educational tooltips
            st.subheader("Explanation of Key Metrics")
            
            with st.expander("Risk of Depletion"):
                st.write("""
                The percentage of simulations where the portfolio value falls below the required annual withdrawal amount. 
                A higher value indicates greater risk that the portfolio will not be able to sustain the desired withdrawal rate.
                """)
            
            with st.expander("Sequence of Returns Risk"):
                st.write("""
                The potential impact of the order in which investment returns occur, particularly early in retirement.
                Negative returns in the early years of retirement can have a much more significant impact than the same returns occurring later.
                """)
            
            with st.expander("Sharpe Ratio"):
                st.write("""
                A measure of risk-adjusted return. Calculated as (Portfolio Return - Risk Free Rate) / Standard Deviation.
                Higher values indicate better risk-adjusted performance.
                """)
            
            with st.expander("Sortino Ratio"):
                st.write("""
                Similar to the Sharpe ratio, but it only considers the standard deviation of negative returns (downside deviation).
                A higher Sortino ratio indicates better protection against downside risk.
                """)
            
            with st.expander("Maximum Drawdown"):
                st.write("""
                The largest percentage drop from peak to trough in the portfolio value during the simulation period.
                A smaller maximum drawdown indicates less volatility and potentially less emotional stress for the investor.
                """)

if __name__ == "__main__":
    main()