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

def create_plotly_figures(results, portfolio_returns, withdrawals, asset_returns, asset_classes, initial_investment):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Asset Class Returns Distribution',
            'Portfolio Return Distribution',
            'Annual Withdrawals Over Time'
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
            name='Annual Withdrawal',
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
        elif 'Annual Withdrawal' in trace.name:
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
    
    # Calculate relative changes
    def calc_relative_change(values):
        changes = []
        for i in range(1, len(values)):
            changes.append(abs((values[i] - values[i-1]) / values[i-1]) * 100)
        return changes
    
    median_changes = calc_relative_change(medians)
    p90_changes = calc_relative_change(p90)
    p10_changes = calc_relative_change(p10)
    risk_changes = calc_relative_change(risk)
    
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
    """Plot advanced risk metrics"""
    # Create two separate figures - one for bar charts and one for the indicator
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sharpe Ratio',
            'Sortino Ratio',
            'Maximum Drawdown',
            'Sequence Risk Impact'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Plot Sharpe Ratio
    sharpe = metrics['sharpe_ratio']
    fig1.add_trace(
        go.Bar(
            x=['Mean', 'Median', '10th Percentile', '90th Percentile'],
            y=[sharpe['mean'], sharpe['median'], sharpe['p10'], sharpe['p90']],
            marker_color=['#636EFA', '#636EFA', '#EF553B', '#00CC96'],
            text=[f"{val:.2f}" for val in [sharpe['mean'], sharpe['median'], sharpe['p10'], sharpe['p90']]],
            textposition='auto',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot Sortino Ratio
    sortino = metrics['sortino_ratio']
    fig1.add_trace(
        go.Bar(
            x=['Mean', 'Median', '10th Percentile', '90th Percentile'],
            y=[sortino['mean'], sortino['median'], sortino['p10'], sortino['p90']],
            marker_color=['#636EFA', '#636EFA', '#EF553B', '#00CC96'],
            text=[f"{val:.2f}" for val in [sortino['mean'], sortino['median'], sortino['p10'], sortino['p90']]],
            textposition='auto',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot Maximum Drawdown
    drawdown = metrics['max_drawdown']
    fig1.add_trace(
        go.Bar(
            x=['Mean', 'Median', '10th Percentile', '90th Percentile'],
            y=[drawdown['mean'], drawdown['median'], drawdown['p10'], drawdown['p90']],
            marker_color=['#636EFA', '#636EFA', '#00CC96', '#EF553B'],  # Reversed colors for drawdown (lower is better)
            text=[f"{val:.1%}" for val in [drawdown['mean'], drawdown['median'], drawdown['p10'], drawdown['p90']]],
            textposition='auto',
            hovertemplate='%{x}: %{y:.1%}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot Sequence Risk as a simple bar chart instead of an indicator
    sequence = metrics['sequence_risk']
    fig1.add_trace(
        go.Bar(
            x=['Impact', 'Probability'],
            y=[sequence['impact'] * 100, sequence['probability'] * 100],
            marker_color=['#EF553B' if sequence['impact'] > 0 else '#00CC96', '#636EFA'],
            text=[f"{sequence['impact']*100:.1f}%", f"{sequence['probability']*100:.1f}%"],
            textposition='auto',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig1.update_layout(
        height=700,
        title_text="Advanced Risk Metrics",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(size=14)
    )
    
    return fig1

def plot_correlation_matrix(correlation_matrix, asset_classes):
    """Create a heatmap visualization of the correlation matrix"""
    # Extract asset names
    asset_names = [asset.name for asset in asset_classes]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=asset_names,
        y=asset_names,
        colorscale='RdBu_r',  # Red-blue diverging colorscale
        zmin=-1,
        zmax=1,
        text=[[f"{correlation_matrix[i][j]:.2f}" for j in range(len(asset_names))] for i in range(len(asset_names))],
        hovertemplate='%{y} × %{x}: %{text}<extra></extra>',
        colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        title="Asset Class Correlation Matrix",
        height=500,
        width=600,
        template="plotly_white",
        font=dict(size=14)
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
    file_path = "profile_presets.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return default_presets
    else:
        return default_presets

def save_profile_presets(presets):
    """Save profile presets to file"""
    file_path = "profile_presets.json"
    try:
        with open(file_path, 'w') as f:
            json.dump(presets, f, indent=2)
        return True
    except:
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
        format="%d"
    )
    
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
        
        stocks_return = st.number_input("Expected Return (%)", -10, 30, 10, step=1, key="stocks_return")
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
        value=30
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
    
    # Withdrawal Strategy
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
                
                # Determine which simulation method to use
                if use_correlation and withdrawal_strategy.lower() != "fixed" or rebalance_frequency > 0:
                    # Use enhanced portfolio for advanced features
                    analyzer = PortfolioAnalyzer()
                    analyzer.asset_classes = asset_classes
                    
                    # Set correlation matrix if requested
                    if use_correlation:
                        analyzer.set_correlation_matrix(create_default_correlation_matrix())
                    else:
                        analyzer.set_correlation_matrix()  # Identity matrix (no correlation)
                    
                    # Generate portfolio with rebalancing
                    portfolio_values, asset_values, asset_returns = analyzer.calculate_rebalanced_portfolio(
                        initial_investment=initial_investment,
                        years=years,
                        num_simulations=num_simulations,
                        initial_allocation=asset_classes,
                        rebalance_frequency=rebalance_frequency,
                        rebalance_threshold=rebalance_threshold/100
                    )
                    
                    # Apply withdrawal strategy
                    if withdrawal_strategy.lower() == "percentage":
                        withdrawal_amount = withdrawal_rate / 100
                    else:  # dynamic or fixed
                        withdrawal_amount = initial_withdrawal
                    
                    results, withdrawals_array = analyzer.apply_withdrawal_strategy(
                        portfolio_values=portfolio_values,
                        withdrawal_strategy=withdrawal_strategy.lower(),
                        initial_withdrawal=withdrawal_amount,
                        inflation_rate=inflation_rate
                    )
                    
                    # Format withdrawals for plotting
                    if withdrawal_strategy.lower() == "fixed":
                        withdrawals = np.array([initial_withdrawal * (1 + inflation_rate)**year 
                                             for year in range(years)])
                    else:
                        # Use median withdrawal for each year
                        withdrawals = np.median(withdrawals_array, axis=0)
                    
                    # Calculate portfolio returns
                    portfolio_returns = np.zeros((num_simulations, years))
                    for i in range(years):
                        if i == 0:
                            prev_value = initial_investment
                        else:
                            prev_value = portfolio_values[:, i]
                        portfolio_returns[:, i] = (portfolio_values[:, i+1] + withdrawals_array[:, i]) / prev_value - 1
                    
                    # Calculate risk metrics
                    risk_metrics = analyzer.calculate_risk_metrics(portfolio_values, withdrawals_array)
                    
                else:
                    # Use standard simulation for basic features
                    results, portfolio_returns, withdrawals, asset_returns, _ = run_simulation(
                        initial_investment=initial_investment,
                        years=years,
                        num_simulations=num_simulations,
                        initial_withdrawal=initial_withdrawal,
                        inflation_rate=inflation_rate,
                        asset_classes=asset_classes
                    )
                    risk_metrics = None
                
                # Run convergence analysis
                convergence_results = run_convergence_analysis(
                    initial_investment=initial_investment,
                    years=years,
                    initial_withdrawal=initial_withdrawal,
                    inflation_rate=inflation_rate,
                    asset_classes=asset_classes
                )
                
                # Store results in session state
                st.session_state.simulation_results = {
                    'results': results,
                    'portfolio_returns': portfolio_returns,
                    'withdrawals': withdrawals,
                    'asset_returns': asset_returns,
                    'asset_classes': asset_classes,
                    'initial_investment': initial_investment,
                    'convergence_results': convergence_results,
                    'risk_metrics': risk_metrics,
                    'rebalance_frequency': rebalance_frequency,
                    'withdrawal_strategy': withdrawal_strategy.lower()
                }
        
        # If we have simulation results, display them
        if 'simulation_results' in st.session_state:
            # Extract results from session state
            sim_data = st.session_state.simulation_results
            results = sim_data['results']
            portfolio_returns = sim_data['portfolio_returns']
            withdrawals = sim_data['withdrawals']
            asset_returns = sim_data['asset_returns']
            asset_classes = sim_data['asset_classes']
            initial_investment = sim_data['initial_investment']
            
            # Calculate and display statistics
            final_values = results[:, -1]
            risk_of_depletion = np.mean(final_values < withdrawals[-1]) * 100
            
            # Calculate year of depletion (first year portfolio hits zero)
            years_of_depletion = []
            for sim in results:
                depleted_years = np.where(sim <= 0)[0]
                if len(depleted_years) > 0:
                    years_of_depletion.append(depleted_years[0])
            
            # Calculate median year of depletion if it occurs
            if years_of_depletion:
                median_year_of_depletion = int(np.median(years_of_depletion))
                depletion_text = f"Year {median_year_of_depletion}"
            else:
                depletion_text = "Never"
            
            # Calculate year of escape (first year hitting $10MM)
            escape_threshold = 10_000_000
            years_of_escape = []
            for sim in results:
                escape_years = np.where(sim >= escape_threshold)[0]
                if len(escape_years) > 0:
                    years_of_escape.append(escape_years[0])
            
            # Calculate median year of escape if it occurs
            if years_of_escape:
                median_year_of_escape = int(np.median(years_of_escape))
                escape_text = f"Year {median_year_of_escape}"
                escape_probability = (len(years_of_escape) / len(results)) * 100
                escape_delta = f"{escape_probability:.1f}% probability"
            else:
                escape_text = "Never"
                escape_delta = "0% probability"
            
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Add space between main simulation and convergence analysis
            st.markdown("---")
            
            # Run and display convergence analysis at the bottom
            conv_fig, median_changes, p90_changes, p10_changes, risk_changes = plot_convergence(sim_data['convergence_results'])
            st.header("Convergence Analysis")
            
            # Display convergence plot
            st.plotly_chart(conv_fig, use_container_width=True)
            
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
    
    # Tab 2: Advanced Analysis
    with tabs[1]:
        if 'simulation_results' in st.session_state:
            # Add advanced risk metrics visualization if available
            if st.session_state.simulation_results.get('risk_metrics'):
                risk_metrics = st.session_state.simulation_results['risk_metrics']
                
                # Display advanced metrics plot
                st.header("Advanced Risk Metrics")
                st.plotly_chart(plot_advanced_metrics(risk_metrics), use_container_width=True)
            
            # Portfolio allocation heatmap
            st.header("Portfolio Allocation Analysis")
            st.write("Explore how different asset allocations affect portfolio outcomes.")
            
            # Input for allocation ranges
            col1, col2 = st.columns(2)
            with col1:
                stock_min = st.slider("Minimum Stock Allocation (%)", 0, 80, 10, 5)
                stock_max = st.slider("Maximum Stock Allocation (%)", stock_min, 100, min(stock_min + 50, 100), 5)
            with col2:
                bond_min = st.slider("Minimum Bond Allocation (%)", 0, 80, 10, 5)
                bond_max = st.slider("Maximum Bond Allocation (%)", bond_min, 100, min(bond_min + 50, 100), 5)
            
            # Button to generate heatmap
            generate_heatmap = st.button("Generate Allocation Heatmap")
            
            if generate_heatmap:
                with st.spinner("Generating allocation heatmap..."):
                    # Convert percentage to decimal
                    stock_range = (stock_min / 100, stock_max / 100)
                    bond_range = (bond_min / 100, bond_max / 100)
                    
                    # Generate heatmap data
                    stock_allocs, bond_allocs, median_values, sharpe_values, risk_values = create_allocation_heatmap(
                        stock_range, bond_range, st.session_state.simulation_results['asset_classes']
                    )
                    
                    # Display heatmaps in 2 columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            plot_allocation_heatmap(stock_allocs, bond_allocs, median_values, "Median Final Value ($M)"),
                            use_container_width=True
                        )
                    with col2:
                        st.plotly_chart(
                            plot_allocation_heatmap(stock_allocs, bond_allocs, sharpe_values, "Sharpe Ratio"),
                            use_container_width=True
                        )
            
            # Correlation matrix visualization
            st.header("Asset Correlation Analysis")
            if 'asset_classes' in st.session_state.simulation_results:
                assets = st.session_state.simulation_results['asset_classes']
                
                # Create default correlation matrix
                corr_matrix = create_default_correlation_matrix()
                
                # Display correlation matrix
                st.plotly_chart(
                    plot_correlation_matrix(corr_matrix, assets),
                    use_container_width=True
                )
                
                st.info("This correlation matrix shows the statistical relationship between the returns of different asset classes. A value of 1.0 means perfect correlation, 0 means no correlation, and -1.0 means perfect negative correlation.")
        else:
            st.info("Run a simulation to see advanced analysis.")
    
    # Tab 3: Historical Analysis
    with tabs[2]:
        st.header("Historical Backtest Analysis")
        st.write("Analyze how your portfolio would have performed using actual historical returns.")
        
        # Historical backtest parameters
        backtest_years = st.slider(
            "Backtest Years",
            min_value=5,
            max_value=30,
            value=20
        )
        
        backtest_strategy = st.selectbox(
            "Withdrawal Strategy",
            options=["Fixed", "Percentage", "Dynamic"],
            index=0,
            format_func=lambda x: {
                "Fixed": "Fixed (Inflation-Adjusted)",
                "Percentage": "Percentage of Portfolio",
                "Dynamic": "Dynamic (Market-Responsive)"
            }.get(x, x)
        )
        
        # Historical data display
        hist_data = get_historical_returns()
        
        # Option to run historical backtest
        run_backtest = st.button("Run Historical Backtest")
        
        if run_backtest and 'simulation_results' in st.session_state:
            with st.spinner("Running historical backtest..."):
                # Use asset classes from simulation
                asset_classes = st.session_state.simulation_results['asset_classes']
                initial_investment = st.session_state.simulation_results['initial_investment']
                
                # Determine initial withdrawal based on strategy
                if backtest_strategy.lower() == "fixed":
                    initial_withdrawal = st.session_state.simulation_results['withdrawals'][0]
                else:
                    # Use withdrawal rate from session state or default to 4%
                    rate = st.session_state.get('withdrawal_rate', 4.0) / 100
                    initial_withdrawal = rate
                
                # Run historical backtest
                backtest_results = run_historical_backtest(
                    initial_investment=initial_investment,
                    years_to_simulate=backtest_years,
                    initial_withdrawal=initial_withdrawal,
                    asset_classes=asset_classes,
                    withdrawal_strategy=backtest_strategy.lower()
                )
                
                # Display backtest results
                st.plotly_chart(
                    plot_historical_backtest(backtest_results, hist_data, backtest_years),
                    use_container_width=True
                )
                
                # Historical data table
                st.subheader("Historical Returns Data (Last 10 Years)")
                st.dataframe(
                    hist_data.sort_values('year', ascending=False).head(10).style.format({
                        'stocks': '{:.1%}',
                        'bonds': '{:.1%}',
                        'alternatives': '{:.1%}',
                        'private': '{:.1%}',
                        'cash': '{:.1%}'
                    }),
                    use_container_width=True
                )
        else:
            st.info("Run a simulation first, then run a historical backtest to compare.")
    
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